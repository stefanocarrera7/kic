import json
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm


def generate_sample(
    data=None,
    model=None,
    tokenizer=None,
    n_solutions: int = 1,
    temperature: float = 1.0,
    max_new_t: int = 512,
    top_p: float = 0.95,
    save_token_log: bool = False,
    batch_size: int = 4,
    **kwargs
) -> Dataset:

    # Gestione robusta dell'ordine degli argomenti posizionali
    # Se il primo argomento è il modello anziché il dataset (es. autophagy_gen.py riga 59)
    if hasattr(data, "generate") and not isinstance(data, (list, dict, Dataset)):
        actual_model = data
        actual_tok = model
        actual_data = tokenizer
        model = actual_model
        tokenizer = actual_tok
        data = actual_data

    # Configurazione padding a sinistra per decoder-only in inferenza a batch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-processing dei prompt conforme alla versione originale
    processed_prompts = []
    for row in range(len(data)):
        prompt = "Solve the following task in Python:\n" + data[row]["content"]      ####################
        if prompt.endswith(".."):
            prompt = prompt[:-1] + "\n"
        if not prompt.endswith("\n"):
            prompt += "\n"
        processed_prompts.append(prompt)

    # Se n_solutions > 1, duplica i prompt per generare sequenze multiple
    expanded_indices = []
    expanded_prompts = []
    for idx, p in enumerate(processed_prompts):
        for _ in range(n_solutions):
            expanded_indices.append(idx)
            expanded_prompts.append(p)

    generated_completions = []
    generated_top_k = []

    # Inferenza a batch
    for i in tqdm(range(0, len(expanded_prompts), batch_size), desc="[GEN] Batch Processing"):
        batch_prompts = expanded_prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        batch_input_len = inputs.input_ids.shape[1]

        # Generazione accelerata
        with torch.inference_mode():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_t,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.0),
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=save_token_log,
            )

        sequences = gen_outputs.sequences

        # Calcolo logprob solo se richiesto save_token_log=True
        log_probs = None
        if save_token_log and gen_outputs.logits is not None:
            logits_stacked = torch.stack(gen_outputs.logits, dim=1)
            log_probs = F.log_softmax(logits_stacked, dim=-1)

        for b in range(len(batch_prompts)):
            gen_tokens = sequences[b, batch_input_len:]

            # Rimozione dei token di padding finali
            if tokenizer.pad_token_id is not None:
                valid_mask = gen_tokens != tokenizer.pad_token_id
                if valid_mask.any():
                    last_idx = valid_mask.nonzero()[-1].item() + 1
                    gen_tokens = gen_tokens[:last_idx]
                else:
                    gen_tokens = gen_tokens[:0]

            num_gen = len(gen_tokens)

            # Decodifica della completion (senza il prompt)
            decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generated_completions.append(decoded_text)

            # Costruzione di top_k_progression coerente con l'output originale
            if save_token_log and log_probs is not None and num_gen > 0:
                step_log_probs = log_probs[b, :num_gen, :]
                selected_log_probs = step_log_probs.gather(1, gen_tokens.unsqueeze(-1)).squeeze(-1).tolist()
                tokens_str = tokenizer.convert_ids_to_tokens(gen_tokens.tolist())
                
                token_log_entry = [
                    {"token": tok, "logprob": float(lp)}
                    for tok, lp in zip(tokens_str, selected_log_probs)
                ]
                generated_top_k.append(token_log_entry)
            else:
                generated_top_k.append(None)

    # Assemblaggio finale del dataset con le colonne originali esatte
    sample = []
    for idx_expanded, original_row_idx in enumerate(expanded_indices):
        orig_row = data[original_row_idx]
        top_k_val = generated_top_k[idx_expanded]
        
        sample.append({
            "id": orig_row["id"],
            "prompt": processed_prompts[original_row_idx],
            "completion": generated_completions[idx_expanded],
            "top_k_progression": json.dumps(top_k_val) if top_k_val is not None else None,
            "content": orig_row["content"],
            "difficulty": orig_row["difficulty"]
        })

    return Dataset.from_list(sample)