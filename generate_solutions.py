import re
import ast
import torch
from transformers import LogitsProcessor, LogitsProcessorList

class FP16OverflowClamper(LogitsProcessor):
    """
    Versione ultra-robusta per Tesla T4:
    1. Rimuove NaN e li sostituisce con valori neutri.
    2. Sostituisce Inf con i limiti massimi.
    3. Applica un clamp più stringente.
    """
    def __call__(self, input_ids, scores):
        # Sostituisce NaN con 0.0 e +/- Inf con +/- 100.0
        scores.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
        # Clamp finale di sicurezza
        scores.clamp_(-100.0, 100.0)
        return scores
    
class Float32LogitsProcessor(LogitsProcessor):
    """ Forza i logit in float32 prima del sampling per stabilità """
    def __call__(self, input_ids, scores):
        return scores.to(torch.float32)



def compute_predictive_entropy_batch(probs):
    """
    Calcola l'Entropia di Shannon Predittiva per un batch di probabilità.
    probs shape: (n_solutions, vocab_size)
    """
    epsilon = 1e-10

    entropy = -torch.sum(probs * torch.log2(probs + epsilon), dim=-1)
    
    return entropy


def get_top_k_predictions(scores_tuple, tokenizer, n_solutions, k=5):
    """
    Estrae i Top-K token e le loro probabilità per ogni step di generazione.
    Calcola anche l'entropia predittiva (incertezza) del modello a ogni step.
    """
    progressions = [{} for _ in range(n_solutions)]
    
    for step_idx, step_logits in enumerate(scores_tuple):
        
        probs = torch.softmax(step_logits, dim=-1)
        step_entropies = compute_predictive_entropy_batch(probs)
        
        # Otteniamo top K per TUTTE le n_solutions simultaneamente
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
        
        for sol_idx in range(n_solutions):
            step_tokens_list = []
            for i in range(k):
                token_id = top_indices[sol_idx, i].item()
                prob = top_probs[sol_idx, i].item()
                
                # Decodifica il token
                token_str = tokenizer.convert_ids_to_tokens(token_id)
                # Formattazione per pulire eventuali caratteri speciali (opzionale)
                if isinstance(token_str, bytes):
                    token_str = token_str.decode('utf-8', errors='ignore')
                    
                step_tokens_list.append({
                    "token": token_str,
                    "prob": round(prob, 5)
                })
            
            # dizionario per includere l'entropia
            progressions[sol_idx][f"step_{step_idx}"] = {
                "top_k": step_tokens_list,
                "pred_entropy": round(step_entropies[sol_idx].item(), 5)
            }
            
    return progressions



def generate_solutions(prompt: str,
                       model,
                       tokenizer,
                       temperature:float = 1.0,
                       max_new_tokens:int = 300,
                       top_p = 0.95,
                       n_solutions: int = 1,
                       do_sample: bool = True,
                       save_token_log: bool = False) -> list:

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_temperature = temperature if do_sample else None
    gen_top_p = top_p if do_sample else None

    # Iniettiamo i salvavita in ordine:
    # 1. Pulizia dei valori folli (NaN/Inf)
    # 2. Casting a Float32 per calcoli precisi nel sampling
    processors = LogitsProcessorList([
        FP16OverflowClamper(),
        Float32LogitsProcessor()
    ])

    return_dict_in_generate = False
    output_scores = False
    if save_token_log:
        return_dict_in_generate = True
        output_scores = True

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=3,
        temperature=gen_temperature,
        top_p=gen_top_p,
        do_sample=do_sample,
        num_return_sequences=n_solutions,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=processors,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores
    )
    
    if save_token_log:
        sequences = outputs.sequences
        scores_tuple = outputs.scores
    else:
        sequences = outputs

    # Decodifica del testo
    raw_solutions = [tokenizer.decode(out, skip_special_tokens=True) for out in sequences]
    final_solutions = [x.strip() for x in raw_solutions]
    
    if save_token_log:
        # Estrazione delle probabilità
        top_k_progs = get_top_k_predictions(scores_tuple, tokenizer, n_solutions=n_solutions, k=5)
    else:
        top_k_progs = None

    return final_solutions, top_k_progs