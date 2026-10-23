import math
import json
from datasets import Dataset
from vllm import LLM, SamplingParams


def compute_step_entropy(top_logprobs_step: dict, epsilon: float = 1e-10) -> float:
    """Calcola l'entropia predittiva di Shannon sui top token: H = - sum(p * log2(p))."""
    entropy = 0.0
    for token_id, logprob_obj in top_logprobs_step.items():
        prob = math.exp(logprob_obj.logprob)
        if prob > 0:
            entropy -= prob * math.log2(prob + epsilon)
    return entropy


def generate_sample(
    data,
    llm: LLM,
    n_solutions: int = 1,
    temperature: float = 1.0,
    max_new_t: int = 512,
    top_p: float = 0.95,
    save_token_log: bool = False,
    k_logprobs: int = 5
) -> Dataset:
    
    # 1. Preprocessing e formattazione di tutti i prompt
    prepared_prompts = []
    for row in range(len(data)):
        prompt = data[row]['prompt']
        if prompt.endswith('..'):
            prompt = prompt[:-1] + '\n'
        if not prompt.endswith('\n'):
            prompt += '\n'
        prepared_prompts.append(prompt)

    # 2. Configurazione parametri vLLM
    sampling_params = SamplingParams(
        n=n_solutions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_t,
        min_tokens=3,
        logprobs=k_logprobs if save_token_log else None
    )

    # 3. Generazione batch su tutti i prompt in un'unica chiamata
    outputs = llm.generate(prepared_prompts, sampling_params)

    # 4. Parsing e assemblaggio del dataset
    sample = []
    for row, req_output in enumerate(outputs):
        cleaned_prompt = prepared_prompts[row]
        orig_row = data[row]

        for s, comp in enumerate(req_output.outputs):
            raw_solution = comp.text.strip()
            
            # Parsing metadati token / entropia se richiesto
            top_k_str = None
            if save_token_log and comp.logprobs is not None:
                step_dict = {}
                for step_idx, step_logprobs in enumerate(comp.logprobs):
                    step_tokens_list = []
                    for token_id, logp in step_logprobs.items():
                        prob = math.exp(logp.logprob)
                        token_str = logp.decoded_token if logp.decoded_token is not None else ""
                        step_tokens_list.append({
                            "token": token_str,
                            "prob": round(prob, 5)
                        })
                    
                    step_entropy = compute_step_entropy(step_logprobs)
                    step_dict[f"step_{step_idx}"] = {
                        "top_k": step_tokens_list,
                        "pred_entropy": round(step_entropy, 5)
                    }
                top_k_str = json.dumps(step_dict)

            sample.append({
                "id": orig_row["id"],
                "prompt": cleaned_prompt,
                "completion": raw_solution,
                "top_k_progression": top_k_str,
                "content": orig_row["content"],
                "difficulty": orig_row["difficulty"]
            })

    return Dataset.from_list(sample)