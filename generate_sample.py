import json
from datasets import Dataset
from generate_solutions import generate_solutions


def generate_sample(data,
                    model,
                    tokenizer,
                    n_solutions:int = 1,
                    temperature: float = 1,
                    max_new_t = 300,
                    top_p: float = 0.95,
                    save_token_log: bool = False
                    ) -> Dataset:

    sample = []

    for row in range(len(data)):

        prompt = data[row]['prompt']

        if prompt.endswith('..'):
            prompt = prompt[:-1] + '\n'
        # Forza il ritorno a capo se non c'è già
        if not prompt.endswith('\n'):
            prompt += '\n'
        

        solutions, top_k_progs = generate_solutions(prompt, model, tokenizer,
                                       n_solutions=n_solutions, 
                                       max_new_tokens=max_new_t, 
                                       do_sample=True,
                                       temperature=temperature,
                                       top_p=top_p,
                                       save_token_log=save_token_log)

        # Aggiunta delle soluzioni
        for s in range(n_solutions):
            sample.append({
                            "id": data[row]["id"],
                            "prompt": prompt,
                            "completion": solutions[s],
                            "top_k_progression": json.dumps(top_k_progs[s]) if top_k_progs is not None else None,
                            "content": data[row]["content"],
                            "difficulty": data[row]["difficulty"]
                        })
        
        if (row+1) % 10 == 0:
            print(f"{row+1} / {len(data)} tasks processed.")

    return Dataset.from_list(sample)


