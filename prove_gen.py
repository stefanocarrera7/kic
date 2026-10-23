from datasets import load_dataset
import time

ds = load_dataset("stefanocarrera/kic_D_leetcode-java", split = "train")
sub_ds = ds.select(range(10))


import torch
from unsloth import FastLanguageModel
from generate_sample import generate_sample

model, tok = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-1.5B",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
    device_map = {"": 0},
)              
FastLanguageModel.for_inference(model)

start_time = time.perf_counter()

result_dataset = generate_sample(sub_ds,
                               model=model,
                               tokenizer=tok,
                               temperature=0.1,
                               max_new_t=300, 
                               top_p=0.95, 
                               n_solutions=1,
                               save_token_log=False)

elapsed_time = time.perf_counter() - start_time

print(f"Tempo totale impiegato: {elapsed_time:.2f} secondi")

print(f"\nGenerazione completata! Righe generate: {len(result_dataset)}\n")
print(f"Tempo totale impiegato: {elapsed_time:.2f} secondi")
print(f"Tempo medio per campione: {elapsed_time / len(sub_ds):.2f} secondi/campione\n")

for i in range(min(3, len(result_dataset))):  # Stampa i primi 3 per controllo
    print(f"--- Esempio {i + 1} (ID: {result_dataset[i]['id']}) ---")
    print("\nCOMPLETION:")
    print(result_dataset[i]['completion'])




# #### vLLM #####
# from generate_sample_vllm import generate_sample

# from datasets import load_dataset
# from vllm import LLM
# # Se la funzione generate_sample si trova in un altro file (es. generate_sample.py), importala:
# # from generate_sample import generate_sample

# # 3. Inizializza il motore vLLM
# llm = LLM(
#     model="Qwen/Qwen2.5-Coder-1.5B",  # Sostituisci con il modello o repo che stai usando
#     max_model_len=2048,
#     dtype="float16",
#     gpu_memory_utilization=0.5
# )

# start_time = time.perf_counter()

# # 4. Esegui il test chiamando la funzione sui 10 elementi
# print("Avvio della generazione per 10 campioni...")
# result_dataset = generate_sample(
#     data=sub_ds,
#     llm=llm,
#     n_solutions=1,
#     temperature=0.1,
#     max_new_t=300,
#     top_p=0.95,
#     save_token_log=True  # Metti False se non vuoi salvare i log dei token e l'entropia
# )

# elapsed_time = time.perf_counter() - start_time

# # 5. Verifica i risultati
# print(f"\nGenerazione completata! Righe generate: {len(result_dataset)}\n")
# print(f"Tempo totale impiegato: {elapsed_time:.2f} secondi")
# print(f"Tempo medio per campione: {elapsed_time / len(sub_ds):.2f} secondi/campione\n")

# for i in range(min(3, len(result_dataset))):  # Stampa i primi 3 per controllo
#     print(f"--- Esempio {i + 1} (ID: {result_dataset[i]['id']}) ---")
#     print("\nCOMPLETION:")
#     print(result_dataset[i]['completion'])