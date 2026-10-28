import argparse
import gc
import torch
from datasets import load_dataset
from generate_sample_vllm import generate_sample
from vllm import LLM
from utils import run_id_to_seed


def gen_n_push(data_lang: str,
               model_id: str,
               base_tag: str,
               temperature: float,
               g: int,
               tot_gen: int,
               run_id: int = 0,
               max_new_t: int = 512,
               top_p: float = 0.95):
    
    data_id = f"stefanocarrera/kic_D_leetcode-{data_lang}"

    print(f"\n--- [GEN] Inizio Generazione con vLLM ---")
    print(f"Dataset sorgente: {data_id}")
    print(f"Modello sorgente: {model_id}")
    print(f"Generazione: {g} | Run ID: {run_id} | Temp: {temperature}")

    # Inizializzazione istanza vLLM
    llm = LLM(
        model=model_id,
        quantization ="awq",
        max_model_len=2048,
        dtype="float16",
        gpu_memory_utilization=0.6,
        trust_remote_code=True
    )

    seed = run_id_to_seed(run_id)

    # --- Generazione di Test ---
    print("\n[GEN-TEST] Caricamento split di test...")
    ds_test = load_dataset(data_id, split="test")

    out_data_test = generate_sample(
        ds_test, 
        llm,
        temperature=temperature,
        max_new_t=max_new_t,
        top_p=top_p,
        seed=seed,
        save_token_log=True,
        k_logprobs=5
    )
    
    output_hub_id_test = f"stefanocarrera/kic_D_test_{data_lang}_{base_tag}_t{temperature}_g{g}_run{run_id}"
    print(f"[GEN-TEST] Pushing dataset generato su Hugging Face: {output_hub_id_test}")
    out_data_test.push_to_hub(output_hub_id_test, token=True)
    print(f"[GEN-TEST] Generazione completata con successo.\n")

    # salta la generazione di train se abbiamo raggiunto il limite di generazioni previste
    if g == tot_gen:
        del llm
        del out_data_train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # --- Generazione di Train ---
    print("[GEN-TRAIN] Caricamento split di train...")
    ds_train = load_dataset(data_id, split="train")

    # Calcolo degli indici per il subset del round
    chunk_size = int(len(ds_train) / tot_gen)
    start_idx = g * chunk_size

    if g == tot_gen - 1:
        end_idx = len(ds_train)
    else:
        end_idx = min((g + 1) * chunk_size, len(ds_train))

    current_subset = ds_train.select(range(start_idx, end_idx))

    out_data_train = generate_sample(
        current_subset, 
        llm,
        temperature=temperature,
        max_new_t=max_new_t,
        top_p=top_p,
        seed=seed,
        save_token_log=False,
        k_logprobs=5
    )

    output_hub_id_train = f"stefanocarrera/kic_D_train_{data_lang}_{base_tag}_t{temperature}_g{g}_run{run_id}"
    print(f"[GEN-TRAIN] Pushing dataset generato su Hugging Face: {output_hub_id_train}")
    out_data_train.push_to_hub(output_hub_id_train, token=True)
    print(f"[GEN-TRAIN] Generazione completata con successo.\n")

    del llm
    del out_data_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Esegui generazione campioni con vLLM (train e test) e push su Hub.")
    
    # Argomenti richiesti
    parser.add_argument("--data_lang", type=str, required=True, help="Linguaggio/estensione del dataset (es. python, cpp, java)")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face repo ID o path del modello")
    parser.add_argument("--base_tag", type=str, required=True, help="Tag del modello base")
    parser.add_argument("--temperature", type=float, required=True, help="Temperatura di sampling")
    parser.add_argument("--g", type=int, required=True, help="Indice della generazione corrente (es. 1, 2, 3...)")
    parser.add_argument("--tot_gen", type=int, required=True, help="Numero totale di generazioni previste")
    parser.add_argument("--run_id", type=int, required=True, help="ID identificativo dell'esperimento/run")

    # Argomenti opzionali con default
    parser.add_argument("--max_new_t", type=int, default=512, help="Massimo numero di nuovi token generati")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    args = parser.parse_args()

    gen_n_push(
        data_lang=args.data_lang,
        model_id=args.model_id,
        base_tag=args.base_tag,
        temperature=args.temperature,
        g=args.g,
        tot_gen=args.tot_gen,
        run_id=args.run_id,
        max_new_t=args.max_new_t,
        top_p=args.top_p
    )