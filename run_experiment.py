import argparse
import os
from huggingface_hub import login
from transformers import set_seed
import autophagy_gen as autophagy 

# Configurazione ambiente
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def run_single_experiment():
    parser = argparse.ArgumentParser(description="Esegui una singola run della tesi.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--temp", type=float, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    
    args = parser.parse_args()
    HF_TOKEN = os.getenv("token_write")
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Errore: token_write non trovato nelle variabili d'ambiente.")
        return

    print(f"\n>>> AVVIO: {args.model_key} | Temp: {args.temp} | Run: {args.run_id} | Seed: {args.seed}")
    
    set_seed(args.seed)

    start_round = 0
    # resume_model_id = f"stefanocarrera/autophagycode_M_mercury_{model_key}_lr0.0001_c142_{args.strategy}_t{args.temp}_g{start_round}_{args.run_id}"

    try:
        autophagy.autophagy(
            base_model_id=args.model_id,
            data_lang="java",
            g=10,
            lr=1e-4,
            start_round=start_round,                     
            resume_model_id=None,
            temperature=args.temp,
            top_p=0.99,
            save_token_log=True,
            run_id=args.run_id,
            seed=args.seed,
        )
        print(f"\n>>> COMPLETATO: {args.model_key} (Temp: {args.temp})")
    except Exception as e:
        print(f"!!! ERRORE CRITICO: {e}")

if __name__ == '__main__':
    run_single_experiment()