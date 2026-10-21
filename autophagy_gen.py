import gc
import os
import shutil
import torch
from generate_sample import generate_sample
from datasets import load_dataset
from unsloth import FastLanguageModel
from finetuning import finetune_model


def _sanitize_repo_name(text: str) -> str:
    """Sanitizza e accorcia il nome del repository per rispettare il limite di 96 char di HF."""
    # 1. Rimuove l'organizzazione (es. 'unsloth/')
    if "/" in text:
        text = text.split("/")[-1]
    
    # 2. Rimuove suffissi lunghi standard
    text = text.replace("-unsloth-bnb-4bit", "")
    text = text.replace("-bnb-4bit", "")
    text = text.replace("-Base", "")
    text = text.replace("-Instruct", "")
    
    return text.replace(" ", "_")


def autophagy_gen(
    base_model_id: str,
    data_lang: str,
    g: int = 10,
    lr: float = 1e-5,
    start_round: int = 0,                        
    resume_model_id: str = None,           
    temperature: float = 1,
    top_p: float = 0.95,
    save_token_log: bool = False,
    run_id: str = "",
    seed: int = 42
    ):

    # DATA
    train_data = load_dataset(f"stefanocarrera/kic_D_leetcode-{data_lang}", split = "train")
    test_data = load_dataset(f"stefanocarrera/kic_D_leetcode-{data_lang}", split = "test")

    base_tag = _sanitize_repo_name(base_model_id)

    prev_adapter_repo = resume_model_id

    # ================= MAIN LOOP ==========================
    for t in range(start_round, g+1):
        print(f"\n{'='*40}")
        print(f"   === Generation round {t}/{g} ===")
        print(f"{'='*40}")

        # --- Caricamento Modello ---
        current_model_id = prev_adapter_repo if prev_adapter_repo else base_model_id
        
        print(f"Loading model for generation via Unsloth: {current_model_id}")
        gen_model, gen_tok = FastLanguageModel.from_pretrained(
            model_name = current_model_id,
            max_seq_length = 2048,
            dtype = torch.float16,
            load_in_4bit = True,
            device_map = {"": 0},
        )              
        FastLanguageModel.for_inference(gen_model)


        # --- Generazione del dataset per il test ---  
        print(f"\nGenerating synthetic test set for generation {t}...")
        test_synth = generate_sample(test_data, gen_model, gen_tok,
                                        n_solutions=1,
                                        temperature=temperature, top_p=top_p,
                                        save_token_log=save_token_log)

        test_data_id = f"stefanocarrera/kic_D_test-{data_lang}_{base_tag}_t{temperature}_g{t}_{run_id}"
        test_synth.push_to_hub(test_data_id)

        if t == g:
            print("\nUltima generazione completata, Pipeline terminata.")
            break

        # --- Calcolo degli indici per il subset del round ---
        chunk_size = int(len(train_data) / g)
        start_idx = t * chunk_size
        end_idx = min((t + 1) * chunk_size, len(train_data)) # Impedisce l'Out of Bounds
        
        # Se start_idx supera la lunghezza del dataset, interrompiamo il ciclo
        if start_idx >= len(train_data):
            print("\nDati di training esauriti per il prossimo round. Interruzione della pipeline.")
            break
         
        current_subset = train_data.select(range(start_idx, end_idx))

        # --- Generazione del dataset sintetico per il train ---
        print(f"\nStarting sample generation for training (round {t})...")
        synth = generate_sample(current_subset,
                                gen_model, gen_tok,
                                n_solutions=1,
                                temperature=temperature, top_p=top_p,
                                save_token_log=save_token_log)


        # --- PULIZIA DELLA VRAM (PRE-TRAINING) ---
        print("\nPulizia della VRAM in corso prima del finetuning...")
        del gen_model
        del gen_tok
        gc.collect()
        torch.cuda.empty_cache()


        # --- Finetuning ---
        print("\nStarting finetuning...")
        ft_dir = f"runs/gen_{t:02d}/adapters"
        
        ft_model, ft_tok = finetune_model(
            dataset = synth,
            base_model_id = base_model_id,
            output_dir = ft_dir,
            num_train_epochs = 3,
            lr = lr,
            batch_size = 1,
            grad_accum = 16,
            pack_to_max=False,
            resume_adapter_repo = prev_adapter_repo,
            seed = seed
        )

        print("\nEnd Finetuning...")

        model_id = f"stefanocarrera/kic_M_{base_tag}_{data_lang}_t{temperature}_g{t+1}_{run_id}"
        data_id  = f"stefanocarrera/kic_D_train-{data_lang}_{base_tag}_t{temperature}_g{t}_{run_id}"

        # --- Salvataggio su HF ---
        print("\nPushing to HuggingFace Hub...")
        synth.push_to_hub(data_id)
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

        # Aggiorniamo i riferimenti per il prossimo giro
        prev_adapter_repo = model_id

        # Pulizia del disco
        print("\nCancellazione dei file locali dell'adapter per liberare spazio su disco...")
        if os.path.exists(ft_dir):
            shutil.rmtree(ft_dir)
            print(f"Cartella {ft_dir} eliminata con successo.")
        
        # --- 7. PULIZIA DELLA VRAM (POST-TRAINING) EXTREME ---
        print("\nPulizia della VRAM in corso prima del prossimo round di generazione...")
        
        ft_model.cpu()
        
        del ft_model
        del ft_tok
        
        gc.collect()
        gc.collect() 
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print("Memoria liberata con successo. VRAM attuale allocata:", torch.cuda.memory_allocated() / 1e9, "GB")
    
    print("\n[PIPELINE COMPLETATA]")
    return prev_adapter_repo