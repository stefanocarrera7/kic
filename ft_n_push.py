import torch
import shutil
import os
import gc
from finetuning import finetune_model

def finetune_n_push(base_model_id:str,
                    train_data,
                    lr:float,
                    seed:int,
                    data_lang:str,
                    temperature:float,
                    run_id):
        
        print("\nStarting finetuning...")
        
        ft_model, ft_tok = finetune_model(
            dataset = train_data,
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

        # --- Salvataggio su HF ---
        print("\nPushing to HuggingFace Hub...")
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)
        print(f"Pushed model (adapter) to {model_id}")

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

if __name__ == '__main__':
    finetune_n_push()