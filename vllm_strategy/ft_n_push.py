import argparse
import os
import shutil
import gc
import torch
from datasets import load_dataset
from finetuning import finetune_model
from utils import run_id_to_seed


def finetune_n_push(
    base_model_id: str,
    data_id: str,
    g: int,
    data_lang: str,
    temperature: float,
    run_id: int,
    base_tag: str,
    lr: float = 1e-5,
    prev_adapter_repo: str = None,
    merge_weights: bool = True
):
    print(f"\n--- [FT] Inizio Fine-Tuning - Generazione {g} ---")
    print(f"Modello Base: {base_model_id}")
    print(f"Dataset Input: {data_id}")

    ft_dir = f"./tmp_ft_checkpoint_g{g}_{run_id}"
    train_data = load_dataset(data_id, split="train")

    seed = run_id_to_seed(run_id)

    ft_model, ft_tok = finetune_model(
        dataset=train_data,
        base_model_id=base_model_id,
        output_dir=ft_dir,
        num_train_epochs=3,
        lr=lr,
        batch_size=1,
        grad_accum=16,
        pack_to_max=False,
        resume_adapter_repo=prev_adapter_repo,
        seed=seed
    )

    print("\nEnd Finetuning...")

    # Nomenclatura del checkpoint di output
    model_id = f"stefanocarrera/kic_M_{base_tag}_{data_lang}_t{temperature}_g{g}_{run_id}"

    print(f"\n[FT] Salvataggio e push su HuggingFace Hub ({model_id})...")

    # Fusione pesi per consentire la lettura diretta a vLLM nel venv successivo
    if merge_weights:
        if hasattr(ft_model, "push_to_hub_merged"):
            # Metodo nativo Unsloth FastLanguageModel
            ft_model.push_to_hub_merged(model_id, ft_tok, save_method="merged_16bit")
        elif hasattr(ft_model, "merge_and_unload"):
            # Metodo nativo Hugging Face PEFT
            merged_model = ft_model.merge_and_unload()
            merged_model.push_to_hub(model_id)
            ft_tok.push_to_hub(model_id)
            del merged_model
        else:
            ft_model.push_to_hub(model_id)
            ft_tok.push_to_hub(model_id)
    else:
        ft_model.push_to_hub(model_id)
        ft_tok.push_to_hub(model_id)

    print(f"Pushed model to {model_id}")

    # Pulizia del disco locale
    if os.path.exists(ft_dir):
        shutil.rmtree(ft_dir)
        print(f"Cartella temporanea {ft_dir} eliminata.")

    # Pulizia VRAM
    ft_model.cpu()
    del ft_model
    del ft_tok

    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memoria liberata con successo. VRAM allocata:", torch.cuda.memory_allocated() / 1e9, "GB\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Esegui fine-tuning e push del modello su Hub.")
    parser.add_argument("--base_model_id", type=str, required=True, help="Modello base o checkpoint precedente")
    parser.add_argument("--data_id", type=str, required=True, help="Hugging Face repo ID del dataset sintetico appena generato")
    parser.add_argument("--g", type=int, required=True, help="Indice della generazione corrente (t+1)")
    parser.add_argument("--data_lang", type=str, default="python", help="Linguaggio/tag del dataset")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura usata nella generazione")
    parser.add_argument("--run_id", type=int, required=True, help="ID progressivo numerico dell'esperimento/run")
    parser.add_argument("--base_tag", type=str, default="starcoder", help="Identificativo breve dell'architettura base")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--prev_adapter_repo", type=str, default=None, help="Eventuale adapter precedente da cui riprendere")
    parser.add_argument("--merge_weights", action="store_true", help="Fondi gli adapter LoRA nel modello base prima del push per vLLM")

    args = parser.parse_args()

    finetune_n_push(
        base_model_id=args.base_model_id,
        data_id=args.data_id,
        g=args.g,
        data_lang=args.data_lang,
        temperature=args.temperature,
        run_id=args.run_id,
        base_tag=args.base_tag,
        lr=args.lr,
        prev_adapter_repo=args.prev_adapter_repo,
        merge_weights=args.merge_weights
    )