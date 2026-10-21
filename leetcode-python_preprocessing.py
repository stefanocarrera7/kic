from datasets import load_dataset

# Carica l'intero DatasetDict (sia 'train' che 'test')
ds = load_dataset("stefanocarrera/kic_D_leetcode-python-raw")

# ========= ESTRAZIONE NATURAL LANGUAGE DESCRIPTION =========
def extract_prompt(example):
    text = example.get("content")
    if text is None:
        return {"prompt_nl": None}

    clean_text = text.split("**Example 1")[0].rstrip()
    return {"prompt_nl": clean_text}

ds = ds.map(extract_prompt)

# ========= AGGIUNTA COLONNA PROMPT (PYTHON SENZA CLASSE) =========
def format_python_prompt(row):
    prompt_nl = row.get("prompt_nl")
    if not prompt_nl:
        return {"prompt": None}

    lines = [
        f"# {line.strip()}" if line.strip() else "#"
        for line in prompt_nl.strip().splitlines()
    ]
    formatted_body = "\n".join(lines)

    prompt = (
        "Continue the following Python code:\n"
        f"{formatted_body}\n"
    )
    return {"prompt": prompt}

ds = ds.map(format_python_prompt)

# ============================================
HF_USERNAME = "stefanocarrera"
repo_id = f"{HF_USERNAME}/kic_D_leetcode-python"

print(f"Caricamento su {repo_id} (Train: {len(ds['train'])}, Test: {len(ds['test'])})...")

ds.push_to_hub(
    repo_id=repo_id,
    private=False
)

print("Caricamento completato!")