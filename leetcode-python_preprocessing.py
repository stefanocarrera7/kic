from datasets import load_dataset

# Carica l'intero DatasetDict (sia 'train' che 'test')
ds = load_dataset("stefanocarrera/kic_D_leetcode-python-raw")

# ========= ESTRAZIONE NATURAL LANGUAGE DESCRIPTION =========
def extract_prompt(example):
    text = example.get("content")

    task_desc = text.split("**Example 1")[0].rstrip()
    entry = example.get("slug").replace("-", "_")

    prefix = "Write only the code. Do not include explanations, markdown text, or reasoning outside the code block.\n Solve the following task:\n"
    func_init = f"```python\ndef {entry}("

    prompt = prefix + task_desc + '\n' + func_init

    return {"prompt":prompt}

ds = ds.map(extract_prompt)


# ============================================
HF_USERNAME = "stefanocarrera"
repo_id = f"{HF_USERNAME}/kic_D_leetcode-python"

print(f"Caricamento su {repo_id} (Train: {len(ds['train'])}, Test: {len(ds['test'])})...")

ds.push_to_hub(
    repo_id=repo_id,
    private=False
)

print("Caricamento completato!")