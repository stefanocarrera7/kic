from datasets import load_dataset

# Carica l'intero DatasetDict (sia 'train' che 'test')
ds = load_dataset("stefanocarrera/kic_D_leetcode-cpp-raw")

# ========= ESTRAZIONE NATURAL LANGUAGE DESCRIPTION =========
def extract_prompt(example):
    text = example.get("content")
    if text is None:
        return {"prompt_nl": None}

    # Prende solo ciò che precede la prima occorrenza di "**Example 1"
    clean_text = text.split("**Example 1")[0].rstrip()
    return {"prompt_nl": clean_text}

# .map su DatasetDict agisce in automatico su ogni split
ds = ds.map(extract_prompt)

# ========= AGGIUNTA COLONNA PROMPT (C++) =========
def format_cpp_prompt(row):
    prompt_nl = row.get("prompt_nl")
    if not prompt_nl:
        return {"prompt": None}

    # Formattazione Doxygen/Javadoc indentata dentro class Solution
    lines = [
        f"\t * {line.strip()}" if line.strip() else "\t *"
        for line in prompt_nl.strip().splitlines()
    ]
    formatted_body = "\n".join(lines)

    prompt = (
        "Continue the following C++ code:\n"
        "class Solution {\n"
        "public:\n"
        "\t/**\n"
        f"{formatted_body}\n"
        "\t */\n"
    )
    return {"prompt": prompt}

ds = ds.map(format_cpp_prompt)

# ============================================
HF_USERNAME = "stefanocarrera"
repo_id = f"{HF_USERNAME}/kic_D_leetcode-cpp"

print(f"Caricamento su {repo_id} (Train: {len(ds['train'])}, Test: {len(ds['test'])})...")

# Push dell'intero DatasetDict con entrambi gli split
ds.push_to_hub(
    repo_id=repo_id,
    private=False
)

print("Caricamento completato!")