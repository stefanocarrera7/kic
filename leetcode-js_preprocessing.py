from datasets import load_dataset

ds = load_dataset("stefanocarrera/kic_D_leetcode-javascript-raw", split = "train")

# ========= estrazione natural language description ==========
def extract_prompt(example):
    text = example['content']
    if text is None:
        return {'prompt_nl': None}
    
    # Prende solo ciò che precede la prima occorrenza di "**Example 1"
    clean_text = text.split("**Example 1")[0].rstrip()
    
    return {'prompt_nl': clean_text}

# Applica la modifica
ds = ds.map(extract_prompt)
# print("Esempio di prompt estratto:")
# print(ds[0]['prompt_nl'])  # Stampa il prompt estratto del primo
# ================================================================


# ========= aggiunta colonna prompt ==========
def format_jsdoc(text: str) -> str:
    # Gestisce ogni riga: aggiunge " *" se vuota/spazi, altrimenti " * " + riga
    formatted_body = "\n".join(
        f" * {line.strip()}" if line.strip() else " *"
        for line in text.strip().splitlines()
    )
    return f"/**\n{formatted_body}\n */\n"

# Aggiunge la nuova colonna 'prompt' (o 'full_prompt')
ds = ds.map(lambda row: {
    "prompt": format_jsdoc(row["prompt_nl"]) if row["prompt_nl"] is not None else None
})
# =============================================