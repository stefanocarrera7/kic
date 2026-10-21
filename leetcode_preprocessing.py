from datasets import DatasetDict, load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

# ========== DATA LOADING ==========
ds = load_dataset("greengerong/leetcode", split="train")

# ========== NULL REPLACING ==========
ds = ds.map(
    lambda row: {
        "difficulty": "Medium" if row["difficulty"] is None else row["difficulty"]
    }
)

# ========== STRATIFIED SPLIT (2000 nel Train) ==========
TRAIN_SIZE = 2160
SEED = 42

labels = ds["difficulty"]
indices = list(range(len(ds)))

# Campionamento stratificato esatto su 'difficulty'
train_idx, test_idx = train_test_split(
    indices,
    train_size=TRAIN_SIZE,
    stratify=labels,
    random_state=SEED,
    shuffle=True,
)

train_ds = ds.select(train_idx)
test_ds = ds.select(test_idx)

# ========== ORDINAMENTO DEL TRAIN IN 10 BLOCCHI STRATIFICATI ==========
# Dividiamo i 2160 campioni del train in 10 fold stratificati
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
train_labels = np.array(train_ds["difficulty"])

reordered_train_idx = []
for fold_idx, (_, fold_test_indices) in enumerate(skf.split(np.zeros(len(train_ds)), train_labels)):
    # fold_test_indices contiene i 216 indici per questo blocco con quote bilanciate
    reordered_train_idx.extend(fold_test_indices.tolist())

# Riapplichiamo l'ordinamento a blocchi al train_ds
train_ds = train_ds.select(reordered_train_idx)

# Verifica della distribuzione sui 10 blocchi consecutivi
print("\nVerifica proporzioni sui 10 chunk consecutivi del Train (216 campioni ciascuno):")
chunk_size = len(train_ds) // 10
for i in range(10):
    chunk = train_ds.select(range(i * chunk_size, (i + 1) * chunk_size))
    counts = {d: chunk["difficulty"].count(d) for d in set(chunk["difficulty"])}
    pcts = {d: f"{c / chunk_size:.1%}" for d, c in counts.items()}
    print(f" - Blocco {i+1} [idx {i*chunk_size}:{(i+1)*chunk_size}]: {pcts}")


# ========== DATA PARTITION FOR CODING LANGUAGES ==========
lang_cols = {
    "python": "python",
    "java": "java",
    "cpp": "c++" if "c++" in ds.column_names else "cpp",
    "javascript": "javascript",
}
all_lang_cols = set(lang_cols.values())

HF_USERNAME = "stefanocarrera"

for lang, col_to_keep in lang_cols.items():
    cols_to_remove = [c for c in all_lang_cols if c != col_to_keep]

    # Preprocessing per entrambi gli split
    processed_splits = {}
    for split_name, current_ds in [("train", train_ds), ("test", test_ds)]:
        sub = current_ds.remove_columns(cols_to_remove)

        # Rimuove righe con codice nullo o vuoto per il linguaggio target
        sub = sub.filter(
            lambda row: row[col_to_keep] is not None
            and str(row[col_to_keep]).strip() != ""
        )
        processed_splits[split_name] = sub

    # Raggruppa in un DatasetDict standard HF con split 'train' e 'test'
    dataset_dict = DatasetDict(processed_splits)

    repo_id = f"{HF_USERNAME}/kic_D_leetcode-{lang}-raw"
    print(
        f"\nCaricamento su {repo_id} (Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])})..."
    )

    # Push su HF Hub
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=False,
    )

print("\nTutti i dataset sono stati processati, riordinati e caricati con successo!")