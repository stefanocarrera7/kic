from datasets import load_dataset

dataset_stream = load_dataset("stefanocarrera/kic_D_test-python_Qwen3-4B_t0.5_g7_run0", split="train", streaming=True)
ds = dataset_stream.take(10)

for row in ds:
    print(row["content"])