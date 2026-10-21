import torch
from generate_solutions import generate_solutions
from datasets import load_dataset
from unsloth import FastLanguageModel

ds = load_dataset("stefanocarrera/kic_D_leetcode-java", split = "train")

model, tok = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
    device_map = {"": 0},
)              
FastLanguageModel.for_inference(model)

print("ID: ", ds[321]['id'])

solutions = generate_solutions(prompt=ds[321]['prompt'],
                               model=model,
                               tokenizer=tok,
                               temperature=0.1,
                               max_new_tokens=300, 
                               top_p=0.95, 
                               n_solutions=1, 
                               do_sample=True, 
                               save_token_log=False)

code_str = solutions[0][0] if isinstance(solutions[0], list) else solutions[0]
clean_code = code_str.replace('\u00a0', ' ').expandtabs(4)
print(clean_code)