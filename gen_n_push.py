from datasets import load_dataset
from generate_sample_vllm import generate_sample
from vllm import LLM

def gen_n_push(data_id:str, model_id:str):

    ds = load_dataset(data_id, split = "train")

    llm = LLM(
    model=model_id,
    max_model_len=2048,
    dtype="float16",
    gpu_memory_utilization=0.8
    )

    out_data = generate_sample(ds, llm,
                    temperature = 1.0,
                    max_new_t = 512,
                    top_p = 0.95,
                    save_token_log = False,
                    k_logprobs = 5
    )

    out_data.push_to_hub(f"{data_id}_{model_id.replace('/', '_')}_vllm_gen", token=True)










if __name__ == '__main__':
    gen_n_push()