def run_id_to_seed(run_id: int) -> int:
    return 100 + ((run_id * 563 + 127) % 900)