# import os
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor
# from tqdm import tqdm
# from typing import Tuple, Dict
# from eval import JavaEvalPipeline

# def _worker_eval(args: Tuple[int, str, str, int]) -> Tuple[int, Dict]:
#     """
#     Funzione top-level isolata per il multiprocessing.
#     Riceve l'indice di riga, il testo della completion, la descrizione e il timeout.
#     """
#     row_idx, completion, description, timeout_sec = args
#     try:
#         if not isinstance(completion, str) or not completion.strip():
#             return row_idx, {
#                 "status": "EMPTY_COMPLETION", 
#                 "passed": False, 
#                 "details": "Empty code", 
#                 "extracted_code": ""
#             }
            
#         if not isinstance(description, str) or not description.strip():
#             return row_idx, {
#                 "status": "NO_DESCRIPTION", 
#                 "passed": False, 
#                 "details": "Empty prompt/description", 
#                 "extracted_code": ""
#             }

#         res = JavaEvalPipeline.evaluate(
#             llm_output=completion,
#             problem_description=description,
#             timeout_sec=timeout_sec
#         )
#         return row_idx, res
#     except Exception as e:
#         return row_idx, {
#             "status": "PIPELINE_EXCEPTION", 
#             "passed": False, 
#             "details": str(e), 
#             "extracted_code": ""
#         }


# def evaluate_dataset(
#     df: pd.DataFrame,
#     completion_col: str = "completion",
#     description_col: str = "content",
#     output_path: str = "results.csv",
#     num_workers: int = os.cpu_count(),
#     timeout_sec: int = 5
# ) -> pd.DataFrame:
#     """
#     Valuta l'intero DataFrame ed esporta un CSV contenente solo:
#     - completion: testo originale generato dall'LLM
#     - extracted_code: codice Java estratto
#     - result: 'PASSED' se superato, altrimenti il tipo di errore / dettaglio
#     """
#     print(f"Avvio valutazione su {len(df)} esempi usando {num_workers} core...")

#     tasks = [
#         (idx, row[completion_col], row[description_col], timeout_sec)
#         for idx, row in df.iterrows()
#     ]

#     results = {}
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for idx, res in tqdm(executor.map(_worker_eval, tasks), total=len(tasks), desc="Valutazione Java"):
#             results[idx] = res

#     # Costruzione dei risultati riga per riga
#     extracted_codes = []
#     final_outcomes = []
#     statuses = []

#     for i in df.index:
#         res = results.get(i, {})
#         status = res.get("status", "UNKNOWN")
#         passed = res.get("passed", False)
#         details = str(res.get("details", "")).strip()
#         code = res.get("extracted_code", "")

#         extracted_codes.append(code if code is not None else "")

#         if passed or status == "PASSED":
#             outcome = "PASSED"
#             statuses.append("PASSED")
#         else:
#             outcome = f"{status}: {details}" if details else status
#             statuses.append(status)
        
#         final_outcomes.append(outcome)

#     # DataFrame con solo le 3 colonne richieste per il CSV
#     df_out = pd.DataFrame({
#         "completion": df[completion_col],
#         "extracted_code": extracted_codes,
#         "result": final_outcomes
#     })

#     # Stampa metriche aggregate e breakdown errori a terminale
#     total = len(df_out)
#     passed_count = (df_out["result"] == "PASSED").sum()
#     status_counts = pd.Series(statuses).value_counts()

#     print("\n" + "="*50)
#     print("METRICHE DI VALUTAZIONE FINALI")
#     print("="*50)
#     print(f"Totale Esempi:            {total}")
#     print(f"Pass Rate (Pass@1):       {passed_count / total * 100:.2f}% ({passed_count}/{total})")
#     print("-" * 50)
#     print("DETTAGLIO PER CATEGORIA / TIPO DI ERRORE:")
#     for st, count in status_counts.items():
#         print(f"  - {st:<24} {count:>5} ({count / total * 100:>6.2f}%)")
#     print("="*50 + "\n")

#     # Salvataggio
#     if output_path.endswith(".parquet"):
#         df_out.to_parquet(output_path, index=False)
#     else:
#         df_out.to_csv(output_path, index=False)
        
#     print(f"Dataset salvato con successo in: {output_path}")
#     return df_out


# if __name__ == "__main__":
#     df = pd.read_parquet("hf://datasets/stefanocarrera/kic_D_test-java_Qwen3-4B_t0.5_g0_run0/data/train-00000-of-00001.parquet")

#     evaluated_df = evaluate_dataset(
#         df=df,
#         completion_col="completion",
#         description_col="content",
#         output_path="results.csv",
#         num_workers=4,
#         timeout_sec=5
#     )


from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from eval import PythonEvalPipeline

if __name__ == "__main__":

    ds = load_dataset("stefanocarrera/kic_D_test-python_Qwen3-4B_t0.5_g7_run0", split="train")
    ds = ds.shuffle().select([i for i in list(range(5))])

    for row in ds:
        print("LLM Output:\n")
        print(row["completion"])
        print("-"*100)
        print("EXTRACTED CODE:\n")
        print(PythonEvalPipeline.extract_function_code(row["completion"]))
        print("-"*100)
        print("EXTRACTED ENTRY:\n")
        print(PythonEvalPipeline.extract_entry(row["completion"]))
        print("="*100)
        print("\n")