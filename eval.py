import re
import os
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional


class JavaEvalPipeline:

    @staticmethod
    def _extract_balanced_class_from_text(text: str) -> List[str]:
        """
        Scansiona il testo ed estrae TUTTI i blocchi 'class Solution { ... }' sintatticamente completi[cite: 2].
        """
        matches = list(re.finditer(r'(?:public\s+)?class\s+Solution\b', text))
        candidates = []

        for match in matches:
            start_idx = match.start()
            brace_count = 0
            found_first_brace = False
            in_string = False
            in_char = False
            in_single_comment = False
            in_multi_comment = False

            i = start_idx
            n = len(text)

            while i < n:
                ch = text[i]
                nxt = text[i + 1] if i + 1 < n else ''

                if in_single_comment:
                    if ch == '\n':
                        in_single_comment = False
                elif in_multi_comment:
                    if ch == '*' and nxt == '/':
                        in_multi_comment = False
                        i += 1
                elif in_string:
                    if ch == '\\':
                        i += 1
                    elif ch == '"':
                        in_string = False
                elif in_char:
                    if ch == '\\':
                        i += 1
                    elif ch == "'":
                        in_char = False
                else:
                    if ch == '/' and nxt == '/':
                        in_single_comment = True
                        i += 1
                    elif ch == '/' and nxt == '*':
                        in_multi_comment = True
                        i += 1
                    elif ch == '"':
                        in_string = True
                    elif ch == "'":
                        in_char = True
                    elif ch == '{':
                        brace_count += 1
                        found_first_brace = True
                    elif ch == '}':
                        brace_count -= 1
                        if found_first_brace and brace_count == 0:
                            code = text[start_idx:i + 1]
                            code = re.sub(r'^public\s+class\s+Solution', 'class Solution', code)
                            candidates.append(code)
                            break
                i += 1
        return candidates

    @staticmethod
    def _is_empty_or_stub(code: str) -> bool:
        """
        Verifica se la classe Solution contiene solo uno scheletro vuoto del prompt
        o metodi non implementati.
        """
        lower_code = code.lower()
        stub_markers = [
            "// todo",
            "// write your code",
            "// write code",
            "// your implementation",
            "// implement",
            "// your code"
        ]
        if any(marker in lower_code for marker in stub_markers):
            return True

        clean = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        clean = re.sub(r'//.*', '', clean)

        # Se contiene solo un main fittizio di esempio (es. riga 9 e 161) senza i metodi del problema
        if "public static void main" in clean and clean.count("public ") == 1:
            return True

        # Controlla se i metodi pubblici presenti hanno tutti un corpo vuoto o un return banale
        method_bodies = re.findall(r'public\s+(?!class\b)[\w<>\[\],\s]+?\s+\w+\s*\([^)]*\)\s*\{([^{}]*)\}', clean)
        if method_bodies:
            non_trivial = [
                b.strip() for b in method_bodies
                if b.strip() not in ['', ';', 'return 0;', 'return -1;', 'return null;', 'return false;', 'return true;', 'return "";', 'return 0.0;']
            ]
            if not non_trivial:
                return True

        return False

    @classmethod
    def extract_solution_code(cls, llm_output: str) -> Tuple[Optional[str], bool]:
        """
        Restituisce (codice_estratto, is_truncated).
        1. Cerca nei blocchi markdown ```java ... ```[cite: 2].
        2. Cerca nel testo grezzo scartando gli scheletri vuoti del prompt[cite: 2].
        """
        if not llm_output:
            return None, False

        # 1. Cerca nei blocchi markdown completi
        md_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', llm_output, flags=re.DOTALL)
        for block in reversed(md_blocks):
            candidates = cls._extract_balanced_class_from_text(block)
            for c in reversed(candidates):
                if not cls._is_empty_or_stub(c):
                    return c, False

        # 2. Cerca tutte le classi Solution bilanciate nell'intero testo
        all_candidates = cls._extract_balanced_class_from_text(llm_output)
        valid_candidates = [c for c in all_candidates if not cls._is_empty_or_stub(c)]

        if valid_candidates:
            # Restituisce l'ultima classe valida trovata (quella generata dal modello)
            return valid_candidates[-1], False

        # 3. Se ci sono solo stub vuoti, controlla se il modello aveva aperto un secondo blocco rimasto troncato
        class_occurrences = len(re.findall(r'class\s+Solution\b', llm_output))
        if class_occurrences > len(all_candidates) or llm_output.count("```") % 2 != 0:
            return (all_candidates[0] if all_candidates else None), True

        if all_candidates:
            return all_candidates[-1], True

        return None, True

    @staticmethod
    def extract_method_signature(java_code: str) -> Optional[Tuple[str, str, List[Tuple[str, str]]]]:
        """
        Estrae (return_type, method_name, [(param_type, param_name), ...])
        dal primo metodo pubblico della classe Solution[cite: 2].
        """
        clean_code = re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)
        clean_code = re.sub(r'//.*', '', clean_code)

        pattern = r'public\s+(?:static\s+)?(?!class\b)([\w<>\[\],\s]+?)\s+(\w+)\s*\(([^)]*)\)'
        match = re.search(pattern, clean_code)
        if not match:
            return None

        return_type = match.group(1).strip()
        method_name = match.group(2).strip()
        params_raw = match.group(3).strip()

        params = []
        if params_raw:
            param_list = re.split(r',(?![^<]*>)', params_raw)
            for p in param_list:
                p = p.strip()
                parts = p.rsplit(' ', 1)
                if len(parts) == 2:
                    params.append((parts[0].strip(), parts[1].strip()))

        return return_type, method_name, params

    @staticmethod
    def parse_examples_from_description(description: str) -> List[Dict[str, str]]:
        """
        Estrae le coppie Input/Output ripulendo i backslash di escape Markdown[cite: 2].
        """
        clean_desc = (
            description.replace(r'\[', '[')
                       .replace(r'\]', ']')
                       .replace(r'\{', '{')
                       .replace(r'\}', '}')
                       .replace('\\', '')
        )

        pattern = r'\*\*Input:\*\*\s*(.*?)\s*\*\*Output:\*\*\s*([^\n\r*]+)'
        matches = re.findall(pattern, clean_desc, flags=re.DOTALL)

        test_cases = []
        for inp_raw, out_raw in matches:
            inp_clean = inp_raw.strip()
            out_clean = out_raw.strip()
            test_cases.append({"input": inp_clean, "output": out_clean})
        return test_cases

    @classmethod
    def format_value_to_java_expr(cls, val_str: str, java_type: str) -> str:
        """
        Converte una stringa di input/output in un'espressione Java valida per il tipo specificato[cite: 1].
        """
        val_str = val_str.strip()

        if java_type == "String":
            if val_str.startswith('"') and val_str.endswith('"'):
                inner = val_str[1:-1].strip()
                return f'"{inner}"'
            return f'"{val_str}"'

        elif java_type in ["int", "long", "double", "float", "boolean"]:
            if java_type == "long" and not val_str.endswith(("L", "l")):
                return val_str + "L"
            if java_type == "float" and not val_str.endswith(("f", "F")):
                return val_str + "f"
            return val_str

        elif java_type == "char":
            inner = val_str.strip("'\" ")
            return f"'{inner}'"

        elif java_type == "TreeNode":
            inner = val_str.replace('"', '').strip()
            return f'TreeHelper.buildTree("{inner}")'

        elif java_type == "ListNode":
            inner = val_str.replace('"', '').strip()
            return f'ListHelper.buildList("{inner}")'

        elif java_type.endswith("[][]"):
            inner = val_str.replace('[', '{').replace(']', '}')
            if "String" in java_type:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', inner)
            elif "char" in java_type:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r"'\1'", inner)
            return f"new {java_type}{inner}"

        elif java_type.endswith("[]"):
            inner = val_str[1:-1].strip() if (val_str.startswith('[') and val_str.endswith(']')) else val_str
            if "String" in java_type and inner:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', inner)
            elif "char" in java_type and inner:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r"'\1'", inner)
            return f"new {java_type}{{{inner}}}"

        elif java_type.startswith("List<List<"): # - Converte [[a, [c, b], d]]> Arrays.asList(Arrays.asList(a, b), Arrays.asList(c, d))
            inner = val_str[1:-1].strip() if (val_str.startswith('[') and val_str.endswith(']')) else val_str
            if "String" in java_type:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', inner)
            sub_lists = re.findall(r'\[([^\[\]]*)\]', inner)
            if not sub_lists:
                return "new java.util.ArrayList<>()"
            sub_exprs = [f"java.util.Arrays.asList({s.strip()})" for s in sub_lists]
            return f"java.util.Arrays.asList({', '.join(sub_exprs)})"

        elif java_type.startswith("List<"):
            inner = val_str[1:-1].strip() if (val_str.startswith('[') and val_str.endswith(']')) else val_str
            if not inner:
                return "new java.util.ArrayList<>()"
            if "String" in java_type:
                inner = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', inner)
            return f"java.util.Arrays.asList({inner})"

        return val_str

    @classmethod
    def split_input_assignments(cls, input_str: str, params: List[Tuple[str, str]]) -> List[str]:
        # Splitta tutte le assegnazioni "nome = valore" di primo livello presenti nell'input_str
        raw_vals = [m.group(1).strip() for m in re.finditer(r'\b\w+\s*=\s*(.*?)(?=\s*,\s*\w+\s*=|$)', input_str, flags=re.DOTALL)]
        if not raw_vals:
            raw_vals = [input_str.strip()]
        # Associa in ordine ai parametri richiesti dal metodo Java
        values = []
        for i, (p_type, p_name) in enumerate(params):
            val_str = raw_vals[i] if i < len(raw_vals) else raw_vals[-1]
            values.append(cls.format_value_to_java_expr(val_str, p_type))
        return values

        values = []
        for i, (p_type, p_name) in enumerate(params):
            if i < len(params) - 1:
                next_p_name = params[i + 1][1]
                pattern = rf'{re.escape(p_name)}\s*=\s*(.*?)\s*,\s*{re.escape(next_p_name)}\s*='
                m = re.search(pattern, input_str, flags=re.DOTALL)
                if m:
                    val_str = m.group(1)
                else:
                    raise ValueError(f"Impossibile effettuare il parsing del parametro {p_name}")
            else:
                pattern = rf'{re.escape(p_name)}\s*=\s*(.*)$'
                m = re.search(pattern, input_str, flags=re.DOTALL)
                if m:
                    val_str = m.group(1)
                else:
                    raise ValueError(f"Impossibile effettuare il parsing dell'ultimo parametro {p_name}")

            values.append(cls.format_value_to_java_expr(val_str, p_type))
        return values

    @classmethod
    def build_java_test_harness(cls, solution_code: str, signature: Tuple, test_cases: List[Dict]) -> str:
        """
        Genera il sorgente completo Main.java con helper per TreeNode/ListNode e test isolati[cite: 1, 2].
        """
        ret_type, method_name, params = signature

        test_blocks = []
        for idx, tc in enumerate(test_cases):
            args_expr = cls.split_input_assignments(tc["input"], params)

            # Caso 1: Metodo void (modifica in-place del primo parametro)
            if ret_type == "void":
                target_type = params[0][0]
                target_var = f"target_{idx}"
                other_args = ", ".join(args_expr[1:])
                call_args = f"{target_var}, {other_args}" if other_args else target_var

                expected_expr = cls.format_value_to_java_expr(tc["output"], target_type)

                if target_type.endswith("[][]"):
                    cmp_expr = f"Arrays.deepEquals({target_var}, {expected_expr})"
                elif target_type.endswith("[]"):
                    cmp_expr = f"Arrays.equals({target_var}, {expected_expr})"
                elif target_type == "ListNode":
                    cmp_expr = f"ListHelper.isEqual({target_var}, {expected_expr})"
                elif target_type == "TreeNode":
                    cmp_expr = f"TreeHelper.isEqual({target_var}, {expected_expr})"
                else:
                    cmp_expr = f"Objects.equals({target_var}, {expected_expr})"

                block = f"""
            try {{
                Solution sol = new Solution();
                {target_type} {target_var} = {args_expr[0]};
                sol.{method_name}({call_args});
                if ({cmp_expr}) {{
                    System.out.println("TEST_{idx}:PASS");
                }} else {{
                    System.out.println("TEST_{idx}:FAIL");
                }}
            }} catch (Exception e) {{
                System.out.println("TEST_{idx}:ERROR:" + e.toString());
            }}
                """
            # Caso 2: Metodo con valore di ritorno
            else:
                expected_expr = cls.format_value_to_java_expr(tc["output"], ret_type)
                args_joined = ", ".join(args_expr)

                if ret_type in ["int", "long", "boolean", "char"]:
                    cmp_code = f"actual_{idx} == expected_{idx}"
                elif ret_type in ["double", "float"]:
                    cmp_code = f"Math.abs(actual_{idx} - expected_{idx}) < 1e-5"
                elif ret_type.endswith("[][]"):
                    cmp_code = f"Arrays.deepEquals(actual_{idx}, expected_{idx})"
                elif ret_type.endswith("[]"):
                    cmp_code = f"Arrays.equals(actual_{idx}, expected_{idx})"
                elif ret_type == "ListNode":
                    cmp_code = f"ListHelper.isEqual(actual_{idx}, expected_{idx})"
                elif ret_type == "TreeNode":
                    cmp_code = f"TreeHelper.isEqual(actual_{idx}, expected_{idx})"
                else:
                    cmp_code = f"Objects.equals(actual_{idx}, expected_{idx})"

                block = f"""
            try {{
                Solution sol = new Solution();
                {ret_type} actual_{idx} = sol.{method_name}({args_joined});
                {ret_type} expected_{idx} = {expected_expr};
                if ({cmp_code}) {{
                    System.out.println("TEST_{idx}:PASS");
                }} else {{
                    System.out.println("TEST_{idx}:FAIL:Expected " + expected_{idx} + " got " + actual_{idx});
                }}
            }} catch (Exception e) {{
                System.out.println("TEST_{idx}:ERROR:" + e.toString());
            }}
                """
            test_blocks.append(block)

        tests_joined = "\n".join(test_blocks)

        return f"""
import java.util.*;
import java.util.stream.*;
import java.math.*;

class ListNode {{
    int val;
    ListNode next;
    ListNode() {{}}
    ListNode(int val) {{ this.val = val; }}
    ListNode(int val, ListNode next) {{ this.val = val; this.next = next; }}
}}

class ListHelper {{
    public static ListNode buildList(String str) {{
        str = str.replace("[", "").replace("]", "").trim();
        if (str.isEmpty() || str.equals("null")) return null;
        String[] parts = str.split(",");
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        for (String p : parts) {{
            curr.next = new ListNode(Integer.parseInt(p.trim()));
            curr = curr.next;
        }}
        return dummy.next;
    }}

    public static boolean isEqual(ListNode l1, ListNode l2) {{
        while (l1 != null && l2 != null) {{
            if (l1.val != l2.val) return false;
            l1 = l1.next;
            l2 = l2.next;
        }}
        return l1 == null && l2 == null;
    }}
}}

class TreeNode {{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {{}}
    TreeNode(int val) {{ this.val = val; }}
    TreeNode(int val, TreeNode left, TreeNode right) {{
        this.val = val;
        this.left = left;
        this.right = right;
    }}
}}

class TreeHelper {{
    public static TreeNode buildTree(String str) {{
        str = str.replace("[", "").replace("]", "").trim();
        if (str.isEmpty() || str.equals("null")) return null;
        String[] parts = str.split(",");
        if (parts[0].trim().equals("null") || parts[0].trim().isEmpty()) return null;

        TreeNode root = new TreeNode(Integer.parseInt(parts[0].trim()));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        int i = 1;
        while (!queue.isEmpty() && i < parts.length) {{
            TreeNode curr = queue.poll();
            if (i < parts.length && !parts[i].trim().equals("null") && !parts[i].trim().isEmpty()) {{
                curr.left = new TreeNode(Integer.parseInt(parts[i].trim()));
                queue.offer(curr.left);
            }}
            i++;
            if (i < parts.length && !parts[i].trim().equals("null") && !parts[i].trim().isEmpty()) {{
                curr.right = new TreeNode(Integer.parseInt(parts[i].trim()));
                queue.offer(curr.right);
            }}
            i++;
        }}
        return root;
    }}

    public static boolean isEqual(TreeNode t1, TreeNode t2) {{
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return t1.val == t2.val && isEqual(t1.left, t2.left) && isEqual(t1.right, t2.right);
    }}
}}

{solution_code}

public class Main {{
    public static void main(String[] args) {{
        {tests_joined}
    }}
}}
"""

    @classmethod
    def evaluate(cls, llm_output: str, problem_description: str, timeout_sec: int = 5) -> Dict:
        sol_code, is_truncated = cls.extract_solution_code(llm_output)

        if not sol_code:
            return {
                "status": "EXTRACTION_ERROR",
                "passed": False,
                "details": "Nessuna classe Solution bilanciata trovata.",
                "extracted_code": ""
            }

        if is_truncated or cls._is_empty_or_stub(sol_code):
            return {
                "status": "INCOMPLETE_GENERATION",
                "passed": False,
                "details": "Generazione troncata da max_tokens o contenente solo lo scheletro vuoto del prompt.",
                "extracted_code": sol_code
            }

        sig = cls.extract_method_signature(sol_code)
        if not sig:
            return {
                "status": "SIGNATURE_ERROR",
                "passed": False,
                "details": "Firma del metodo non trovata.",
                "extracted_code": sol_code
            }

        test_cases = cls.parse_examples_from_description(problem_description)
        if not test_cases:
            return {
                "status": "NO_TESTS_FOUND",
                "passed": False,
                "details": "Nessun test case trovato nella descrizione.",
                "extracted_code": sol_code
            }

        try:
            full_java_code = cls.build_java_test_harness(sol_code, sig, test_cases)
        except Exception as e:
            return {
                "status": "HARNESS_BUILD_ERROR",
                "passed": False,
                "details": str(e),
                "extracted_code": sol_code
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            main_file = os.path.join(tmpdir, "Main.java")
            with open(main_file, "w", encoding="utf-8") as f:
                f.write(full_java_code)

            comp = subprocess.run(["javac", "Main.java"], cwd=tmpdir, capture_output=True, text=True)
            if comp.returncode != 0:
                return {
                    "status": "COMPILATION_ERROR",
                    "passed": False,
                    "details": comp.stderr.strip(),
                    "extracted_code": sol_code
                }

            try:
                exec_res = subprocess.run(
                    ["java", "Main"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "TIMEOUT",
                    "passed": False,
                    "details": f"Timeout {timeout_sec}s",
                    "extracted_code": sol_code
                }

            if exec_res.returncode != 0:
                return {
                    "status": "RUNTIME_ERROR",
                    "passed": False,
                    "details": exec_res.stderr.strip(),
                    "extracted_code": sol_code
                }

            stdout_lines = exec_res.stdout.strip().splitlines()
            results = [line for line in stdout_lines if line.startswith("TEST_")]

            passed_count = sum(1 for r in results if ":PASS" in r)
            all_passed = (passed_count == len(test_cases)) and (len(test_cases) > 0)

            return {
                "status": "PASSED" if all_passed else "FAILED",
                "passed": all_passed,
                "passed_tests": passed_count,
                "total_tests": len(test_cases),
                "test_outputs": results,
                "extracted_code": sol_code
            }