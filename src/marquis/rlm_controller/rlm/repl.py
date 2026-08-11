from __future__ import annotations

import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from marquis.rlm_controller.rlm.rlm import RLM


class Sub_RLM(RLM):
    def __init__(self, model: str = "gpt-5"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.model = model
        from marquis.rlm_controller.rlm.utils.llm import OpenAIClient

        self.client = OpenAIClient(api_key=self.api_key, model=model)

    def completion(self, prompt) -> str:
        try:
            response = self.client.completion(messages=prompt, timeout=300)
            return response
        except Exception as e:
            return f"Error making LLM query: {str(e)}"

    def cost_summary(self) -> dict[str, float]:
        raise NotImplementedError("Cost tracking is not implemented for the Sub-RLM.")

    def reset(self):
        raise NotImplementedError("Reset is not implemented for the Sub-RLM.")


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __init__(self, stdout: str, stderr: str, locals: dict, execution_time: float = None):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time

    def __str__(self):
        return (
            f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, "
            f"locals={self.locals}, execution_time={self.execution_time})"
        )


class REPLEnv:
    def __init__(
        self,
        recursive_model: str = "gpt-5-mini",
        context_json: dict | list | None = None,
        context_str: str | None = None,
        setup_code: str = None,
        extra_tools: dict | None = None,
        execution_timeout_seconds: float | None = None,
    ):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")
        self.execution_timeout_seconds = self._resolve_timeout(execution_timeout_seconds)

        self.sub_rlm: RLM = Sub_RLM(model=recursive_model)

        self.globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "bin": bin,
                "oct": oct,
                "repr": repr,
                "ascii": ascii,
                "format": format,
                "__import__": __import__,
                "open": open,
                "any": any,
                "all": all,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                "dir": dir,
                "vars": vars,
                "range": range,
                "reversed": reversed,
                "slice": slice,
                "iter": iter,
                "next": next,
                "pow": pow,
                "divmod": divmod,
                "complex": complex,
                "bytes": bytes,
                "bytearray": bytearray,
                "memoryview": memoryview,
                "hash": hash,
                "id": id,
                "callable": callable,
                "issubclass": issubclass,
                "super": super,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "object": object,
                "BaseException": BaseException,
                "ArithmeticError": ArithmeticError,
                "LookupError": LookupError,
                "EnvironmentError": EnvironmentError,
                "AssertionError": AssertionError,
                "NotImplementedError": NotImplementedError,
                "UnicodeError": UnicodeError,
                "Warning": Warning,
                "UserWarning": UserWarning,
                "DeprecationWarning": DeprecationWarning,
                "PendingDeprecationWarning": PendingDeprecationWarning,
                "SyntaxWarning": SyntaxWarning,
                "RuntimeWarning": RuntimeWarning,
                "FutureWarning": FutureWarning,
                "ImportWarning": ImportWarning,
                "UnicodeWarning": UnicodeWarning,
                "BytesWarning": BytesWarning,
                "ResourceWarning": ResourceWarning,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "FileNotFoundError": FileNotFoundError,
                "OSError": OSError,
                "IOError": IOError,
                "RuntimeError": RuntimeError,
                "NameError": NameError,
                "ImportError": ImportError,
                "StopIteration": StopIteration,
                "GeneratorExit": GeneratorExit,
                "SystemExit": SystemExit,
                "KeyboardInterrupt": KeyboardInterrupt,
                "input": None,
                "eval": None,
                "exec": None,
                "compile": None,
                "globals": None,
                "locals": None,
            }
        }
        self.locals = {}
        self._lock = threading.Lock()
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()

        self.load_context(context_json, context_str)

        def llm_query(prompt: str) -> str:
            return self.sub_rlm.completion(prompt)

        self.globals["llm_query"] = llm_query

        def final_var(variable_name: str) -> str:
            variable_name = variable_name.strip().strip('"').strip("'").strip("\n").strip("\r")
            try:
                if variable_name in self.locals:
                    value = self.locals[variable_name]
                    return str(value)
                else:
                    return f"Error: Variable '{variable_name}' not found in REPL environment"
            except Exception as e:
                return f"Error retrieving variable '{variable_name}': {str(e)}"

        self.globals["FINAL_VAR"] = final_var

        if extra_tools:
            for tool_name, tool_fn in extra_tools.items():
                self.globals[tool_name] = tool_fn

        if setup_code:
            self.code_execution(setup_code)

    @staticmethod
    def _resolve_timeout(value: float | None) -> float | None:
        if value is not None:
            return value
        raw = os.getenv("MARQUIS_REPL_TIMEOUT_SECONDS", "300")
        if not raw:
            return None
        try:
            timeout = float(raw)
        except ValueError:
            return 300.0
        return timeout if timeout > 0 else None

    def load_context(self, context_json: dict | list | None = None, context_str: str | None = None):
        if context_json is not None:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_json, f, indent=2)
            context_code = (
                f"import json\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = json.load(f)\n"
            )
            self.code_execution(context_code)

        if context_str is not None:
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_str)
            context_code = (
                f"import os\nwith open(r'{context_path}', 'r') as f:\n    context = f.read()\n"
            )
            self.code_execution(context_code)

    def __del__(self):
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    @contextmanager
    def _capture_output(self):
        with self._lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    @contextmanager
    def _execution_timeout(self):
        timeout = self.execution_timeout_seconds
        can_signal = (
            timeout is not None
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        if not can_signal:
            yield
            return

        def _raise_timeout(_signum, _frame):
            raise TimeoutError(f"REPL execution exceeded {timeout:g}s")

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        old_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

    def code_execution(self, code) -> REPLResult:
        start_time = time.time()
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    with self._execution_timeout():
                        lines = code.split("\n")
                        import_lines = []
                        other_lines = []

                        for line in lines:
                            if line.startswith(("import ", "from ")) and not line.startswith("#"):
                                import_lines.append(line)
                            else:
                                other_lines.append(line)

                        if import_lines:
                            import_code = "\n".join(import_lines)
                            exec(import_code, self.globals, self.globals)

                        if other_lines:
                            other_code = "\n".join(other_lines)
                            combined_namespace = {**self.globals, **self.locals}
                            non_comment_lines = [
                                line for line in other_lines if line and not line.startswith("#")
                            ]

                            if non_comment_lines:
                                last_line = non_comment_lines[-1]
                                is_expression = (
                                    not last_line.startswith(
                                        (
                                            "import ",
                                            "from ",
                                            "def ",
                                            "class ",
                                            "if ",
                                            "for ",
                                            "while ",
                                            "try:",
                                            "with ",
                                            "return ",
                                            "yield ",
                                            "break",
                                            "continue",
                                            "pass",
                                        )
                                    )
                                    and "=" not in last_line.split("#")[0]
                                    and not last_line.endswith(":")
                                    and not last_line.startswith("print(")
                                )

                                if is_expression:
                                    if len(non_comment_lines) > 1:
                                        last_line_start = -1
                                        for i, line in enumerate(other_lines):
                                            if line == last_line:
                                                last_line_start = i
                                                break
                                        if last_line_start > 0:
                                            statements_code = "\n".join(
                                                other_lines[:last_line_start]
                                            )
                                            exec(
                                                statements_code,
                                                combined_namespace,
                                                combined_namespace,
                                            )
                                    result = eval(last_line, combined_namespace, combined_namespace)
                                    if result is not None:
                                        print(repr(result))
                                else:
                                    exec(other_code, combined_namespace, combined_namespace)
                            else:
                                exec(other_code, combined_namespace, combined_namespace)

                            for key, value in combined_namespace.items():
                                if key not in self.globals:
                                    self.locals[key] = value

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()
                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()

        end_time = time.time()
        execution_time = end_time - start_time

        self.locals["_stdout"] = stdout_content
        self.locals["_stderr"] = stderr_content

        return REPLResult(stdout_content, stderr_content, self.locals.copy(), execution_time)

    def get_cost_summary(self):
        raise NotImplementedError("Cost tracking is not implemented for the REPL Environment.")
