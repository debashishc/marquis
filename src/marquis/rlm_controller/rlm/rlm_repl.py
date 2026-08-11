from __future__ import annotations

import time
from typing import Any

from marquis.rlm_controller.rlm.logger.repl_logger import REPLEnvLogger
from marquis.rlm_controller.rlm.logger.root_logger import ColorfulLogger
from marquis.rlm_controller.rlm.repl import REPLEnv
from marquis.rlm_controller.rlm.rlm import RLM
from marquis.rlm_controller.rlm.utils import utils
from marquis.rlm_controller.rlm.utils.prompts import (
    DEFAULT_QUERY,
    build_system_prompt,
    next_action_prompt,
)


class RLM_REPL(RLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5",
        recursive_model: str = "gpt-5",
        max_iterations: int = 20,
        depth: int = 0,
        enable_logging: bool = False,
        extra_tools: dict[str, Any] | None = None,
        system_prompt_builder: Any | None = None,
    ):
        from marquis.rlm_controller.rlm.utils.llm import OpenAIClient

        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(api_key, model)
        self._extra_tools = extra_tools or {}
        self._system_prompt_builder = system_prompt_builder or build_system_prompt

        self.repl_env = None
        self.depth = depth
        self._max_iterations = max_iterations

        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)

        self.messages = []
        self.trajectory = []
        self.query = None

    def setup_context(
        self, context: list[str] | str | list[dict[str, str]], query: str | None = None
    ):
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.trajectory = []
        self.logger.log_query_start(query)

        self.messages = self._system_prompt_builder()
        self.logger.log_initial_messages(self.messages)

        context_data, context_str = utils.convert_context_for_repl(context)

        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            extra_tools=self._extra_tools if self._extra_tools else None,
        )

        return self.messages

    def completion(
        self, context: list[str] | str | list[dict[str, str]], query: str | None = None
    ) -> str:
        self.messages = self.setup_context(context, query)

        for iteration in range(self._max_iterations):
            iteration_record = {
                "iteration": iteration,
                "prompt_type": "next_action",
            }

            root_start = time.time()
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])
            root_elapsed = time.time() - root_start
            iteration_record["root_response"] = response
            iteration_record["root_response_time"] = root_elapsed

            code_blocks = utils.find_code_blocks(response)
            iteration_record["code_blocks"] = code_blocks or []
            self.logger.log_model_response(response, has_tool_calls=bool(code_blocks))

            assistant_message = {
                "role": "assistant",
                "content": response,
            }
            self.messages.append(assistant_message)

            if code_blocks:
                self.messages, execution_trace = utils.process_code_execution_with_trace(
                    response, self.messages, self.repl_env, self.repl_env_logger, self.logger
                )
                iteration_record["executions"] = execution_trace
            else:
                iteration_record["executions"] = []

            final_answer = utils.check_for_final_answer(
                response,
                self.repl_env,
                self.logger,
            )
            iteration_record["final_answer"] = final_answer
            self.trajectory.append(iteration_record)

            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        root_start = time.time()
        final_answer = self.llm.completion(self.messages)
        root_elapsed = time.time() - root_start
        self.trajectory.append(
            {
                "iteration": self._max_iterations,
                "prompt_type": "fallback_final_answer",
                "root_response": final_answer,
                "root_response_time": root_elapsed,
                "code_blocks": [],
                "executions": [],
                "final_answer": final_answer,
            }
        )
        self.logger.log_final_response(final_answer)

        return final_answer

    def cost_summary(self) -> dict[str, Any]:
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        self.repl_env = REPLEnv()
        self.messages = []
        self.trajectory = []
        self.query = None


if __name__ == "__main__":
    pass
