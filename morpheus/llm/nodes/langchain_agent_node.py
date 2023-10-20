# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging

from langchain.agents import AgentExecutor

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase

logger = logging.getLogger(__name__)


class LangChainAgentNode(LLMNodeBase):

    def __init__(self, agent_executor: AgentExecutor):
        super().__init__()

        self._agent_executor = agent_executor

        self._input_names = self._agent_executor.input_keys

    def get_input_names(self):
        return self._input_names

    async def _run_single(self, **kwargs):

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

        results_async = [self._agent_executor.arun(**x) for x in input_list]

        results = await asyncio.gather(*results_async)

        return results

    async def execute(self, context: LLMContext):

        input_dict = context.get_inputs()

        # Transform from dict[str, list[Any]] to list[dict[str, Any]]
        input_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

        results_async = [self._run_single(**x) for x in input_list]

        results = await asyncio.gather(*results_async)

        # Transform from list[dict[str, Any]] to dict[str, list[Any]]
        # results = {k: [x[k] for x in results] for k in results[0]}

        context.set_output(results)