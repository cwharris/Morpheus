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
import typing

from morpheus.llm import InputMap
from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.llm import LLMTask
from morpheus.llm import LLMTaskHandler
from morpheus.messages import ControlMessage


def execute_node(node: LLMNodeBase, **input_values: dict) -> typing.Any:
    """
    Executes an LLM Node with the necessary LLM context, and extracts the output values.
    """
    inputs: list[InputMap] = []
    parent_context = LLMContext()

    for input_name, input_value in input_values.items():
        inputs.append(InputMap(f"/{input_name}", input_name))
        parent_context.set_output(input_name, input_value)

    context = parent_context.push("test", inputs)

    context = asyncio.run(node.execute(context))

    return context.view_outputs


def execute_task_handler(task_handler: LLMTaskHandler, task_dict: dict, input_message,
                         **input_values: dict) -> ControlMessage:
    """
    Executes an LLM task handler with the necessary LLM context.
    """
    task = LLMTask("unittests", task_dict)
    inputs: list[InputMap] = []
    parent_context = LLMContext(task, input_message)

    for input_name, input_value in input_values.items():
        inputs.append(InputMap(f"/{input_name}", input_name))
        parent_context.set_output(input_name, input_value)

    context = parent_context.push("test", inputs)

    message = asyncio.run(task_handler.try_handle(context))

    return message
