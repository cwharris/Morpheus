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
import threading
import typing

import mrc

import morpheus._lib.llm as _llm
from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(f"morpheus.{__name__}")


class LLMEngineStage(SinglePortStage):

    def __init__(self, c: Config, *, engine: LLMEngine):
        super().__init__(c)

        self._engine = engine

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "llm-engine"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (ControlMessage, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return True

    def _start(self):
        # Create a thread with its own asyncio loop

        self._thread = threading.Thread(target=self._engine.run)

    def _async_main(self):

        self._loop = asyncio.new_event_loop()

        self._loop.run_forever()

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        node = _llm.LLMEngineStage(builder, self.unique_name, self._engine)
        node.launch_options.pe_count = 2

        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
