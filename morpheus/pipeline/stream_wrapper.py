# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import inspect
import logging
import typing
from abc import ABC
from abc import abstractmethod

import neo

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType

logger = logging.getLogger(__name__)


def _save_init_vals(func: _DecoratorType) -> _DecoratorType:

    # Save the signature only once
    sig = inspect.signature(func, follow_wrapped=True)

    def inner(self: "StreamWrapper", c: Config, *args, **kwargs):

        # Actually call init first. This way any super classes strings will be overridden
        func(self, c, *args, **kwargs)

        # Determine all set values
        bound = sig.bind(self, c, *args, **kwargs)
        bound.apply_defaults()

        init_pairs = []

        for key, val in bound.arguments.items():

            # We really dont care about these
            if (key == "self" or key == "c"):
                continue

            init_pairs.append(f"{key}={val}")

        # Save values on self
        self._init_str = ", ".join(init_pairs)

        return

    return typing.cast(_DecoratorType, inner)


class StreamWrapper(ABC, collections.abc.Hashable):
    """
    This abstract class serves as the morpheus.pipeline's base class. This class wraps a `neo.Node`
    object and aids in hooking stages up together.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    __ID_COUNTER = AtomicInteger(0)

    def __init__(self, c: Config):
        # Save the config
        self._config = c

        self._id = StreamWrapper.__ID_COUNTER.get_and_inc()
        self._pipeline: _pipeline.Pipeline = None
        self._init_str: str = ""  # Stores the initialization parameters used for creation. Needed for __repr__

        # Indicates whether or not this wrapper has been built. Can only be built once
        self._is_built = False

        # Input/Output ports used for connecting stages
        self._input_ports: typing.List[_pipeline.Receiver] = []
        self._output_ports: typing.List[_pipeline.Sender] = []

    def __init_subclass__(cls) -> None:

        # Wrap __init__ to save the arg values
        cls.__init__ = _save_init_vals(cls.__init__)

        return super().__init_subclass__()

    def __hash__(self) -> int:
        return self._id

    def __str__(self):
        text = f"<{self.unique_name}; {self.__class__.__name__}({self._init_str})>"

        return text

    __repr__ = __str__

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the stage. Used in logging. Each derived class should override this property with a unique
        name.

        Returns
        -------
        str
            Name of a stage.

        """
        pass

    @property
    def unique_name(self) -> str:
        """
        Unique name of stag. Generated by appending stage id to stage name.

        Returns
        -------
        str
            Unique name of stage.
        """
        return f"{self.name}-{self._id}"

    @property
    def is_built(self) -> bool:
        """
        Indicates if this stage has been built.

        Returns
        -------
        bool
            True if stage is built, False otherwise.
        """
        return self._is_built

    @property
    def input_ports(self) -> typing.List[_pipeline.Receiver]:
        """Input ports to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Receiver`]
            Input ports to this stage.
        """
        return self._input_ports

    @property
    def output_ports(self) -> typing.List[_pipeline.Sender]:
        """
        Output ports from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Sender`]
            Output ports from this stage.
        """
        return self._output_ports

    @property
    def has_multi_input_ports(self) -> bool:
        """
        Indicates if this stage has multiple input ports.

        Returns
        -------
        bool
            True if stage has multiple input ports, False otherwise.
        """
        return len(self._input_ports) > 1

    @property
    def has_multi_output_ports(self) -> bool:
        """
        Indicates if this stage has multiple output ports.

        Returns
        -------
        bool
            True if stage has multiple output ports, False otherwise.
        """
        return len(self._output_ports) > 1

    def get_all_inputs(self) -> typing.List[_pipeline.Sender]:
        """
        Get all input senders to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Sender`]
            All input senders.
        """

        senders = []

        for in_port in self._input_ports:
            senders.extend(in_port._input_senders)

        return senders

    def get_all_input_stages(self) -> typing.List["StreamWrapper"]:
        """
        Get all input stages to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.StreamWrapper`]
            All input stages.
        """
        return [x.parent for x in self.get_all_inputs()]

    def get_all_outputs(self) -> typing.List[_pipeline.Receiver]:
        """
        Get all output receivers from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Receiver`]
            All output receivers.
        """
        receivers = []

        for out_port in self._output_ports:
            receivers.extend(out_port._output_receivers)

        return receivers

    def get_all_output_stages(self) -> typing.List["StreamWrapper"]:
        """
        Get all output stages from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.StreamWrapper`]
            All output stages.
        """
        return [x.parent for x in self.get_all_outputs()]

    def supports_cpp_node(self):
        """
        Specifies whether this Stage is even capable of creating C++ nodes. During the build phase, this value will be
        combined with Config.get().use_cpp to determine whether or not a C++ node is created. This is an instance method
        to allow runtime decisions and derived classes to override base implementations.
        """
        # By default, return False unless otherwise specified
        return False

    def _build_cpp_node(self):
        """
        Specifies whether or not to build a C++ node. Only should be called during the build phase.
        """
        return CppConfig.get_should_use_cpp() and self.supports_cpp_node()

    def can_build(self, check_ports=False) -> bool:
        """
        Determines if all inputs have been built allowing this node to be built.

        Parameters
        ----------
        check_ports : bool, optional
            Check if we can build based on the input ports, by default False.

        Returns
        -------
        bool
            True if we can build, False otherwise.
        """

        # Can only build once
        if (self.is_built):
            return False

        if (not check_ports):
            # We can build if all input stages have been built. Easy and quick check. Works for non-circular pipelines
            for in_stage in self.get_all_input_stages():
                if (not in_stage.is_built):
                    return False

            return True
        else:
            # Check if we can build based on the input ports. We can build
            for r in self.input_ports:
                if (not r.is_partial):
                    return False

            return True

    def build(self, seg: neo.Segment, do_propagate=True):
        """Build this stage.

        Parameters
        ----------
        seg : `neo.Segment`
            Neo segment for this stage.
        do_propagate : bool, optional
            Whether to propagate to build output stages, by default True.

        """
        assert not self.is_built, "Can only build stages once!"
        assert self._pipeline is not None, "Must be attached to a pipeline before building!"

        # Pre-Build returns the input pairs for each port
        in_ports_pairs = self._pre_build()

        out_ports_pair = self._build(seg, in_ports_pairs)

        # Allow stages to do any post build steps (i.e., for sinks, or timing functions)
        out_ports_pair = self._post_build(seg, out_ports_pair)

        assert len(out_ports_pair) == len(self.output_ports), \
            "Build must return same number of output pairs as output ports"

        # Assign the output ports
        for port_idx, out_pair in enumerate(out_ports_pair):
            self.output_ports[port_idx]._out_stream_pair = out_pair

        self._is_built = True

        if (not do_propagate):
            return

        # Now build for any dependents
        for dep in self.get_all_output_stages():
            if (not dep.can_build()):
                continue

            dep.build(seg, do_propagate=do_propagate)

    def _pre_build(self) -> typing.List[StreamPair]:
        in_pairs: typing.List[StreamPair] = [x.get_input_pair() for x in self.input_ports]

        return in_pairs

    @abstractmethod
    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        """
        This function is responsible for constructing this stage's internal `neo.Node` object. The input
        of this function contains the returned value from the upstream stage.

        The input values are the `neo.Segment` for this stage and a `StreamPair` tuple which contain the input
        `neo.Node` object and the message data type.

        :meta public:

        Parameters
        ----------
        seg : `neo.Segment`
            `neo.Segment` object for the pipeline. This should be used to construct/attach the internal `neo.Node`.
        in_ports_streams : `morpheus.pipeline.pipeline.StreamPair`
            List of tuples containing the input `neo.Node` object and the message data type.

        Returns
        -------
        `typing.List[morpheus.pipeline.pipeline.StreamPair]`
            List of tuples containing the output `neo.Node` object from this stage and the message data type.

        """
        pass

    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:
        return out_ports_pair

    def start(self):

        assert self.is_built, "Must build before starting!"

        self._start()

    def _start(self):
        pass

    def stop(self):
        """
        Stages can implement this to perform cleanup steps when pipeline is stopped.
        """
        pass

    async def join(self):
        pass

    def _create_ports(self, input_count: int, output_count: int):
        assert len(self._input_ports) == 0 and len(self._output_ports) == 0, "Can only create ports once!"

        self._input_ports = [_pipeline.Receiver(parent=self, port_number=i) for i in range(input_count)]
        self._output_ports = [_pipeline.Sender(parent=self, port_number=i) for i in range(output_count)]
