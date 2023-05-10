# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from .mutate import MutateOp
from .schema_converters import dataframe_input_schema_to_nvt_workflow


def register_morpheus_extensions():
    from datetime import datetime

    import merlin.dtypes.aliases as mn
    from merlin.dtypes import register
    from merlin.dtypes.mapping import DTypeMapping

    morpheus_extension = DTypeMapping(mapping={
        mn.datetime64: [datetime],
    }, )

    register("morpheus_ext", morpheus_extension)


register_morpheus_extensions()
