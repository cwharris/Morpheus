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

import logging
import typing

import nvtabular as nvt
import pandas as pd

import cudf

from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.nvt import dataframe_input_schema_to_nvt_workflow
# Apply patches to NVT
# TODO(Devin): Can be removed, once numpy mappings are updated in Merlin
# ========================================================================
from morpheus.utils.nvt.patches import patch_numpy_dtype_registry

patch_numpy_dtype_registry()
# ========================================================================

logger = logging.getLogger(__name__)

def process_dataframe(
    df_in: typing.Union[pd.DataFrame, cudf.DataFrame],
    input_schema: typing.Union[nvt.Workflow, DataFrameInputSchema],
) -> pd.DataFrame:
    """
    Applies column transformations as defined by `input_schema`
    """

    workflow = input_schema
    if (isinstance(input_schema, DataFrameInputSchema)):
        workflow = dataframe_input_schema_to_nvt_workflow(input_schema)

    convert_to_pd = False
    if (isinstance(df_in, pd.DataFrame)):
        convert_to_pd = True
        df_in = cudf.DataFrame(df_in)

    dataset = nvt.Dataset(df_in)

    result = workflow.fit_transform(dataset).to_ddf().compute()

    if (convert_to_pd):
        return result.to_pandas()
    else:
        return result
