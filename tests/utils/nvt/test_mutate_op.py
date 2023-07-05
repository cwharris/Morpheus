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

import numpy as np
import pandas as pd
from merlin.core.dispatch import DataFrameType
from merlin.schema import ColumnSchema
from merlin.schema import Schema
from nvtabular.ops.operator import ColumnSelector

from morpheus.utils.nvt import MutateOp


def setUp():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    def example_transform(col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selected_columns = col_selector.names
        for col in selected_columns:
            df[col + '_new'] = df[col] * 2
        return df

    return df, example_transform


def test_transform():
    df, example_transform = setUp()
    op = MutateOp(example_transform, output_columns=[('A_new', np.dtype('int64')), ('B_new', np.dtype('int64'))])
    col_selector = ColumnSelector(['A', 'B'])
    transformed_df = op.transform(col_selector, df)

    expected_df = df.copy()
    expected_df['A_new'] = df['A'] * 2
    expected_df['B_new'] = df['B'] * 2

    print("")
    print(expected_df)
    print(transformed_df)

    assert transformed_df.equals(expected_df), "Test transform failed"


# Test for lambda function transformation
def test_transform_lambda():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    op = MutateOp(lambda col_selector,
                  df: df.assign(**{f"{col}_new": df[col] * 2
                                   for col in col_selector.names}),
                  output_columns=[('A_new', np.dtype('int64')), ('B_new', np.dtype('int64'))])
    col_selector = ColumnSelector(['A', 'B'])
    transformed_df = op.transform(col_selector, df)

    expected_df = df.copy()
    expected_df['A_new'] = df['A'] * 2
    expected_df['B_new'] = df['B'] * 2

    assert transformed_df.equals(expected_df), "Test transform with lambda failed"


def test_transform_additional_columns():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    def additional_transform(col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selected_columns = col_selector.names
        for col in selected_columns:
            df[col + '_new'] = df[col] * 2
        df['D'] = df['A'] + df['B']
        return df

    op = MutateOp(additional_transform,
                  output_columns=[('A_new', np.dtype('int64')), ('B_new', np.dtype('int64')), ('D', np.dtype('int64'))])
    col_selector = ColumnSelector(['A', 'B'])
    transformed_df = op.transform(col_selector, df)

    expected_df = df.copy()
    expected_df['A_new'] = df['A'] * 2
    expected_df['B_new'] = df['B'] * 2
    expected_df['D'] = df['A'] + df['B']

    assert transformed_df.equals(expected_df), "Test transform with additional columns failed"


def test_column_mapping():
    _, example_transform = setUp()
    op = MutateOp(example_transform, output_columns=[('A_new', np.dtype('int64')), ('B_new', np.dtype('int64'))])
    col_selector = ColumnSelector(['A', 'B'])
    column_mapping = op.column_mapping(col_selector)

    expected_mapping = {'A_new': ['A', 'B'], 'B_new': ['A', 'B']}

    assert column_mapping == expected_mapping, "Test column mapping failed"


def test_compute_output_schema():
    _, example_transform = setUp()
    op = MutateOp(example_transform, output_columns=[('A_new', np.dtype('int64')), ('B_new', np.dtype('int64'))])
    col_selector = ColumnSelector(['A', 'B'])

    input_schema = Schema([
        ColumnSchema('A', dtype=np.dtype('int64')),
        ColumnSchema('B', dtype=np.dtype('int64')),
        ColumnSchema('C', dtype=np.dtype('int64'))
    ])

    output_schema = op.compute_output_schema(input_schema, col_selector)

    expected_schema = Schema(
        [ColumnSchema('A_new', dtype=np.dtype('int64')), ColumnSchema('B_new', dtype=np.dtype('int64'))])

    assert str(output_schema) == str(expected_schema), "Test compute output schema failed"