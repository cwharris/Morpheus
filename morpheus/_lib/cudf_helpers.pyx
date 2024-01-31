# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cudf

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.io.types cimport table_with_metadata
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport data_from_unique_ptr
from cudf._lib.utils cimport get_column_names
from cudf._lib.utils cimport table_view_from_table


cdef extern from "morpheus/objects/table_info.hpp" namespace "morpheus" nogil:


    cdef cppclass TableInfoData:
        TableInfoData()
        TableInfoData(table_view view,
                      vector[string] indices,
                      vector[string] columns)

        table_view table_view
        vector[string] index_names
        vector[string] column_names
        vector[size_type] column_indices


cdef public api:
    object make_table_from_table_with_metadata(table_with_metadata table, int index_col_count):

        schema_infos = [x.name.decode() for x in table.metadata.schema_info]
        index_names = schema_infos[0:index_col_count] if index_col_count > 0 else None
        column_names = schema_infos[index_col_count:]

        data, index = data_from_unique_ptr(move(table.tbl), column_names=column_names, index_names=index_names)

        return cudf.DataFrame._from_data(data, index)

    object make_table_from_table_info_data(TableInfoData table_info, object owner):

        index_names = None

        if (table_info.index_names.size() > 0):
            index_names = []

            for c_name in table_info.index_names:
                name = c_name.decode()
                index_names.append(name if name != "" else None)

        column_names = []

        for c_name in table_info.column_names:
                name = c_name.decode()
                column_names.append(name if name != "" else None)


        column_indicies = []

        for c_index in table_info.column_indices:
            print("c_index:", c_index)
            column_indicies.append(c_index)

        x = table_info.table_view

        try:
            data, index = data_from_table_view_indexed(x, owner=owner, column_names=column_names, column_indices=column_indicies, index_names=index_names)
        except Exception:
            import traceback
            print(traceback.format_exc())

        return cudf.DataFrame._from_data(data, index)


    TableInfoData make_table_info_data_from_table(object table):

        cdef vector[string] temp_col_names = get_column_names(table, True)

        cdef table_view input_table_view = table_view_from_table(table, ignore_index=False)
        cdef vector[string] index_names
        cdef vector[string] column_names

        # cuDF does a weird check where if there is only one name in both index and columns, and that column is empty or
        # None, then change it to '""'. Not sure what this is used for
        check_empty_name = get_column_names(table, True).size() == 1

        for name in table._index.names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            index_names.push_back(str.encode(name))

        for name in table._column_names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            column_names.push_back(str.encode(name))

        return TableInfoData(input_table_view, index_names, column_names)

    cdef data_from_table_view_indexed(
        table_view tv,
        object owner,
        object column_names,
        object column_indices,
        object index_names
    ):
        """
        Given a ``cudf::table_view``, constructs a Frame from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a Frame we reach inside of it and
        reach inside of each ``cudf.Column`` to make the owner of each newly
        created ``Buffer`` underneath the ``cudf.Column`` objects of the
        created Frame the respective ``Buffer`` from the relevant
        ``cudf.Column`` of the ``owner`` Frame
        """
        cdef size_type column_idx = 0
        table_owner = isinstance(owner, cudf.core.frame.Frame)

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                column_owner = owner
                if table_owner:
                    column_owner = owner._index._columns[column_idx]
                index_columns.append(
                    Column.from_column_view(
                        tv.column(column_idx),
                        column_owner
                    )
                )
                column_idx += 1
            index = cudf.core.index._index_from_data(
                dict(zip(index_names, index_columns)))

        # Construct the data dict
        cdef size_type source_column_idx = 0
        data_columns = []
        for _ in column_names:
            column_owner = owner
            print("source_column_idx: ", source_column_idx)
            print("column_indices[source_column_idx]:", column_indices[source_column_idx])
            if table_owner:
                column_owner = owner._columns[column_indices[source_column_idx]]
            data_columns.append(
                Column.from_column_view(tv.column(column_idx), column_owner)
            )
            column_idx += 1
            source_column_idx += 1

        return dict(zip(column_names, data_columns)), index
