# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import pickle
import time
import typing
from dataclasses import dataclass

import mrc
from mrc.core import operators as ops
from pydantic import ValidationError

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.modules.schemas.write_to_vector_db_schema import WriteToVDBSchema
from morpheus.service.vdb.milvus_client import DATA_TYPE_MAP
from morpheus.service.vdb.vector_db_service import VectorDbServiceProvider
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

WriteToVectorDBLoaderFactory = ModuleLoaderFactory(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)


def preprocess_vdb_resources(service, recreate: bool, resource_schemas: dict):
    for resource_name, resource_schema_config in resource_schemas.items():
        has_object = service.has_store_object(name=resource_name)

        if (recreate and has_object):
            # Delete the existing resource
            service.drop(name=resource_name)
            has_object = False

        # Ensure that the resource exists
        if (not has_object):
            # TODO(Devin)
            import pymilvus
            schema_fields = []
            for field_data in resource_schema_config["schema_conf"]["schema_fields"]:
                if "dtype" in field_data:
                    field_data["dtype"] = DATA_TYPE_MAP.get(field_data["dtype"])
                    field_schema = pymilvus.FieldSchema(**field_data)
                    schema_fields.append(field_schema.to_dict())
                else:
                    schema_fields.append(field_data)

            resource_schema_config["schema_conf"]["schema_fields"] = schema_fields
            # function that we need to call first to turn resource_kwargs into a milvus config spec.

            service.create(name=resource_name, **resource_schema_config)


@dataclass
class AccumulationStats:
    msg_count: int
    last_insert_time: float
    data: list[cudf.DataFrame]


@register_module(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)
def _write_to_vector_db(builder: mrc.Builder):
    """
    Deserializes incoming messages into either MultiMessage or ControlMessage format.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'embedding_column_name': str, the name of the column containing embeddings (default is "embedding").
    - 'recreate': bool, whether to recreate the resource if it already exists (default is False).
    - 'service': instance of VectorDbServiceProvider.
    - 'default_resource_name': str, the name of the collection resource (must not be None or empty).
    - 'resource_kwargs': dict, additional keyword arguments for resource creation.
    - 'resource_schemas': dict, additional keyword arguments for resource creation.
    - 'batch_size': int, accumulates messages until reaching the specified batch size for writing to VDB.
    - 'write_time_interval': float, specifies the time interval (in seconds) for writing messages, or writing messages
    when the accumulated batch size is reached.

    Raises
    ------
    ValueError
        If 'resource_name' is None or empty.
        If 'service' is not provided or is not a valid service name or a serialized instance of VectorDBService.
    """

    module_config = builder.get_current_module_config()

    try:
        write_to_vdb_config = WriteToVDBSchema(**module_config)
    except ValidationError as e:
        # Format the error message for better readability
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration for write_to_vector_db: {error_messages}"
        logger.error(log_error_message)

        raise

    embedding_column_name = write_to_vdb_config.embedding_column_name
    recreate = write_to_vdb_config.recreate
    service_provider = write_to_vdb_config.service_provider
    default_resource_name = write_to_vdb_config.default_resource_name
    resource_kwargs = write_to_vdb_config.resource_kwargs
    resource_schemas = write_to_vdb_config.resource_schemas
    batch_size = write_to_vdb_config.batch_size
    write_time_interval = write_to_vdb_config.write_time_interval

    service_provider: VectorDbServiceProvider = pickle.loads(bytes(service_provider, "latin1"))
    
    service = service_provider.create()

    preprocess_vdb_resources(service, recreate, resource_schemas)

    accumulator_dict = {default_resource_name: AccumulationStats(msg_count=0, last_insert_time=-1, data=[])}

    def on_completed():
        final_df_references = []

        # Pushing remaining messages
        for key, accum_stats in accumulator_dict.items():
            try:
                if accum_stats.data:
                    merged_df = cudf.concat(accum_stats.data)
                    collection = service.load_resource(name=key)
                    collection.insert_dataframe(df=merged_df)
                    final_df_references.append(accum_stats.data)
            except Exception as e:
                logger.error("Unable to upload dataframe entries to vector database: %s", e)
        # Close vector database service connection
        service.close()

    def extract_df(msg: typing.Union[ControlMessage, MultiResponseMessage, MultiMessage]):
        df = None
        resource_name = None

        if isinstance(msg, ControlMessage):
            df = msg.payload().df
            if (msg.has_metadata("vdb_resource")):
                resource_name = msg.get_metadata("vdb_resource")
            else:
                resource_name = None
        elif isinstance(msg, MultiResponseMessage):
            df = msg.get_meta()
            if df is not None and not df.empty:
                embeddings = msg.get_probs_tensor()
                df[embedding_column_name] = embeddings.tolist()
        elif isinstance(msg, MultiMessage):
            df = msg.get_meta()
        else:
            raise RuntimeError(f"Unexpected message type '{type(msg)}' was encountered.")

        return df, resource_name

    def on_data(msg: typing.Union[ControlMessage, MultiResponseMessage, MultiMessage]):
        msg_resource_target = None
        try:
            df, msg_resource_target = extract_df(msg)

            if df is not None and not df.empty:
                if (not isinstance(df, cudf.DataFrame)):
                    df = cudf.DataFrame(df)

                df_size = len(df)
                current_time = time.time()

                # Use default resource name
                if not msg_resource_target:
                    msg_resource_target = default_resource_name
                    if not service.has_store_object(msg_resource_target):
                        logger.error("Resource not exists in the vector database: %s", msg_resource_target)
                        raise ValueError(f"Resource not exists in the vector database: {msg_resource_target}")

                if msg_resource_target in accumulator_dict:
                    accumulator: AccumulationStats = accumulator_dict[msg_resource_target]
                    accumulator.msg_count += df_size
                    accumulator.data.append(df)
                else:
                    accumulator_dict[msg_resource_target] = AccumulationStats(msg_count=df_size,
                                                                              last_insert_time=-1,
                                                                              data=[df])

                for key, accum_stats in accumulator_dict.items():
                    if accum_stats.msg_count >= batch_size or (accum_stats.last_insert_time != -1 and
                                                               (current_time - accum_stats.last_insert_time)
                                                               >= write_time_interval):
                        if accum_stats.data:
                            merged_df = cudf.concat(accum_stats.data)

                            # pylint: disable=not-a-mapping
                            collection = service.load_resource(name=key)
                            collection.insert_dataframe(df=merged_df, **resource_kwargs)
                            # Reset accumulator stats
                            accum_stats.data.clear()
                            accum_stats.last_insert_time = current_time
                            accum_stats.msg_count = 0

                        if (isinstance(msg, ControlMessage)):
                            msg.set_metadata(
                                "insert_response",
                                {
                                    "status": "inserted",
                                    "accum_count": 0,
                                    "insert_count": df_size,
                                    "succ_count": df_size,
                                    "err_count": 0
                                })
                    else:
                        logger.debug("Accumulated %d rows for collection: %s", accum_stats.msg_count, key)
                        if (isinstance(msg, ControlMessage)):
                            msg.set_metadata(
                                "insert_response",
                                {
                                    "status": "accumulated",
                                    "accum_count": df_size,
                                    "insert_count": 0,
                                    "succ_count": 0,
                                    "err_count": 0
                                })

                return msg

        except Exception as exc:
            logger.error("Unable to insert into collection: %s due to %s", msg_resource_target, exc)
            # TODO(Devin): This behavior is likely buggy; we need to decide whether or not to collect control messages
            # and output all of them when an accumulation is flushed, or to simply mark a control message as "done",
            # even if it is just accumulated.
            if (isinstance(msg, ControlMessage)):
                msg.set_metadata("insert_response", {"status": "failed", "err_count": 1})

        return msg

    node = builder.make_node(WRITE_TO_VECTOR_DB,
                             ops.map(on_data),
                             ops.filter(lambda val: val is not None),
                             ops.on_completed(on_completed))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
