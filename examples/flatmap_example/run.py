from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.messages import MessageMeta, MultiMessage

from morpheus.pipeline import LinearPipeline
from morpheus.config import Config

import cudf
import mrc
import typing
import asyncio

class MyFlatmapStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "flatmap"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    async def on_data_async_gen(self, message: MultiMessage):
        for duration in message.get_meta("sleep").to_pandas():
            await asyncio.sleep(duration)
            new_series = message.get_meta("value")
            new_df = cudf.DataFrame({"value": new_series})
            new_meta = MessageMeta(new_df)
            new_message = MultiMessage(meta=new_meta)
            yield new_message

    def _build_single(self, builder: mrc.Builder, input: StreamPair) -> StreamPair:
        [input_node, input_type] = input
        node = builder.make_node(self.unique_name, mrc.operators.flatmap_async(self.on_data_async_gen))
        builder.make_edge(input_node, node)
        return node, input_type

def test_flatmap():

    input_df_a = cudf.DataFrame({ "sleep": [1], "value": [0] })
    input_df_b = cudf.DataFrame({ "sleep": [2], "value": [1] })
    input_df_c = cudf.DataFrame({ "sleep": [1], "value": [2] })
    config = Config()
    pipeline = LinearPipeline(config)

    pipeline.set_source(InMemorySourceStage(config, [input_df_a, input_df_b, input_df_c]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MyFlatmapStage(config))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()

    for message in messages:
        print(message.copy_dataframe())