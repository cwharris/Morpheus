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

class MyFlatmapStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "flatmap"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MultiMessage):
        results = []
        for count in message.get_meta("count").to_pandas():
            for i in range(count):
                new_df = cudf.DataFrame({"value": [i]})
                new_meta = MessageMeta(new_df)
                new_message = MultiMessage(meta=new_meta)
                results.append(new_message)
        return results

    def _build_single(self, builder: mrc.Builder, input: StreamPair) -> StreamPair:
        [input_node, input_type] = input
        node = builder.make_node(self.unique_name, mrc.operators.flatmap(self.on_data))
        builder.make_edge(input_node, node)
        return node, input_type

def test_flatmap():

    input_df = cudf.DataFrame({ "count": [5] })
    config = Config()
    pipeline = LinearPipeline(config)

    pipeline.set_source(InMemorySourceStage(config, [input_df]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MyFlatmapStage(config))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()

    for message in messages:
        print(message.copy_dataframe())