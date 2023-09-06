import mrc
import cudf
import typing

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.messages.message_meta import MessageMeta

class NoopStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "noop"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def _subscribe(self, observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.subscribe(subscriber)
    
    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        [input_node, input_type] = input_stream
        node = builder.make_node(self.unique_name, mrc.core.operators.build(self._subscribe))
        builder.make_edge(input_node, node)
        return node, input_type

def run_pipeline():

    input_df = cudf.DataFrame({"a": [0]})

    config = Config()
    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [input_df]))
    pipeline.add_stage(NoopStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))
    pipeline.run()

    messages: list[MessageMeta] = sink.get_messages()
    
    print(messages[0].copy_dataframe())

if __name__ == '__main__':
    run_pipeline()
