from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
import morpheus._lib.stages as _stages
import mrc
import typing

# @register_stage("inf-basic", modes=[PipelineModes.NLP, PipelineModes.FIL, PipelineModes.OTHER])
class BasicInferenceStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c: Config):
        super().__init__(c)
        pass

    @property
    def name(self) -> str:
        return "inf-basic"

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return True

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = _stages.BasicInferenceStage(builder, self.unique_name)
        node.launch_options.pe_count = 1

        builder.make_edge(input_node, node)

        return node
