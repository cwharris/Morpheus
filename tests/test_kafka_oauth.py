
from _utils import assert_results
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage

import cudf

def test_kafka_oauth(config):
    print("hello")

    expected = cudf.DataFrame({ "a": "b" })

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers="kafka:9092",
                         input_topic="morpheus_input_topic",
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         client_id='morpheus_kafka_source_stage_pipe',
                         stop_after=1))

    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected))

    pipe.run()

    assert_results(comp_stage.get_results())

