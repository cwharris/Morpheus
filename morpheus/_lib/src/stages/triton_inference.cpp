/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/stages/triton_inference.hpp"

#include "common.h"
#include "grpc_client.h"
#include "grpc_service.pb.h"
#include "model_config.pb.h"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/objects/dev_mem_info.hpp"           // for DevMemInfo
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor.hpp"         // for Tensor::create
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/objects/triton_in_out.hpp"  // for TritonInOut
#include "morpheus/types.hpp"                  // for TensorIndex, TensorMap
#include "morpheus/utilities/matx_util.hpp"    // for MatxUtil::logits, MatxUtil::reduce_max
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR
#include "morpheus/utilities/tensor_util.hpp"  // for get_elem_count

#include <boost/fiber/policy.hpp>
#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice
#include <glog/logging.h>
#include <http_client.h>
#include <mrc/coroutines/when_all.hpp>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <nlohmann/json.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <algorithm>  // for min
#include <chrono>
#include <compare>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ratio>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>  // for runtime_error, out_of_range
#include <utility>
// IWYU pragma: no_include <initializer_list>

/**
 * @addtogroup stages
 * @{
 * @file
 */

namespace {

/**
 * @brief Checks the status object returned by a Triton client call logging any potential errors.
 *
 */
#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

// Component-private free functions.
void InferenceClientStage__check_triton_errors(triton::client::Error status,
                                               const std::string& methodName,
                                               const std::string& filename,
                                               const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg = MORPHEUS_CONCAT_STR("Triton Error while executing '"
                                                  << methodName << "'. Error: " + status.Message() << "\n"
                                                  << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

using namespace morpheus;
using buffer_map_t = std::map<std::string, std::shared_ptr<rmm::device_buffer>>;

static ShapeType get_seq_ids(const InferenceClientStage::sink_type_t& message)
{
    // Take a copy of the sequence Ids allowing us to map rows in the response to rows in the dataframe
    // The output tensors we store in `reponse_memory` will all be of the same length as the the
    // dataframe. seq_ids has three columns, but we are only interested in the first column.
    auto seq_ids         = message->get_input("seq_ids");
    const auto item_size = seq_ids.dtype().item_size();

    ShapeType host_seq_ids(message->count);
    MRC_CHECK_CUDA(cudaMemcpy2D(host_seq_ids.data(),
                                item_size,
                                seq_ids.data(),
                                seq_ids.stride(0) * item_size,
                                item_size,
                                host_seq_ids.size(),
                                cudaMemcpyDeviceToHost));

    return host_seq_ids;
}

static void reduce_outputs(const InferenceClientStage::sink_type_t& x, TensorMap& output_tensors)
{
    // When our tensor lengths are longer than our dataframe we will need to use the seq_ids array to
    // lookup how the values should map back into the dataframe.
    auto host_seq_ids = get_seq_ids(x);

    for (auto& mapping : output_tensors)
    {
        auto output_tensor = mapping.second;

        ShapeType shape  = output_tensor.get_shape();
        ShapeType stride = output_tensor.get_stride();

        ShapeType reduced_shape{shape};
        reduced_shape[0] = x->mess_count;

        auto rmm_tensor = reinterpret_cast<RMMTensor const*>(&output_tensor);

        CHECK_NOTNULL(rmm_tensor);

        auto reduced_buffer = MatxUtil::reduce_max(
            DevMemInfo{rmm_tensor->buffer(), output_tensor.dtype(), shape, stride}, host_seq_ids, 0, reduced_shape);

        output_tensor.swap(Tensor::create(std::move(reduced_buffer), output_tensor.dtype(), reduced_shape, stride, 0));
    }
}

static void apply_logits(TensorMap& output_tensors)
{
    for (auto& mapping : output_tensors)
    {
        auto output_tensor = mapping.second;

        auto shape  = output_tensor.get_shape();
        auto stride = output_tensor.get_stride();

        auto rmm_tensor = reinterpret_cast<RMMTensor const*>(&output_tensor);

        CHECK_NOTNULL(rmm_tensor);

        auto output_buffer = MatxUtil::logits(DevMemInfo{rmm_tensor->buffer(), output_tensor.dtype(), shape, stride});

        // For logits the input and output shapes will be the same
        output_tensor.swap(Tensor::create(std::move(output_buffer), output_tensor.dtype(), shape, stride, 0));
    }
}

struct TritonInferOperation
{
    bool await_ready() const noexcept
    {
        return false;
    }

    void await_suspend(std::coroutine_handle<> handle)
    {
        std::cout << "infer suspend" << std::endl;
        CHECK_TRITON(m_client.async_infer(
            [this, handle](triton::client::InferResult* result) {
                std::cout << "infer callback" << std::endl;
                m_result.reset(result);
                handle();
            },
            m_options,
            m_inputs,
            m_outputs));
    }

    std::unique_ptr<triton::client::InferResult> await_resume()
    {
        return std::move(m_result);
    }

    ITritonClient& m_client;
    triton::client::InferOptions m_options;
    std::vector<TritonInferInput> m_inputs;
    std::vector<TritonInferRequestedOutput> m_outputs;
    std::unique_ptr<triton::client::InferResult> m_result;
};

}  // namespace

namespace morpheus {

bool HttpTritonClient::is_default_grpc_port(std::string& server_url)
{
    // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
    size_t colon_loc = server_url.find_last_of(':');

    if (colon_loc == -1)
    {
        return false;
    }

    // Check if the port matches 8001
    if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001")
    {
        return false;
    }

    // It matches, change to 8000
    server_url = server_url.substr(0, colon_loc) + ":8000";

    return true;
}

HttpTritonClient::HttpTritonClient(std::string server_url)
{
    std::unique_ptr<triton::client::InferenceServerHttpClient> client;

    CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, server_url, false));

    bool is_server_live;

    auto status = client->IsServerLive(&is_server_live);

    if (not status.IsOk())
    {
        std::string new_server_url = server_url;
        if (is_default_grpc_port(new_server_url))
        {
            LOG(WARNING) << "Failed to connect to Triton at '" << server_url
                         << "'. Default gRPC port of (8001) was detected but C++ "
                            "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

            // We are using the default gRPC port, try the default HTTP
            std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

            CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&unique_client, new_server_url, false));

            client = std::move(unique_client);

            status = client->IsServerLive(&is_server_live);
        }
        else if (status.Message().find("Unsupported protocol") != std::string::npos)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR(
                "Failed to connect to Triton at '"
                << server_url
                << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                   "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                   "specified the HTTP port (Default 8000)."));
        }

        if (not status.IsOk())
            throw std::runtime_error(
                MORPHEUS_CONCAT_STR("Unable to connect to Triton at '"
                                    << server_url << "'. Check the URL and port and ensure the server is running."));
    }

    m_client = std::move(client);
}

bool HttpTritonClient::is_server_live()
{
    bool live;
    CHECK_TRITON(m_client->IsServerLive(&live));
    return live;
}

bool HttpTritonClient::is_server_ready()
{
    bool ready;
    CHECK_TRITON(m_client->IsServerReady(&ready));
    return ready;
}

bool HttpTritonClient::is_model_ready(std::string& model_name)
{
    bool ready;
    CHECK_TRITON(m_client->IsModelReady(&ready, model_name));
    return ready;
}

TritonModelInfo HttpTritonClient::model_info(std::string& model_name)
{
    std::string model_metadata_json;
    CHECK_TRITON(m_client->ModelMetadata(&model_metadata_json, model_name));

    auto model_metadata = nlohmann::json::parse(model_metadata_json);

    std::string model_config_json;
    CHECK_TRITON(m_client->ModelConfig(&model_config_json, model_name));
    auto model_config = nlohmann::json::parse(model_config_json);

    std::vector<TritonInOut> model_inputs;
    std::vector<TritonInOut> model_outputs;

    TensorIndex max_batch_size;

    if (model_config.contains("max_batch_size"))
    {
        max_batch_size = model_config.at("max_batch_size").get<TensorIndex>();
    }

    for (auto const& input : model_metadata.at("inputs"))
    {
        auto shape = input.at("shape").get<ShapeType>();

        auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = max_batch_size;
            }

            bytes *= y;
        }

        auto name = input.at("name").get<std::string>();

        model_inputs.push_back(TritonInOut{name, bytes, dtype, shape, "", 0});
    }

    for (auto const& output : model_metadata.at("outputs"))
    {
        auto shape = output.at("shape").get<ShapeType>();

        auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = max_batch_size;
            }

            bytes *= y;
        }

        model_outputs.push_back(TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, "", 0});
    }

    return TritonModelInfo{std::move(model_inputs), std::move(model_outputs), max_batch_size};
}

triton::client::Error HttpTritonClient::async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                                    const triton::client::InferOptions& options,
                                                    const std::vector<TritonInferInput>& inputs,
                                                    const std::vector<TritonInferRequestedOutput>& outputs)
{
    std::vector<std::unique_ptr<triton::client::InferInput>> inference_inputs;
    std::vector<triton::client::InferInput*> inference_input_ptrs;

    for (auto& input : inputs)
    {
        triton::client::InferInput* inference_input_ptr;
        triton::client::InferInput::Create(&inference_input_ptr, input.name, input.shape, input.type);

        inference_input_ptr->AppendRaw(input.data);

        inference_input_ptrs.emplace_back(inference_input_ptr);
        inference_inputs.emplace_back(inference_input_ptr);
    }

    std::vector<std::unique_ptr<const triton::client::InferRequestedOutput>> inference_outputs;
    std::vector<const triton::client::InferRequestedOutput*> inference_output_ptrs;

    for (auto& output : outputs)
    {
        triton::client::InferRequestedOutput* inference_output_ptr;
        triton::client::InferRequestedOutput::Create(&inference_output_ptr, output.name);
        inference_output_ptrs.emplace_back(inference_output_ptr);
        inference_outputs.emplace_back(inference_output_ptr);
    }

    return m_client->AsyncInfer(
        [&inference_inputs, &inference_outputs, callback](triton::client::InferResult* result) {
            callback(result);
        },
        options,
        inference_input_ptrs,
        inference_output_ptrs);
}

GrpcTritonClient::GrpcTritonClient(std::string server_url)
{
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client;

    CHECK_TRITON(triton::client::InferenceServerGrpcClient::Create(&client, server_url, false));

    m_client = std::move(client);
}

bool GrpcTritonClient::is_server_live()
{
    bool live;
    CHECK_TRITON(this->m_client->IsServerLive(&live));
    return live;
}

bool GrpcTritonClient::is_server_ready()
{
    bool ready;
    CHECK_TRITON(this->m_client->IsServerReady(&ready));
    return ready;
}

bool GrpcTritonClient::is_model_ready(std::string& model_name)
{
    bool ready;
    CHECK_TRITON(this->m_client->IsModelReady(&ready, model_name));
    return ready;
}

TritonModelInfo GrpcTritonClient::model_info(std::string& model_name)
{
    inference::ModelConfigResponse config_response;
    inference::ModelMetadataResponse metadata_response;

    CHECK_TRITON(m_client->ModelConfig(&config_response, model_name));
    CHECK_TRITON(m_client->ModelMetadata(&metadata_response, model_name));

    auto max_batch_size = config_response.config().max_batch_size();

    std::vector<TritonInOut> model_inputs;
    std::vector<TritonInOut> model_outputs;

    for (auto i = 0; i < metadata_response.inputs_size(); i++)
    {
        auto& input      = metadata_response.inputs(i);
        auto shape_begin = input.shape().begin();
        auto shape_end   = input.shape().end();
        auto shape       = std::vector<int>(shape_begin, shape_end);
        auto dtype       = DType::from_triton(input.datatype());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = max_batch_size;
            }

            bytes *= y;
        }

        model_inputs.push_back(TritonInOut{input.name(), bytes, dtype, shape, "", 0});
    }

    for (auto i = 0; i < metadata_response.outputs_size(); i++)
    {
        auto& output     = metadata_response.outputs(i);
        auto shape_begin = output.shape().begin();
        auto shape_end   = output.shape().end();
        auto shape       = std::vector<int>(shape_begin, shape_end);
        auto dtype       = DType::from_triton(output.datatype());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = max_batch_size;
            }

            bytes *= y;
        }

        model_outputs.push_back(TritonInOut{output.name(), bytes, dtype, shape, "", 0});
    }

    return TritonModelInfo{std::move(model_inputs), std::move(model_outputs), max_batch_size};
}

triton::client::Error GrpcTritonClient::async_infer(triton::client::InferenceServerGrpcClient::OnCompleteFn callback,
                                                    const triton::client::InferOptions& options,
                                                    const std::vector<TritonInferInput>& inputs,
                                                    const std::vector<TritonInferRequestedOutput>& outputs)
{
    std::vector<std::unique_ptr<triton::client::InferInput>> inference_inputs;
    std::vector<triton::client::InferInput*> inference_input_ptrs;

    for (auto& input : inputs)
    {
        triton::client::InferInput* inference_input_ptr;
        triton::client::InferInput::Create(&inference_input_ptr, input.name, input.shape, input.type);

        inference_input_ptr->AppendRaw(input.data);

        inference_input_ptrs.emplace_back(inference_input_ptr);
        inference_inputs.emplace_back(inference_input_ptr);
    }

    std::vector<std::unique_ptr<const triton::client::InferRequestedOutput>> inference_outputs;
    std::vector<const triton::client::InferRequestedOutput*> inference_output_ptrs;

    for (auto& output : outputs)
    {
        triton::client::InferRequestedOutput* inference_output_ptr;
        triton::client::InferRequestedOutput::Create(&inference_output_ptr, output.name);
        inference_output_ptrs.emplace_back(inference_output_ptr);
        inference_outputs.emplace_back(inference_output_ptr);
    }

    return m_client->AsyncInfer(
        [&inference_inputs, &inference_outputs, callback](triton::client::InferResult* result) {
            callback(result);
        },
        options,
        inference_input_ptrs,
        inference_output_ptrs);
}

TritonInferenceClientSession::TritonInferenceClientSession(std::shared_ptr<ITritonClient> client,
                                                           std::string model_name) :
  m_client(client),
  m_model_name(std::move(model_name))
{
    // Now load the input/outputs for the model

    if (not m_client->is_server_live())
    {
        throw std::runtime_error("Server is not live");
    }

    if (not m_client->is_server_ready())
    {
        throw std::runtime_error("Server is not ready");
    }

    if (not m_client->is_model_ready(this->m_model_name))
    {
        throw std::runtime_error("Model is not ready");
    }

    auto info = m_client->model_info(this->m_model_name);

    m_model_inputs   = std::move(info.inputs);
    m_model_outputs  = std::move(info.outputs);
    m_max_batch_size = info.max_batch_size;
}

std::map<std::string, std::string> TritonInferenceClientSession::get_input_mappings(
    std::map<std::string, std::string> input_map_overrides)
{
    auto mappings = std::map<std::string, std::string>();

    for (auto map : m_model_inputs)
    {
        mappings[map.name] = map.name;
    }

    for (auto override : input_map_overrides)
    {
        auto pos = mappings.find(override.second);

        if (pos == mappings.end())
        {
            LOG(WARNING) << "Input mapping was provided for '" << override.first << "' -> '" << override.second
                         << "' but the input does not exist for this model.";
            continue;
        }

        mappings.erase(pos);
        mappings[override.first] = override.second;
    }

    return mappings;
};

std::map<std::string, std::string> TritonInferenceClientSession::get_output_mappings(
    std::map<std::string, std::string> output_map_overrides)
{
    auto mappings = std::map<std::string, std::string>();

    for (auto map : m_model_outputs)
    {
        mappings[map.name] = map.name;
    }

    for (auto override : output_map_overrides)
    {
        auto pos = mappings.find(override.first);

        if (pos == mappings.end())
        {
            LOG(WARNING) << "Output mapping was provided for '" << override.first << "' -> '" << override.second
                         << "' but the output does not exist for this model.";
            continue;
        }

        mappings.erase(pos);
        mappings[override.first] = override.second;
    }

    return mappings;
}

mrc::coroutines::Task<TensorMap> TritonInferenceClientSession::infer(TensorMap&& inputs)
{
    if (inputs.size() == 0)
    {
        co_return inputs;
    }

    CHECK_EQ(inputs.size(), m_model_inputs.size()) << "Input tensor count does not match model input count";

    auto element_count = inputs.begin()->second.shape(0);

    for (auto& input : inputs)
    {
        CHECK_EQ(element_count, input.second.shape(0)) << "Input tensors are different sizes";
    }

    TensorMap output_tensors;
    std::vector<std::shared_ptr<rmm::device_buffer>> output_buffers;

    // create full inference output
    for (auto& model_output : m_model_outputs)
    {
        ShapeType full_output_shape    = model_output.shape;
        full_output_shape[0]           = element_count;
        auto full_output_element_count = TensorUtils::get_elem_count(full_output_shape);

        auto full_output_buffer = std::make_shared<rmm::device_buffer>(
            full_output_element_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

        output_buffers.emplace_back(full_output_buffer);

        ShapeType stride{full_output_shape[1], 1};

        output_tensors[model_output.name].swap(
            Tensor::create(std::move(full_output_buffer), model_output.datatype, full_output_shape, stride, 0));
    }

    // process all batches

    std::vector<Task<std::unique_ptr<triton::client::InferResult>>> inference_tasks;

    for (TensorIndex start = 0; start < element_count; start += m_max_batch_size)
    {
        TensorIndex stop = std::min(start + m_max_batch_size, static_cast<TensorIndex>(element_count));

        // create batch inputs

        std::vector<TritonInferInput> inference_inputs;

        for (auto model_input : m_model_inputs)
        {
            auto inference_input_slice =
                inputs[model_input.name].slice({start, 0}, {stop, -1}).as_type(model_input.datatype);

            inference_inputs.emplace_back(
                TritonInferInput{model_input.name,
                                 {inference_input_slice.shape(0), inference_input_slice.shape(1)},
                                 model_input.datatype.triton_str(),
                                 inference_input_slice.get_host_data()});
        }

        // create batch outputs

        std::vector<TritonInferRequestedOutput> outputs;

        for (auto model_output : m_model_outputs)
        {
            outputs.emplace_back(TritonInferRequestedOutput{model_output.name});
        }

        // infer batch results

        auto options = triton::client::InferOptions(m_model_name);

        auto operation =
            TritonInferOperation(*m_client, std::move(options), std::move(inference_inputs), std::move(outputs));

        inference_tasks.emplace_back([](auto operation) -> Task<std::unique_ptr<triton::client::InferResult>> {
            co_return co_await operation;
        }(std::move(operation)));
    }

    auto inference_results = co_await mrc::coroutines::when_all(std::move(inference_tasks));

    for (TensorIndex start = 0, i = 0; start < element_count; start += m_max_batch_size, i++)
    {
        TensorIndex stop = std::min(start + m_max_batch_size, static_cast<TensorIndex>(element_count));

        auto& results = inference_results[i].return_value();

        // verify batch results and copy to full output tensors

        for (auto model_output : m_model_outputs)
        {
            auto output_tensor = output_tensors[model_output.name].slice({start, 0}, {stop, -1});

            std::vector<int64_t> output_shape;

            CHECK_TRITON(results->Shape(model_output.name, &output_shape));  // Make sure we have at least 2 dims

            while (output_shape.size() < 2)
            {
                output_shape.push_back(1);
            }

            const uint8_t* output_ptr = nullptr;
            size_t output_ptr_size    = 0;
            CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

            DCHECK_EQ(stop - start, output_shape[0]);
            DCHECK_EQ(output_tensor.bytes(), output_ptr_size);
            DCHECK_NOTNULL(output_ptr);
            DCHECK_NOTNULL(output_tensor.data());

            MRC_CHECK_CUDA(cudaMemcpy(output_tensor.data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));
        }
    }

    co_return output_tensors;
};

TritonInferenceClient::TritonInferenceClient(std::unique_ptr<ITritonClient>&& client, std::string model_name) :
  m_client(std::move(client)),
  m_model_name(std::move(model_name))
{}

std::shared_ptr<IInferenceClientSession> TritonInferenceClient::get_session()
{
    if (m_session == nullptr)
    {
        m_session = std::make_shared<TritonInferenceClientSession>(m_client, m_model_name);
    }

    return reinterpret_pointer_cast<IInferenceClientSession>(m_session);
}

void TritonInferenceClient::reset_session()
{
    m_session.reset();
}

// Component public implementations
// ************ InferenceClientStage ************************* //
InferenceClientStage::InferenceClientStage(std::unique_ptr<IInferenceClient>&& client,
                                           std::string model_name,
                                           bool needs_logits,
                                           std::map<std::string, std::string> input_mapping,
                                           std::map<std::string, std::string> output_mapping) :
  m_model_name(std::move(model_name)),
  m_client(std::move(client)),
  m_needs_logits(needs_logits),
  m_input_mapping(std::move(input_mapping)),
  m_output_mapping(std::move(output_mapping))
{}

using namespace std::chrono_literals;
struct ExponentialBackoff
{
    std::shared_ptr<mrc::coroutines::Scheduler> m_on;
    std::chrono::milliseconds m_delay;
    std::chrono::milliseconds m_delay_max;

    ExponentialBackoff(std::shared_ptr<mrc::coroutines::Scheduler> on,
                       std::chrono::milliseconds delay_initial,
                       std::chrono::milliseconds delay_max) :
      m_on(std::move(on)),
      m_delay(delay_initial),
      m_delay_max(delay_max)
    {}

    mrc::coroutines::Task<> yield()
    {
        if (m_delay > m_delay_max)
        {
            m_delay = m_delay_max;
        }

        co_await m_on->yield_for(m_delay);

        m_delay *= 2;
    }
};

mrc::coroutines::AsyncGenerator<std::shared_ptr<MultiResponseMessage>> InferenceClientStage::on_data(
    std::shared_ptr<MultiInferenceMessage>&& x, std::shared_ptr<mrc::coroutines::Scheduler> on)
{
    int32_t retry_count = 0;

    auto backoff = ExponentialBackoff(on, 100ms, 4000ms);

    while (true)
    {
        try
        {
            // Using the `count` which is the number of rows in the inference tensors. We will check later if this
            // doesn't match the number of rows in the dataframe (`mess_count`). This happens when the size of the
            // input is too large and needs to be broken up in chunks in the pre-process stage. When this is the
            // case we will reduce the rows in the response outputs such that we have a single response for each
            // row int he dataframe.
            // TensorMap output_tensors;
            // buffer_map_t output_buffers;

            auto lock = std::shared_lock(m_session_mutex);

            auto session = m_client->get_session();

            TensorMap input_tensors;

            for (auto mapping : session->get_input_mappings(m_input_mapping))
            {
                CHECK(x->memory->has_tensor(mapping.first))
                    << "Model input '" << mapping.first << "' not found in InferenceMemory";

                input_tensors[mapping.second].swap(x->get_input(mapping.first));
            }

            // TODO(cwharris): Break inference in to batches and attempt retries on per-batch basis.
            auto output_tensors = co_await session->infer(std::move(input_tensors));

            co_await on->yield();

            if (x->mess_count != x->count)
            {
                reduce_outputs(x, output_tensors);
            }

            // If we need to do logits, do that here
            if (m_needs_logits)
            {
                apply_logits(output_tensors);
            }

            TensorMap output_tensor_map;

            for (auto mapping : session->get_output_mappings(m_output_mapping))
            {
                output_tensor_map[mapping.second].swap(std::move(output_tensors[mapping.first]));
            }

            // Final output of all mini-batches
            auto response_mem = std::make_shared<ResponseMemory>(x->mess_count, std::move(output_tensor_map));

            auto response = std::make_shared<MultiResponseMessage>(
                x->meta, x->mess_offset, x->mess_count, std::move(response_mem), 0, response_mem->count);

            co_yield std::move(response);

            co_return;

        } catch (...)
        {
            auto lock = std::unique_lock(m_session_mutex);

            this->m_client->reset_session();

            if (m_retry_max >= 0 and ++retry_count > m_retry_max)
            {
                throw;
            }

            LOG(WARNING) << "Exception while processing message for InferenceClientStage, attempting retry.";
        }

        co_await backoff.yield();
    }
}

// ************ InferenceClientStageInterfaceProxy********* //
std::shared_ptr<mrc::segment::Object<InferenceClientStage>> InferenceClientStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string server_url,
    std::string model_name,
    bool needs_logits,
    std::map<std::string, std::string> input_mapping,
    std::map<std::string, std::string> output_mapping)
{
    auto triton_client           = std::make_unique<GrpcTritonClient>(server_url);
    auto triton_inference_client = std::make_unique<TritonInferenceClient>(std::move(triton_client), model_name);
    auto stage                   = builder.construct_object<InferenceClientStage>(
        name, std::move(triton_inference_client), model_name, needs_logits, input_mapping, output_mapping);

    return stage;
}

}  // namespace morpheus
