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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"  // for MultiResponseMessage
#include "morpheus/objects/triton_in_out.hpp"
#include "morpheus/types.hpp"

#include <grpc_client.h>
#include <http_client.h>
#include <mrc/coroutines/async_generator.hpp>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/pybind11.h>
#include <pymrc/asyncio_runnable.hpp>
#include <cstdint>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceClientStage********************************/

struct MORPHEUS_EXPORT TritonInferInput
{
    std::string name;
    std::vector<int64_t> shape;
    std::string type;
    std::vector<uint8_t> data;
};

struct MORPHEUS_EXPORT TritonInferRequestedOutput
{
    std::string name;
};

struct MORPHEUS_EXPORT TritonModelInfo
{
    std::vector<TritonInOut> inputs;
    std::vector<TritonInOut> outputs;
    TensorIndex max_batch_size;
};

class MORPHEUS_EXPORT ITritonClient
{
  public:
    virtual bool is_server_live() = 0;

    virtual bool is_server_ready() = 0;

    virtual bool is_model_ready(std::string& model_name) = 0;

    virtual TritonModelInfo model_info(std::string& model_name) = 0;

    virtual triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                              const triton::client::InferOptions& options,
                                              const std::vector<TritonInferInput>& inputs,
                                              const std::vector<TritonInferRequestedOutput>& outputs) = 0;
};

class MORPHEUS_EXPORT IInferenceClientSession
{
  public:
    virtual std::map<std::string, std::string> get_input_mappings(
        std::map<std::string, std::string> input_map_overrides) = 0;

    virtual std::map<std::string, std::string> get_output_mappings(
        std::map<std::string, std::string> output_map_overrides) = 0;

    virtual mrc::coroutines::Task<TensorMap> infer(TensorMap&& inputs) = 0;
};

class MORPHEUS_EXPORT IInferenceClient
{
  public:
    virtual std::shared_ptr<IInferenceClientSession> get_session() = 0;
    virtual void reset_session()                                   = 0;
};

class MORPHEUS_EXPORT HttpTritonClient : public ITritonClient
{
  private:
    std::unique_ptr<triton::client::InferenceServerHttpClient> m_client;
    static bool is_default_grpc_port(std::string& server_url);

  public:
    HttpTritonClient(std::string server_url);

    bool is_server_live() override;

    bool is_server_ready() override;

    bool is_model_ready(std::string& model_name) override;

    TritonModelInfo model_info(std::string& model_name) override;

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<TritonInferInput>& inputs,
                                      const std::vector<TritonInferRequestedOutput>& outputs) override;
};

class MORPHEUS_EXPORT GrpcTritonClient : public ITritonClient
{
  private:
    std::unique_ptr<triton::client::InferenceServerGrpcClient> m_client;
    static bool is_default_grpc_port(std::string& server_url);

  public:
    GrpcTritonClient(std::string server_url);

    bool is_server_live() override;

    bool is_server_ready() override;

    bool is_model_ready(std::string& model_name) override;

    TritonModelInfo model_info(std::string& model_name) override;

    triton::client::Error async_infer(triton::client::InferenceServerGrpcClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<TritonInferInput>& inputs,
                                      const std::vector<TritonInferRequestedOutput>& outputs) override;
};

class MORPHEUS_EXPORT TritonInferenceClientSession : public IInferenceClientSession
{
  private:
    std::string m_model_name;
    TensorIndex m_max_batch_size = -1;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    std::shared_ptr<ITritonClient> m_client;

  public:
    TritonInferenceClientSession(std::shared_ptr<ITritonClient> client, std::string model_name);

    std::map<std::string, std::string> get_input_mappings(
        std::map<std::string, std::string> input_map_overrides) override;

    std::map<std::string, std::string> get_output_mappings(
        std::map<std::string, std::string> output_map_overrides) override;

    mrc::coroutines::Task<TensorMap> infer(TensorMap&& inputs) override;
};

class MORPHEUS_EXPORT TritonInferenceClient : public IInferenceClient
{
  private:
    std::shared_ptr<ITritonClient> m_client;
    std::string m_model_name;
    std::shared_ptr<TritonInferenceClientSession> m_session;

  public:
    TritonInferenceClient(std::unique_ptr<ITritonClient>&& client, std::string model_name);

    std::shared_ptr<IInferenceClientSession> get_session() override;

    void reset_session() override;
};

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Perform inference with Triton Inference Server.
 * This class specifies which inference implementation category (Ex: NLP/FIL) is needed for inferencing.
 */
class MORPHEUS_EXPORT InferenceClientStage
  : public mrc::pymrc::AsyncioRunnable<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using sink_type_t   = std::shared_ptr<MultiInferenceMessage>;
    using source_type_t = std::shared_ptr<MultiResponseMessage>;

    /**
     * @brief Construct a new Inference Client Stage object
     *
     * @param client : Inference client instance.
     * @param model_name : Name of the model specifies which model can handle the inference requests that are sent to
     * Triton inference
     * @param needs_logits : Determines if logits are required.
     * @param inout_mapping : Dictionary used to map pipeline input/output names to Triton input/output names. Use this
     * if the Morpheus names do not match the model.
     */
    InferenceClientStage(std::unique_ptr<IInferenceClient>&& client,
                         std::string model_name,
                         bool needs_logits,
                         std::map<std::string, std::string> input_mapping  = {},
                         std::map<std::string, std::string> output_mapping = {});

    /**
     * TODO(Documentation)
     */
    mrc::coroutines::AsyncGenerator<std::shared_ptr<MultiResponseMessage>> on_data(
        std::shared_ptr<MultiInferenceMessage>&& data, std::shared_ptr<mrc::coroutines::Scheduler> on) override;

  private:
    std::string m_model_name;
    std::shared_ptr<IInferenceClient> m_client;
    bool m_needs_logits{true};
    std::map<std::string, std::string> m_input_mapping;
    std::map<std::string, std::string> m_output_mapping;
    std::shared_mutex m_session_mutex;

    int32_t m_retry_max = 0;
};

/****** InferenceClientStageInferenceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT InferenceClientStageInterfaceProxy
{
    /**
     * @brief Create and initialize a InferenceClientStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param model_name : Name of the model specifies which model can handle the inference requests that are sent to
     * Triton inference
     * @param server_url : Triton server URL.
     * @param needs_logits : Determines if logits are required.
     * @param inout_mapping : Dictionary used to map pipeline input/output names to Triton input/output names. Use this
     * if the Morpheus names do not match the model.
     * @return std::shared_ptr<mrc::segment::Object<InferenceClientStage>>
     */
    static std::shared_ptr<mrc::segment::Object<InferenceClientStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::string model_name,
        std::string server_url,
        bool needs_logits,
        std::map<std::string, std::string> input_mapping,
        std::map<std::string, std::string> output_mapping);
};
/** @} */  // end of group
}  // namespace morpheus
