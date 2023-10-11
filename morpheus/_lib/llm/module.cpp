/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./include/py_llm_engine.hpp"
#include "./include/py_llm_node.hpp"
#include "./include/py_llm_node_base.hpp"
#include "./include/py_llm_task_handler.hpp"
#include "py_llm_lambda_node.hpp"
#include "pycoro/pycoro.hpp"

#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_node_base.hpp"
#include "morpheus/llm/llm_node_runner.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/pybind11/json.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/version.hpp"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>
#include <mrc/utils/string_utils.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/attr.h>  // for multiple_inheritance
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pybind11/stl.h>
#include <pymrc/types.hpp>
#include <pymrc/utilities/acquire_gil.hpp>
#include <pymrc/utilities/function_wrappers.hpp>
#include <pymrc/utilities/object_wrappers.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <chrono>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace morpheus::llm {
namespace py = pybind11;

PYBIND11_MODULE(llm, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    CudfHelper::load();

    // Import the pycoro module
    py::module_ pycoro = py::module_::import("morpheus._lib.pycoro");

    py::class_<InputMap>(_module, "InputMap")
        .def(py::init<>())
        .def_readwrite("external_name",
                       &InputMap::external_name,
                       "The name of node that will be mapped to this input. Use a leading '/' to indicate it is a "
                       "sibling node otherwise it will be treated as a parent node. Can also specify a specific node "
                       "output such as '/sibling_node/output1' to map the output 'output1' of 'sibling_node' to this "
                       "input. Can also use a wild card such as '/sibling_node/*' to match all internal node names")
        .def_readwrite("internal_name",
                       &InputMap::internal_name,
                       "The internal node name that the external node maps to. Must match an input returned from "
                       "`get_input_names()` of the desired node. Defaults to '-' which is a placeholder for the "
                       "default input of the node. Use a wildcard '*' to match all inputs of the node (Must also use a "
                       "wild card on the external mapping).");

    py::class_<LLMTask>(_module, "LLMTask")
        .def(py::init<>())
        .def(py::init([](std::string task_type, py::dict task_dict) {
                 return LLMTask(std::move(task_type), mrc::pymrc::cast_from_pyobject(task_dict));
             }),
             py::arg("task_type"),
             py::arg("task_dict"))
        .def_readonly("task_type", &LLMTask::task_type)
        .def(
            "__getitem__",
            [](const LLMTask& self, const std::string& key) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const std::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"))
        .def(
            "__setitem__",
            [](LLMTask& self, const std::string& key, py::object value) {
                try
                {
                    // Convert to C++ nholman object
                    return self.set(key, mrc::pymrc::cast_from_pyobject(std::move(value)));
                } catch (const std::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"),
            py::arg("value"))
        .def("__len__", &LLMTask::size)
        .def(
            "get",
            [](const LLMTask& self, const std::string& key) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const nlohmann::detail::out_of_range&)
                {
                    throw py::key_error("key '" + key + "' does not exist");
                }
            },
            py::arg("key"))
        .def(
            "get",
            [](const LLMTask& self, const std::string& key, py::object default_value) {
                try
                {
                    return mrc::pymrc::cast_from_json(self.get(key));
                } catch (const nlohmann::detail::out_of_range&)
                {
                    return default_value;
                }
            },
            py::arg("key"),
            py::arg("default_value"));
    // .def(
    //     "__iter__",
    //     [](const LLMTask& self) {
    //         return py::detail::make_iterator_impl<iterator_key_access<typename Iterator>,
    //                                               return_value_policy Policy,
    //                                               typename Iterator,
    //                                               typename Sentinel,
    //                                               typename ValueType>(self.task_dict.items().begin(),
    //                                                                   self.task_dict.items().end());
    //         self.task_dict.items().begin().begin() return py::make_key_iterator(map.begin(), map.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "items",
    //     [](const LLMTask& self) {
    //         return py::make_iterator(self.begin(), self.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "values",
    //     [](const LLMTask& map) {
    //         return py::make_value_iterator(self., self.end());
    //     },
    //     py::keep_alive<0, 1>());

    py::class_<LLMContext, std::shared_ptr<LLMContext>>(_module, "LLMContext")
        .def_property_readonly("name", &LLMContext::name)
        .def_property_readonly("full_name", &LLMContext::full_name)
        .def_property_readonly("view_outputs", &LLMContext::view_outputs)
        .def_property_readonly("input_map", &LLMContext::input_map)
        .def_property_readonly("parent", &LLMContext::parent)
        .def("task", &LLMContext::task)
        .def("message", &LLMContext::message)
        .def("get_input", py::overload_cast<>(&LLMContext::get_input, py::const_))
        .def("get_input",
             py::overload_cast<const std::string&>(&LLMContext::get_input, py::const_),
             py::arg("node_name"))
        .def("get_inputs",
             [](LLMContext& self) {
                 // Convert the return value
                 return mrc::pymrc::cast_from_json(self.get_inputs()).cast<py::dict>();
             })
        .def("set_output", py::overload_cast<nlohmann::json>(&LLMContext::set_output), py::arg("outputs"))
        .def("set_output",
             py::overload_cast<const std::string&, nlohmann::json>(&LLMContext::set_output),
             py::arg("output_name"),
             py::arg("output"));

    py::class_<LLMNodeBase, PyLLMNodeBase<>, std::shared_ptr<LLMNodeBase>>(_module, "LLMNodeBase")
        .def(py::init_alias<>())
        .def("get_input_names", &LLMNodeBase::get_input_names)
        .def("execute", &LLMNodeBase::execute, py::arg("context"));

    py::class_<LLMNodeRunner, std::shared_ptr<LLMNodeRunner>>(_module, "LLMNodeRunner")
        .def_property_readonly("inputs", &LLMNodeRunner::inputs)
        .def_property_readonly("name", &LLMNodeRunner::name)
        .def_property_readonly("parent_input_names", &LLMNodeRunner::parent_input_names)
        .def_property_readonly("sibling_input_names", &LLMNodeRunner::sibling_input_names)
        .def("execute", &LLMNodeRunner::execute, py::arg("context"));

    py::class_<LLMNode, LLMNodeBase, PyLLMNode<>, std::shared_ptr<LLMNode>>(_module, "LLMNode")
        .def(py::init_alias<>())
        .def(
            "add_node",
            [](LLMNode& self, std::string name, std::shared_ptr<LLMNodeBase> node, bool is_output) {
                input_mapping_t converted_inputs;

                // Populate the inputs from the node input_names
                for (const auto& single_input : node->get_input_names())
                {
                    converted_inputs.push_back({.external_name = "/" + single_input});
                }

                return self.add_node(std::move(name), std::move(converted_inputs), std::move(node), is_output);
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("node"),
            py::arg("is_output") = false)
        .def(
            "add_node",
            [](LLMNode& self,
               std::string name,
               std::vector<std::variant<std::string, std::pair<std::string, std::string>>> inputs,
               std::shared_ptr<LLMNodeBase> node,
               bool is_output) {
                input_mapping_t converted_inputs;

                for (const auto& single_input : inputs)
                {
                    if (std::holds_alternative<std::string>(single_input))
                    {
                        converted_inputs.push_back({.external_name = std::get<std::string>(single_input)});
                    }
                    else
                    {
                        auto pair = std::get<std::pair<std::string, std::string>>(single_input);

                        converted_inputs.push_back({.external_name = pair.first, .internal_name = pair.second});
                    }
                }

                return self.add_node(std::move(name), std::move(converted_inputs), std::move(node), is_output);
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("inputs"),
            py::arg("node"),
            py::arg("is_output") = false);

    py::class_<LLMTaskHandler, PyLLMTaskHandler, std::shared_ptr<LLMTaskHandler>>(_module, "LLMTaskHandler")
        .def(py::init<>())
        .def("try_handle", &LLMTaskHandler::try_handle, py::arg("context"));

    py::class_<LLMEngine, LLMNode, PyLLMEngine, std::shared_ptr<LLMEngine>>(_module, "LLMEngine")
        .def(py::init_alias<>())
        .def(
            "add_task_handler",
            [](LLMEngine& self,
               std::vector<std::variant<std::string, std::pair<std::string, std::string>>> inputs,
               std::shared_ptr<LLMTaskHandler> handler) {
                input_mapping_t converted_inputs;

                for (const auto& single_input : inputs)
                {
                    if (std::holds_alternative<std::string>(single_input))
                    {
                        converted_inputs.push_back({.external_name = std::get<std::string>(single_input)});
                    }
                    else
                    {
                        auto pair = std::get<std::pair<std::string, std::string>>(single_input);

                        converted_inputs.push_back({.external_name = pair.first, .internal_name = pair.second});
                    }
                }

                return self.add_task_handler(std::move(converted_inputs), std::move(handler));
            },
            py::arg("inputs"),
            py::arg("handler"))
        .def("run", &LLMEngine::run, py::arg("message"));

    py::class_<PyLLMLambdaNode, LLMNodeBase, std::shared_ptr<PyLLMLambdaNode>>(_module, "LLMLambdaNode")
        .def(py::init<>([](py::function fn) {
                 return std::make_shared<PyLLMLambdaNode>(std::move(fn));
             }),
             py::arg("fn"))
        .def("get_input_names", &PyLLMLambdaNode::get_input_names)
        .def("execute", &PyLLMLambdaNode::execute, py::arg("context"));

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus::llm
