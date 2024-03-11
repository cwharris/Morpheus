/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doca_context.hpp"

#include "common.hpp"
#include "error.hpp"

#include "morpheus/utilities/error.hpp"

#include <cuda_runtime.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_mmap.h>
#include <doca_types.h>
#include <doca_version.h>
#include <glog/logging.h>
#include <rte_eal.h>
#include <rte_ethdev.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_PORT_STR_LEN 128    /* Maximal length of port name */

using jobs_check_t = doca_error_t (*)(doca_devinfo*);

static doca_error_t open_doca_device_with_pci(const char *pcie_value, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	doca_error_t res;
	size_t i;
	uint8_t is_addr_equal = 0;

	/* Set default return value */
	*retval = NULL;

	res = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		MORPHEUS_FAIL("Failed to load doca devices list");
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
		if (res == DOCA_SUCCESS && is_addr_equal) {
			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return res;
			}
		}
	}

	MORPHEUS_FAIL("Matching device not found");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return res;
}

doca_flow_port* init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
    std::array<char, MAX_PORT_STR_LEN> port_id_str;
    doca_flow_port_cfg port_cfg = {0};
    doca_flow_port* df_port;
    doca_flow_cfg rxq_flow_cfg = {0};
    rte_eth_dev_info dev_info  = {nullptr};
    rte_eth_conf eth_conf      = {
             .rxmode =
                 {
                     .mtu = 2048, /* Not really used, just to initialize DPDK */
            },
             .txmode =
                 {
                     .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
            },
    };
    rte_mempool* mp = nullptr;
    rte_eth_txconf tx_conf;
    rte_flow_error error;

    /*
     * DPDK should be initialized and started before DOCA Flow.
     * DPDK doesn't start the device without, at least, one DPDK Rx queue.
     * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
     *
     * Following lines of code can be considered the minimum WAR for this issue.
     */

    RTE_TRY(rte_eth_dev_info_get(port_id, &dev_info));
    RTE_TRY(rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf));

    mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, MAX_PKT_SIZE, rte_eth_dev_socket_id(port_id));

    if (mp == nullptr)
    {
        MORPHEUS_FAIL("rte_pktmbuf_pool_create failed.");
    }

    tx_conf = dev_info.default_txconf;
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;

    for (int idx = 0; idx < rxq_num; idx++)
    {
        RTE_TRY(rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), nullptr, mp));
        RTE_TRY(rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf));
    }

    RTE_TRY(rte_flow_isolate(port_id, 1, &error));
    RTE_TRY(rte_eth_dev_start(port_id));

    /* Initialize doca flow framework */
    rxq_flow_cfg.pipe_queues = rxq_num;
    /*
     * HWS: Hardware steering
     * Isolated: don't create RSS rule for DPDK created RX queues
     */
    rxq_flow_cfg.mode_args            = "vnf,hws,isolated";
    rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

    DOCA_TRY(doca_flow_init(&rxq_flow_cfg));

    /* Start doca flow port */
    port_cfg.port_id = port_id;
    port_cfg.type    = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str.begin(), MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str.cbegin();
    DOCA_TRY(doca_flow_port_start(&port_cfg, &df_port));

    return df_port;
}

}  // namespace

namespace morpheus::doca {

DocaContext::DocaContext(std::string nic_addr, std::string gpu_addr) : m_max_queue_count(1)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    m_rte_context = std::make_unique<RTEContext>();

    /* Register a logger backend for internal SDK errors and warnings */
    // DOCA_TRY(doca_log_backend_create_with_file_sdk(stderr, &sdk_log));
    // DOCA_TRY(doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_DEBUG));

    DOCA_TRY(open_doca_device_with_pci(nic_addr_c, &m_dev));
    DOCA_TRY(doca_dpdk_port_probe(m_dev, "dv_flow_en=2"));
    DOCA_TRY(doca_dpdk_get_first_port_id(m_dev, &m_nic_port));
    DOCA_TRY(doca_gpu_create(gpu_addr_c, &m_gpu));

    m_flow_port = init_doca_flow(m_nic_port, m_max_queue_count);

    if (m_flow_port == nullptr)
    {
        throw std::runtime_error("DOCA Flow initialization failed");
    }

    delete[] nic_addr_c;
    delete[] gpu_addr_c;
}

DocaContext::~DocaContext()
{
    doca_flow_port_stop(m_flow_port);
	doca_flow_destroy();

    if (m_gpu != nullptr)
    {
        auto doca_ret = doca_gpu_destroy(m_gpu);
        if (doca_ret != DOCA_SUCCESS)
            LOG(WARNING) << "DOCA cleanup failed (" << doca_ret << ")" << std::endl;
    }
}

doca_gpu* DocaContext::gpu()
{
    return m_gpu;
}

doca_dev* DocaContext::dev()
{
    return m_dev;
}

uint16_t DocaContext::nic_port()
{
    return m_nic_port;
}

doca_flow_port* DocaContext::flow_port()
{
    return m_flow_port;
}

}  // namespace morpheus::doca
