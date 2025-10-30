#pragma once
#include "utils.h"
#include "kernel.h"
#include <cstdio>
#include <cstring>

#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH) 
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>

#include "hostdevcommon/profiler_common.h"
#include "tools/mem_bench/host_utils.hpp"
#include <tt-metalium/distributed.hpp>
#include "impl/context/metal_context.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace NBodyProject; 
using namespace tt;
using namespace tt::tt_metal;

#endif

#if defined(_WH_MESH) 
using namespace tt::tt_metal::distributed;
#endif

#define BLKSIZE 32

#if defined(_WH_MULTIHOST)
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y,
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y,
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y,
                      double* __restrict h_adot_z, Sys& sys) {

    auto all_device_ids = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    auto device_map = tt::tt_metal::detail::CreateDevices(
        std::vector<chip_id_t>{*all_device_ids.begin()}
    );

    auto device = device_map.begin()->second;

    Program program = CreateProgram();

    CommandQueue& cq = device->command_queue();

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::vector<uint32_t> tiles_per_core;
    const uint32_t core_to_print = 4;
    const uint32_t local_N = sys.N / sys.size;
    const uint32_t n_tiles = local_N / tile_size;
    const uint32_t n_jtiles = sys.N;

    auto px_j = MakeBufferFP32(device, n_jtiles, false);
    auto px_i = MakeBufferFP32(device, n_tiles, false);
    auto py_j = MakeBufferFP32(device, n_jtiles, false);
    auto py_i = MakeBufferFP32(device, n_tiles, false);
    auto pz_j = MakeBufferFP32(device, n_jtiles, false);
    auto pz_i = MakeBufferFP32(device, n_tiles, false);

    auto vx_j = MakeBufferFP32(device, n_jtiles, false);
    auto vx_i = MakeBufferFP32(device, n_tiles, false);
    auto vy_j = MakeBufferFP32(device, n_jtiles, false);
    auto vy_i = MakeBufferFP32(device, n_tiles, false);
    auto vz_j = MakeBufferFP32(device, n_jtiles, false);
    auto vz_i = MakeBufferFP32(device, n_tiles, false);
    auto pwj = MakeBufferFP32(device, n_jtiles, false);

    auto ax = MakeBufferFP32(device, n_tiles, false);
    auto ay = MakeBufferFP32(device, n_tiles, false);
    auto az = MakeBufferFP32(device, n_tiles, false);

    auto adx = MakeBufferFP32(device, n_tiles, false);
    auto ady = MakeBufferFP32(device, n_tiles, false);
    auto adz = MakeBufferFP32(device, n_tiles, false);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    const uint32_t cir_buf_num_tile = 4;
    CBHandle cb_px_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_0, cir_buf_num_tile);
    CBHandle cb_px_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_1, cir_buf_num_tile);
    CBHandle cb_py_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_2, cir_buf_num_tile);
    CBHandle cb_py_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_3, cir_buf_num_tile);
    CBHandle cb_pz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_4, cir_buf_num_tile);
    CBHandle cb_pz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_5, cir_buf_num_tile);

    CBHandle cb_vx_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_6, cir_buf_num_tile);
    CBHandle cb_vx_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_7, cir_buf_num_tile);
    CBHandle cb_vy_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_8, cir_buf_num_tile);
    CBHandle cb_vy_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_9, cir_buf_num_tile);
    CBHandle cb_vz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_10, cir_buf_num_tile);
    CBHandle cb_vz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_11, cir_buf_num_tile);
    CBHandle cb_pwj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_12, cir_buf_num_tile);

    CBHandle cb_ax = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_13, cir_buf_num_tile);
    CBHandle cb_ay = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_14, cir_buf_num_tile);
    CBHandle cb_az = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_15, cir_buf_num_tile);

    CBHandle cb_adx = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_16, cir_buf_num_tile);
    CBHandle cb_ady = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_17, cir_buf_num_tile);
    CBHandle cb_adz = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_18, cir_buf_num_tile);

    CBHandle cb_axt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_19, cir_buf_num_tile);
    CBHandle cb_ayt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_20, cir_buf_num_tile);
    CBHandle cb_azt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_21, cir_buf_num_tile);

    CBHandle cb_adxt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_24, cir_buf_num_tile);
    CBHandle cb_adyt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_25, cir_buf_num_tile);
    CBHandle cb_adzt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_26, cir_buf_num_tile);

    CBHandle cb_tmp = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_22, cir_buf_num_tile);
    CBHandle cb_tj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_23, cir_buf_num_tile);

    CBHandle cb_tmp0 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_27, cir_buf_num_tile);
    CBHandle cb_tmp1 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_28, cir_buf_num_tile);
    CBHandle cb_tmp2 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_29, cir_buf_num_tile);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
        (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
        (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
        (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
        (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
        (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
        (std::uint32_t)tt::CBIndex::c_12
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_13,
        (std::uint32_t)tt::CBIndex::c_14,
        (std::uint32_t)tt::CBIndex::c_15,
        (std::uint32_t)tt::CBIndex::c_16,
        (std::uint32_t)tt::CBIndex::c_17,
        (std::uint32_t)tt::CBIndex::c_18
    };

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
        (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
        (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
        (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
        (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
        (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
        (std::uint32_t)tt::CBIndex::c_12,
        (std::uint32_t)tt::CBIndex::c_13,
        (std::uint32_t)tt::CBIndex::c_14,
        (std::uint32_t)tt::CBIndex::c_15,
        (std::uint32_t)tt::CBIndex::c_16,
        (std::uint32_t)tt::CBIndex::c_17,
        (std::uint32_t)tt::CBIndex::c_18
    };

    auto compute_config = ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
            .unpack_to_dest_mode = [] {
            std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
            modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_1)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_3)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_4)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_5)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_6)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_7)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_8)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_9)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_10)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_11)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_12)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_13)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_14)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_15)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_16)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_17)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_18)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_19)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_20)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_21)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_22)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_23)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_24)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_25)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_26)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_27)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_28)] = UnpackToDestMode::UnpackToDestFp32;
            modes[static_cast<uint32_t>(tt::CBIndex::c_29)] = UnpackToDestMode::UnpackToDestFp32;
            return modes;
        }(),
        .math_approx_mode = false,
        .compile_args = compute_compile_time_args,
        .defines = {},
        .opt_level = KernelBuildOptLevel::O3
    };

    auto reader = CreateKernel(
        program,
        "N-Body-Code/kernels/"
        "nb_read.cpp",
        all_device_cores,
        DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args});
    auto writer = CreateKernel(
        program,
        "N-Body-Code/kernels/"
        "nb_write.cpp",
        all_device_cores,
        DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args});
    auto compute = CreateKernel(
        program,
        "N-Body-Code/kernels/"
        "nb_compute.cpp",
        all_device_cores,
	compute_config
		);

    constexpr bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles, row_major);

    sys.num_cores = (int) num_cores;  
    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            SetRuntimeArgs(program, reader, core, std::array<uint32_t, 10>{0});
            SetRuntimeArgs(program, writer, core, std::array<uint32_t, 11>{0});
            SetRuntimeArgs(program, compute, core, std::array<uint32_t, 3>{0});
            continue;
        }
        if (i < core_to_print) {
            tiles_per_core.push_back(num_tiles_per_core);
        }
        SetRuntimeArgs(program, reader, core,
            {num_tiles_per_core, start_tile_id, n_jtiles,
            px_i->address(), px_j->address(), 
            py_i->address(), py_j->address(),
            pz_i->address(), pz_j->address(),
            vx_i->address(), vx_j->address(),
            vy_i->address(), vy_j->address(),
            vz_i->address(), vz_j->address(),
            pwj->address()
        });
        SetRuntimeArgs(program, writer, core, {
            num_tiles_per_core, start_tile_id,
            ax->address(),
            ay->address(),
            az->address(),
            adx->address(),
            ady->address(),
            adz->address()
            });
        SetRuntimeArgs(program, compute, core, {num_tiles_per_core, start_tile_id, n_jtiles,});
        start_tile_id += num_tiles_per_core;
    }

    std::vector<float> px_data(h_par_x + sys.start, h_par_x + sys.end);
    std::vector<float> py_data(h_par_y + sys.start, h_par_y + sys.end);
    std::vector<float> pz_data(h_par_z + sys.start, h_par_z + sys.end);
    std::vector<float> vx_data(h_vel_x + sys.start, h_vel_x + sys.end);
    std::vector<float> vy_data(h_vel_y + sys.start, h_vel_y + sys.end);
    std::vector<float> vz_data(h_vel_z + sys.start, h_vel_z + sys.end);

    std::vector<float> px_j_data, py_j_data, pz_j_data;
    std::vector<float> vx_j_data, vy_j_data, vz_j_data;
    std::vector<float> pw_j_data;
    px_j_data.resize(n_jtiles * tile_size);
    py_j_data.resize(n_jtiles * tile_size);
    pz_j_data.resize(n_jtiles * tile_size);
    vx_j_data.resize(n_jtiles * tile_size);
    vy_j_data.resize(n_jtiles * tile_size);
    vz_j_data.resize(n_jtiles * tile_size);
    pw_j_data.resize(n_jtiles * tile_size);

    #pragma omp parallel for
    for (int particle_idx = 0; particle_idx < sys.N; ++particle_idx) {
        for (int j = 0; j < tile_size; ++j) {
            int k = particle_idx * tile_size + j;
            px_j_data[k] = h_par_x[particle_idx];
            py_j_data[k] = h_par_y[particle_idx];
            pz_j_data[k] = h_par_z[particle_idx];
            vx_j_data[k] = h_vel_x[particle_idx];
            vy_j_data[k] = h_vel_y[particle_idx];
            vz_j_data[k] = h_vel_z[particle_idx];
            pw_j_data[k] = h_par_w[particle_idx];
        }
    }

    std::vector<float> ax_data;
    std::vector<float> ay_data;
    std::vector<float> az_data;
    std::vector<float> adx_data;
    std::vector<float> ady_data;
    std::vector<float> adz_data;

    ax_data.resize(px_data.size());
    ay_data.resize(px_data.size());
    az_data.resize(px_data.size());
    adx_data.resize(px_data.size());
    ady_data.resize(px_data.size());
    adz_data.resize(px_data.size());

    EnqueueWriteBuffer(cq, px_i, px_data, false);
    EnqueueWriteBuffer(cq, px_j, px_j_data, false);
    EnqueueWriteBuffer(cq, py_i, py_data, false);
    EnqueueWriteBuffer(cq, py_j, py_j_data, false);
    EnqueueWriteBuffer(cq, pz_i, pz_data, false);
    EnqueueWriteBuffer(cq, pz_j, pz_j_data, false);

    EnqueueWriteBuffer(cq, vx_i, vx_data, false);
    EnqueueWriteBuffer(cq, vx_j, vx_j_data, false);
    EnqueueWriteBuffer(cq, vy_i, vy_data, false);
    EnqueueWriteBuffer(cq, vy_j, vy_j_data, false);
    EnqueueWriteBuffer(cq, vz_i, vz_data, false);
    EnqueueWriteBuffer(cq, vz_j, vz_j_data, false);

    EnqueueWriteBuffer(cq, pwj, pw_j_data, false);

    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);
    // tt::tt_metal::Synchronize(device);
    auto end = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(cq, ax, ax_data, true);
    EnqueueReadBuffer(cq, ay, ay_data, true);
    EnqueueReadBuffer(cq, az, az_data, true);

    EnqueueReadBuffer(cq, adx, adx_data, true);
    EnqueueReadBuffer(cq, ady, ady_data, true);
    EnqueueReadBuffer(cq, adz, adz_data, true);

    // tt::tt_metal::DumpDeviceProfileResults(device, program);

    std::chrono::duration<double, std::milli> elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    double milliseconds = elapsed_ms.count();
    double seconds = milliseconds / 1000.0;

    sys.kernel_time = seconds;

    #pragma omp parallel for
    for (size_t i = 0; i < sys.N; ++i) {
        h_acc_x[i] = 0.0;
        h_acc_y[i] = 0.0;
        h_acc_z[i] = 0.0;
        h_adot_x[i] = 0.0;
        h_adot_y[i] = 0.0;
        h_adot_z[i] = 0.0;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < local_N; ++i) {
            h_acc_x[i + sys.start] = static_cast<double>(ax_data[i]);
            h_acc_y[i + sys.start] = static_cast<double>(ay_data[i]);
            h_acc_z[i + sys.start] = static_cast<double>(az_data[i]);
            h_adot_x[i + sys.start] = static_cast<double>(adx_data[i]);
            h_adot_y[i + sys.start] = static_cast<double>(ady_data[i]);
            h_adot_z[i + sys.start] = static_cast<double>(adz_data[i]);
    }

  tt::tt_metal::detail::CloseDevices(device_map);
}
#elif defined(_WH_MULTICHIP)
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y,
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y,
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y,
                      double* __restrict h_adot_z, Sys& sys) {

    auto all_device_ids = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    
    std::vector<chip_id_t> device_ids;
    int device_count = 0;
    for (auto id : all_device_ids) {
        device_ids.push_back(id);
        if (++device_count >= 2) break;
    }
    
    if (device_ids.size() < 2) {
        std::cerr << "Warning: Only " << device_ids.size() << " devices available, using " << device_ids.size() << " device(s)" << std::endl;
    }
    
    auto device_map = tt::tt_metal::detail::CreateDevices(device_ids);

    const uint32_t local_N = sys.N / sys.size;
    const uint32_t num_devices = device_ids.size();
    const uint32_t local_N_per_device = local_N / num_devices;
    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t n_tiles_per_device = local_N_per_device / tile_size;
    const uint32_t n_jtiles = sys.N;

    std::vector<std::vector<float>> all_ax_data(num_devices);
    std::vector<std::vector<float>> all_ay_data(num_devices);
    std::vector<std::vector<float>> all_az_data(num_devices);
    std::vector<std::vector<float>> all_adx_data(num_devices);
    std::vector<std::vector<float>> all_ady_data(num_devices);
    std::vector<std::vector<float>> all_adz_data(num_devices);

    double total_kernel_time = 0.0;

    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        auto device = device_map[device_ids[device_idx]];

        Program program = CreateProgram();
        CommandQueue& cq = device->command_queue();

        std::vector<uint32_t> tiles_per_core;
        const uint32_t core_to_print = 4;

        uint32_t device_start = sys.start + device_idx * local_N_per_device;
        uint32_t device_end = device_start + local_N_per_device;

        auto px_j = MakeBufferFP32(device, n_jtiles, false);
        auto px_i = MakeBufferFP32(device, n_tiles_per_device, false);
        auto py_j = MakeBufferFP32(device, n_jtiles, false);
        auto py_i = MakeBufferFP32(device, n_tiles_per_device, false);
        auto pz_j = MakeBufferFP32(device, n_jtiles, false);
        auto pz_i = MakeBufferFP32(device, n_tiles_per_device, false);

        auto vx_j = MakeBufferFP32(device, n_jtiles, false);
        auto vx_i = MakeBufferFP32(device, n_tiles_per_device, false);
        auto vy_j = MakeBufferFP32(device, n_jtiles, false);
        auto vy_i = MakeBufferFP32(device, n_tiles_per_device, false);
        auto vz_j = MakeBufferFP32(device, n_jtiles, false);
        auto vz_i = MakeBufferFP32(device, n_tiles_per_device, false);
        auto pwj = MakeBufferFP32(device, n_jtiles, false);

        auto ax = MakeBufferFP32(device, n_tiles_per_device, false);
        auto ay = MakeBufferFP32(device, n_tiles_per_device, false);
        auto az = MakeBufferFP32(device, n_tiles_per_device, false);

        auto adx = MakeBufferFP32(device, n_tiles_per_device, false);
        auto ady = MakeBufferFP32(device, n_tiles_per_device, false);
        auto adz = MakeBufferFP32(device, n_tiles_per_device, false);

        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x * num_cores_y;
        auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

        const uint32_t cir_buf_num_tile = 4;
        CBHandle cb_px_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_0, cir_buf_num_tile);
        CBHandle cb_px_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_1, cir_buf_num_tile);
        CBHandle cb_py_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_2, cir_buf_num_tile);
        CBHandle cb_py_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_3, cir_buf_num_tile);
        CBHandle cb_pz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_4, cir_buf_num_tile);
        CBHandle cb_pz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_5, cir_buf_num_tile);

        CBHandle cb_vx_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_6, cir_buf_num_tile);
        CBHandle cb_vx_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_7, cir_buf_num_tile);
        CBHandle cb_vy_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_8, cir_buf_num_tile);
        CBHandle cb_vy_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_9, cir_buf_num_tile);
        CBHandle cb_vz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_10, cir_buf_num_tile);
        CBHandle cb_vz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_11, cir_buf_num_tile);
        CBHandle cb_pwj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_12, cir_buf_num_tile);

        CBHandle cb_ax = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_13, cir_buf_num_tile);
        CBHandle cb_ay = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_14, cir_buf_num_tile);
        CBHandle cb_az = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_15, cir_buf_num_tile);

        CBHandle cb_adx = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_16, cir_buf_num_tile);
        CBHandle cb_ady = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_17, cir_buf_num_tile);
        CBHandle cb_adz = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_18, cir_buf_num_tile);

        CBHandle cb_axt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_19, cir_buf_num_tile);
        CBHandle cb_ayt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_20, cir_buf_num_tile);
        CBHandle cb_azt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_21, cir_buf_num_tile);

        CBHandle cb_adxt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_24, cir_buf_num_tile);
        CBHandle cb_adyt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_25, cir_buf_num_tile);
        CBHandle cb_adzt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_26, cir_buf_num_tile);

        CBHandle cb_tmp = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_22, cir_buf_num_tile);
        CBHandle cb_tj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_23, cir_buf_num_tile);

        CBHandle cb_tmp0 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_27, cir_buf_num_tile);
        CBHandle cb_tmp1 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_28, cir_buf_num_tile);
        CBHandle cb_tmp2 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_29, cir_buf_num_tile);

        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
            (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
            (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
            (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
            (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
            (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
            (std::uint32_t)tt::CBIndex::c_12
        };

        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t)tt::CBIndex::c_13,
            (std::uint32_t)tt::CBIndex::c_14,
            (std::uint32_t)tt::CBIndex::c_15,
            (std::uint32_t)tt::CBIndex::c_16,
            (std::uint32_t)tt::CBIndex::c_17,
            (std::uint32_t)tt::CBIndex::c_18
        };

        std::vector<uint32_t> compute_compile_time_args = {
            (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
            (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
            (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
            (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
            (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
            (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
            (std::uint32_t)tt::CBIndex::c_12,
            (std::uint32_t)tt::CBIndex::c_13,
            (std::uint32_t)tt::CBIndex::c_14,
            (std::uint32_t)tt::CBIndex::c_15,
            (std::uint32_t)tt::CBIndex::c_16,
            (std::uint32_t)tt::CBIndex::c_17,
            (std::uint32_t)tt::CBIndex::c_18
        };

        auto compute_config = ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .unpack_to_dest_mode = [] {
                std::vector<UnpackToDestMode> modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
                modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_1)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_3)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_4)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_5)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_6)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_7)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_8)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_9)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_10)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_11)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_12)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_13)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_14)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_15)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_16)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_17)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_18)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_19)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_20)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_21)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_22)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_23)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_24)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_25)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_26)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_27)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_28)] = UnpackToDestMode::UnpackToDestFp32;
                modes[static_cast<uint32_t>(tt::CBIndex::c_29)] = UnpackToDestMode::UnpackToDestFp32;
                return modes;
            }(),
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
            .defines = {},
            .opt_level = KernelBuildOptLevel::O3
        };

        auto reader = CreateKernel(
            program,
            "N-Body-Code/kernels/nb_read.cpp",
            all_device_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});
        
        auto writer = CreateKernel(
            program,
            "N-Body-Code/kernels/nb_write.cpp", 
            all_device_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});
        
        auto compute = CreateKernel(
            program,
            "N-Body-Code/kernels/nb_compute.cpp",
            all_device_cores,
            compute_config);

        constexpr bool row_major = true;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles_per_device, row_major);

        sys.num_cores = (int) num_cores;  
        auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
        
        for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
            const auto& core = cores[i];
            uint32_t num_tiles_per_core;

            if (core_group_1.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                SetRuntimeArgs(program, reader, core, std::array<uint32_t, 10>{0});
                SetRuntimeArgs(program, writer, core, std::array<uint32_t, 11>{0});
                SetRuntimeArgs(program, compute, core, std::array<uint32_t, 3>{0});
                continue;
            }
            
            if (i < core_to_print) {
                tiles_per_core.push_back(num_tiles_per_core);
            }
            
            SetRuntimeArgs(program, reader, core,
                {num_tiles_per_core, start_tile_id, n_jtiles,
                px_i->address(), px_j->address(), 
                py_i->address(), py_j->address(),
                pz_i->address(), pz_j->address(),
                vx_i->address(), vx_j->address(),
                vy_i->address(), vy_j->address(),
                vz_i->address(), vz_j->address(),
                pwj->address()
            });
            
            SetRuntimeArgs(program, writer, core, {
                num_tiles_per_core, start_tile_id,
                ax->address(),
                ay->address(),
                az->address(),
                adx->address(),
                ady->address(),
                adz->address()
            });
            
            SetRuntimeArgs(program, compute, core, {num_tiles_per_core, start_tile_id, n_jtiles,});
            start_tile_id += num_tiles_per_core;
        }

        std::vector<float> px_data(h_par_x + device_start, h_par_x + device_end);
        std::vector<float> py_data(h_par_y + device_start, h_par_y + device_end);
        std::vector<float> pz_data(h_par_z + device_start, h_par_z + device_end);
        std::vector<float> vx_data(h_vel_x + device_start, h_vel_x + device_end);
        std::vector<float> vy_data(h_vel_y + device_start, h_vel_y + device_end);
        std::vector<float> vz_data(h_vel_z + device_start, h_vel_z + device_end);

        std::vector<float> px_j_data, py_j_data, pz_j_data;
        std::vector<float> vx_j_data, vy_j_data, vz_j_data;
        std::vector<float> pw_j_data;
        px_j_data.resize(n_jtiles * tile_size);
        py_j_data.resize(n_jtiles * tile_size);
        pz_j_data.resize(n_jtiles * tile_size);
        vx_j_data.resize(n_jtiles * tile_size);
        vy_j_data.resize(n_jtiles * tile_size);
        vz_j_data.resize(n_jtiles * tile_size);
        pw_j_data.resize(n_jtiles * tile_size);

        #pragma omp parallel for
        for (int particle_idx = 0; particle_idx < sys.N; ++particle_idx) {
            for (int j = 0; j < tile_size; ++j) {
                int k = particle_idx * tile_size + j;
                px_j_data[k] = h_par_x[particle_idx];
                py_j_data[k] = h_par_y[particle_idx];
                pz_j_data[k] = h_par_z[particle_idx];
                vx_j_data[k] = h_vel_x[particle_idx];
                vy_j_data[k] = h_vel_y[particle_idx];
                vz_j_data[k] = h_vel_z[particle_idx];
                pw_j_data[k] = h_par_w[particle_idx];
            }
        }

        all_ax_data[device_idx].resize(px_data.size());
        all_ay_data[device_idx].resize(px_data.size());
        all_az_data[device_idx].resize(px_data.size());
        all_adx_data[device_idx].resize(px_data.size());
        all_ady_data[device_idx].resize(px_data.size());
        all_adz_data[device_idx].resize(px_data.size());

        EnqueueWriteBuffer(cq, px_i, px_data, false);
        EnqueueWriteBuffer(cq, px_j, px_j_data, false);
        EnqueueWriteBuffer(cq, py_i, py_data, false);
        EnqueueWriteBuffer(cq, py_j, py_j_data, false);
        EnqueueWriteBuffer(cq, pz_i, pz_data, false);
        EnqueueWriteBuffer(cq, pz_j, pz_j_data, false);

        EnqueueWriteBuffer(cq, vx_i, vx_data, false);
        EnqueueWriteBuffer(cq, vx_j, vx_j_data, false);
        EnqueueWriteBuffer(cq, vy_i, vy_data, false);
        EnqueueWriteBuffer(cq, vy_j, vy_j_data, false);
        EnqueueWriteBuffer(cq, vz_i, vz_data, false);
        EnqueueWriteBuffer(cq, vz_j, vz_j_data, false);

        EnqueueWriteBuffer(cq, pwj, pw_j_data, false);

        auto start = std::chrono::high_resolution_clock::now();
        EnqueueProgram(cq, program, true);
        auto end = std::chrono::high_resolution_clock::now();

        EnqueueReadBuffer(cq, ax, all_ax_data[device_idx], true);
        EnqueueReadBuffer(cq, ay, all_ay_data[device_idx], true);
        EnqueueReadBuffer(cq, az, all_az_data[device_idx], true);
        EnqueueReadBuffer(cq, adx, all_adx_data[device_idx], true);
        EnqueueReadBuffer(cq, ady, all_ady_data[device_idx], true);
        EnqueueReadBuffer(cq, adz, all_adz_data[device_idx], true);

        std::chrono::duration<double, std::milli> elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        total_kernel_time += elapsed_ms.count() / 1000.0;
    }

    sys.kernel_time = total_kernel_time;

    #pragma omp parallel for
    for (size_t i = 0; i < sys.N; ++i) {
        h_acc_x[i] = 0.0;
        h_acc_y[i] = 0.0;
        h_acc_z[i] = 0.0;
        h_adot_x[i] = 0.0;
        h_adot_y[i] = 0.0;
        h_adot_z[i] = 0.0;
    }

    #pragma omp parallel for
    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        uint32_t device_start = sys.start + device_idx * local_N_per_device;
        
        for (size_t i = 0; i < local_N_per_device; ++i) {
            h_acc_x[device_start + i] = static_cast<double>(all_ax_data[device_idx][i]);
            h_acc_y[device_start + i] = static_cast<double>(all_ay_data[device_idx][i]);
            h_acc_z[device_start + i] = static_cast<double>(all_az_data[device_idx][i]);
            h_adot_x[device_start + i] = static_cast<double>(all_adx_data[device_idx][i]);
            h_adot_y[device_start + i] = static_cast<double>(all_ady_data[device_idx][i]);
            h_adot_z[device_start + i] = static_cast<double>(all_adz_data[device_idx][i]);
        }
    }

    tt::tt_metal::detail::CloseDevices(device_map);
}

#elif defined(_WH_MESH)
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y,
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y,
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y,
                      double* __restrict h_adot_z, Sys& sys) {

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));

    constexpr uint32_t TILE_HEIGHT = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_WIDTH = tt::constants::TILE_WIDTH;
    const uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
    
    const uint32_t local_N_per_rank = sys.N / sys.size;
    const uint32_t total_num_tiles = local_N_per_rank / tile_size;
    const uint32_t tiles_per_device = total_num_tiles / mesh_device->num_devices();
    const uint32_t n_jtiles = sys.N;
    auto tile_size_bytes = tile_size * sizeof(float);
    
    std::shared_ptr<MeshBuffer> px_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> px_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> py_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> py_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> pz_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> pz_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);

    std::shared_ptr<MeshBuffer> vx_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> vx_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> vy_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> vy_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> vz_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
    std::shared_ptr<MeshBuffer> vz_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> pwj = MakeReplicatedBuffer(mesh_device, n_jtiles);

    std::shared_ptr<MeshBuffer> ax = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> ay = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> az = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);

    std::shared_ptr<MeshBuffer> adx = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> ady = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    std::shared_ptr<MeshBuffer> adz = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);

    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto all_device_cores = tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1})});
    sys.num_cores = core_grid.x * core_grid.y;

    const uint32_t cir_buf_num_tile = 8;
    CBHandle cb_px_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_0, cir_buf_num_tile);
    CBHandle cb_px_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_1, cir_buf_num_tile);
    CBHandle cb_py_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_2, cir_buf_num_tile);
    CBHandle cb_py_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_3, cir_buf_num_tile);
    CBHandle cb_pz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_4, cir_buf_num_tile);
    CBHandle cb_pz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_5, cir_buf_num_tile);

    CBHandle cb_vx_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_6, cir_buf_num_tile);
    CBHandle cb_vx_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_7, cir_buf_num_tile);
    CBHandle cb_vy_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_8, cir_buf_num_tile);
    CBHandle cb_vy_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_9, cir_buf_num_tile);
    CBHandle cb_vz_i = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_10, cir_buf_num_tile);
    CBHandle cb_vz_j = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_11, cir_buf_num_tile);
    CBHandle cb_pwj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_12, cir_buf_num_tile);

    CBHandle cb_ax = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_13, cir_buf_num_tile);
    CBHandle cb_ay = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_14, cir_buf_num_tile);
    CBHandle cb_az = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_15, cir_buf_num_tile);
    CBHandle cb_adx = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_16, cir_buf_num_tile);
    CBHandle cb_ady = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_17, cir_buf_num_tile);
    CBHandle cb_adz = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_18, cir_buf_num_tile);

    CBHandle cb_axt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_19, cir_buf_num_tile);
    CBHandle cb_ayt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_20, cir_buf_num_tile);
    CBHandle cb_azt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_21, cir_buf_num_tile);
    CBHandle cb_adxt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_24, cir_buf_num_tile);
    CBHandle cb_adyt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_25, cir_buf_num_tile);
    CBHandle cb_adzt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_26, cir_buf_num_tile);

    CBHandle cb_tmp = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_22, cir_buf_num_tile);
    CBHandle cb_tj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_23, cir_buf_num_tile);

    CBHandle cb_tmp0 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_27, cir_buf_num_tile);
    CBHandle cb_tmp1 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_28, cir_buf_num_tile);
    CBHandle cb_tmp2 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_29, cir_buf_num_tile);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(px_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(px_j->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(py_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(py_j->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pz_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pz_j->get_reference_buffer()).append_to(reader_compile_time_args);

    TensorAccessorArgs(vx_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(vx_j->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(vy_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(vy_j->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(vz_i->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(vz_j->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pwj->get_reference_buffer()).append_to(reader_compile_time_args);

    auto reader = CreateKernel(
      program,
      "N-Body-Code/kernels/"
      "mesh_read.cpp",
      all_device_cores,
      DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args
      });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(ax->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(ay->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(az->get_reference_buffer()).append_to(writer_compile_time_args);

    TensorAccessorArgs(adx->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(ady->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(adz->get_reference_buffer()).append_to(writer_compile_time_args);

    KernelHandle writer = CreateKernel(
      program,
      "N-Body-Code/kernels/"
      "mesh_write.cpp",
      all_device_cores,
      DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args
      });

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_1)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_3)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_4)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_5)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_6)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_7)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_8)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_9)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_10)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_11)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_12)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_13)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_14)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_15)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_16)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_17)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_18)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_19)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_20)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_21)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_22)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_23)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_24)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_25)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_26)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_27)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_28)] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_29)] = UnpackToDestMode::UnpackToDestFp32;

    auto compute_config = ComputeConfig{
      .math_fidelity = MathFidelity::HiFi4,
      .fp32_dest_acc_en = true,
      .dst_full_sync_en = true,
      .unpack_to_dest_mode = unpack_modes,
      .math_approx_mode = false,
      .compile_args = {},// compute_compile_time_args,
      // .defines = {},
      .opt_level = KernelBuildOptLevel::O3
    };
    auto compute = CreateKernel(
        program,
        "N-Body-Code/kernels/"
        "mesh_compute.cpp",
        all_device_cores,
	      compute_config
		  );

    SetRuntimeArgs(program, reader, all_device_cores, {
        tiles_per_device, n_jtiles,
        static_cast<unsigned int>(px_i->address()),
        static_cast<unsigned int>(px_j->address()), 
        static_cast<unsigned int>(py_i->address()),
        static_cast<unsigned int>(py_j->address()),
        static_cast<unsigned int>(pz_i->address()),
        static_cast<unsigned int>(pz_j->address()),
        static_cast<unsigned int>(vx_i->address()),
        static_cast<unsigned int>(vx_j->address()),
        static_cast<unsigned int>(vy_i->address()),
        static_cast<unsigned int>(vy_j->address()),
        static_cast<unsigned int>(vz_i->address()),
        static_cast<unsigned int>(vz_j->address()),
        static_cast<unsigned int>(pwj->address())
    });

    SetRuntimeArgs(program, writer, all_device_cores, {
        tiles_per_device,
        static_cast<unsigned int>(ax->address()),
        static_cast<unsigned int>(ay->address()),
        static_cast<unsigned int>(az->address()),
        static_cast<unsigned int>(adx->address()),
        static_cast<unsigned int>(ady->address()),
        static_cast<unsigned int>(adz->address())
    });

    SetRuntimeArgs(program, compute, all_device_cores, {
        tiles_per_device, n_jtiles
        });

    std::vector<float> px_data(h_par_x + sys.start, h_par_x + sys.end);
    std::vector<float> py_data(h_par_y + sys.start, h_par_y + sys.end);
    std::vector<float> pz_data(h_par_z + sys.start, h_par_z + sys.end);
    std::vector<float> vx_data(h_vel_x + sys.start, h_vel_x + sys.end);
    std::vector<float> vy_data(h_vel_y + sys.start, h_vel_y + sys.end);
    std::vector<float> vz_data(h_vel_z + sys.start, h_vel_z + sys.end);

    std::vector<float> px_j_data, py_j_data, pz_j_data;
    std::vector<float> vx_j_data, vy_j_data, vz_j_data;
    std::vector<float> pw_j_data;
    px_j_data.resize(sys.N * tile_size);
    py_j_data.resize(sys.N * tile_size);
    pz_j_data.resize(sys.N * tile_size);
    vx_j_data.resize(sys.N * tile_size);
    vy_j_data.resize(sys.N * tile_size);
    vz_j_data.resize(sys.N * tile_size);
    pw_j_data.resize(sys.N * tile_size);

    #pragma omp parallel for
    for (int particle_idx = 0; particle_idx < sys.N; ++particle_idx) {
        for (int j = 0; j < tile_size; ++j) {
            int k = particle_idx * tile_size + j;
            px_j_data[k] = h_par_x[particle_idx];
            py_j_data[k] = h_par_y[particle_idx];
            pz_j_data[k] = h_par_z[particle_idx];
            vx_j_data[k] = h_vel_x[particle_idx];
            vy_j_data[k] = h_vel_y[particle_idx];
            vz_j_data[k] = h_vel_z[particle_idx];
            pw_j_data[k] = h_par_w[particle_idx];
        }
    }

    EnqueueWriteMeshBuffer(cq, px_i, px_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, px_j, px_j_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, py_i, py_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, py_j, py_j_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, pz_i, pz_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, pz_j, pz_j_data, false /* blocking */);

    EnqueueWriteMeshBuffer(cq, vx_i, vx_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, vx_j, vx_j_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, vy_i, vy_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, vy_j, vy_j_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, vz_i, vz_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, vz_j, vz_j_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, pwj, pw_j_data, false /* blocking */);

    auto start = std::chrono::high_resolution_clock::now();
    // auto start_event = cq.enqueue_record_event_to_host();

    auto mesh_workload = CreateMeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(program), device_range);
    EnqueueMeshWorkload(cq, mesh_workload, true /* blocking */);
    // EnqueueMeshWorkload(cq, mesh_workload, true /* blocking */);

    // auto end_event = cq.enqueue_record_event_to_host();
    // cq.enqueue_wait_for_event(end_event);
    // distributed::EventSynchronize(end_event);

    auto end = std::chrono::high_resolution_clock::now();

    std::vector<float> ax_data;
    std::vector<float> ay_data;
    std::vector<float> az_data;
    std::vector<float> adx_data;
    std::vector<float> ady_data;
    std::vector<float> adz_data;
    ax_data.resize(px_data.size());
    ay_data.resize(px_data.size());
    az_data.resize(px_data.size());
    adx_data.resize(px_data.size());
    ady_data.resize(px_data.size());
    adz_data.resize(px_data.size());

    EnqueueReadMeshBuffer(cq, ax_data, ax, true /* blocking */);
    EnqueueReadMeshBuffer(cq, ay_data, ay, true /* blocking */);
    EnqueueReadMeshBuffer(cq, az_data, az, true /* blocking */);

    EnqueueReadMeshBuffer(cq, adx_data, adx, true /* blocking */);
    EnqueueReadMeshBuffer(cq, ady_data, ady, true /* blocking */);
    EnqueueReadMeshBuffer(cq, adz_data, adz, true /* blocking */);

    // tt::tt_metal::DumpDeviceProfileResults(device, program);

    std::chrono::duration<double, std::milli> elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    double milliseconds = elapsed_ms.count();
    double seconds = milliseconds / 1000.0;

    sys.kernel_time = seconds;

    #pragma omp parallel for
    for (size_t i = 0; i < sys.N; ++i) {
        h_acc_x[i] = 0.0;
        h_acc_y[i] = 0.0;
        h_acc_z[i] = 0.0;
        h_adot_x[i] = 0.0;
        h_adot_y[i] = 0.0;
        h_adot_z[i] = 0.0;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < local_N_per_rank; ++i) {
      h_acc_x[i + sys.start] = static_cast<double>(ax_data[i]);
      h_acc_y[i + sys.start] = static_cast<double>(ay_data[i]);
      h_acc_z[i + sys.start] = static_cast<double>(az_data[i]);
      h_adot_x[i + sys.start] = static_cast<double>(adx_data[i]);
      h_adot_y[i + sys.start] = static_cast<double>(ady_data[i]);
      h_adot_z[i + sys.start] = static_cast<double>(adz_data[i]);
    }

  mesh_device->close();
}
#elif defined(_F32)
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z, 
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y, 
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y, 
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y, 
                      double* __restrict h_adot_z, Sys &sys) {
  int lN = sys.N;
  int lrank = sys.rank;
  int lsize = sys.size;
  
  #pragma omp parallel for
  for (int i = 0; i < lN; i++) {
    h_acc_x[i] = 0.0;  h_acc_y[i] = 0.0;  h_acc_z[i] = 0.0;
    h_adot_x[i] = 0.0; h_adot_y[i] = 0.0; h_adot_z[i] = 0.0;
  }

#pragma omp parallel reduction(+:h_acc_x[:lN], h_acc_y[:lN], h_acc_z[:lN], h_adot_x[:lN], h_adot_y[:lN], h_adot_z[:lN])
  {
  
  int nthreads = omp_get_num_threads();
  int thid = omp_get_thread_num();

  for (int i = lrank * nthreads + thid; i < lN; i += lsize * nthreads) {
    if (i >= lN) break; 

      for (int j = 0; j < i; j++) {
        float dx = h_par_x[j] - h_par_x[i];
        float dy = h_par_y[j] - h_par_y[i];
        float dz = h_par_z[j] - h_par_z[i];

        float vx = h_vel_x[j] - h_vel_x[i];
        float vy = h_vel_y[j] - h_vel_y[i];
        float vz = h_vel_z[j] - h_vel_z[i];

        float s = dx * dx + dy * dy + dz * dz + EPS2;
        float vr = dx * vx + dy * vy + dz * vz;

        float invss = InvSqrt2D(s);
        float invss2 = invss * invss;
        float q = invss2 * invss;
        float qdotq = -3.0 * vr * invss2;

        float ti = h_par_w[i] * q;
        float tj = h_par_w[j] * q;

        h_acc_x[i] += dx * tj;
        h_acc_y[i] += dy * tj;
        h_acc_z[i] += dz * tj;
        h_acc_x[j] -= dx * ti;
        h_acc_y[j] -= dy * ti;
        h_acc_z[j] -= dz * ti;
  
        h_adot_x[i] += tj * (vx + qdotq * dx);
        h_adot_y[i] += tj * (vy + qdotq * dy);
        h_adot_z[i] += tj * (vz + qdotq * dz);
  
        h_adot_x[j] -= ti * (vx + qdotq * dx);
        h_adot_y[j] -= ti * (vy + qdotq * dy);
        h_adot_z[j] -= ti * (vz + qdotq * dz);
      }
    }
  } // end of pragma
}
#else
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z, 
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y, 
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y, 
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y, 
                      double* __restrict h_adot_z, Sys& sys) {
  int lN = sys.N;
  int lrank = sys.rank;
  int lsize = sys.size;
  
  #pragma omp parallel for
  for (int i = 0; i < lN; i++) {
    h_acc_x[i] = 0.0;  h_acc_y[i] = 0.0;  h_acc_z[i] = 0.0;
    h_adot_x[i] = 0.0; h_adot_y[i] = 0.0; h_adot_z[i] = 0.0;
  }

  #pragma omp parallel reduction(+:h_acc_x[:lN], h_acc_y[:lN], h_acc_z[:lN], h_adot_x[:lN], h_adot_y[:lN], h_adot_z[:lN])
  {
  int nthreads = omp_get_num_threads();
  int thid = omp_get_thread_num();

  for (int i = lrank * nthreads + thid; i < lN; i += lsize * nthreads) {
    if (i >= lN) break; 
    for (int j = 0; j < i; j++) {
      double dx = h_par_x[j] - h_par_x[i];
      double dy = h_par_y[j] - h_par_y[i];
      double dz = h_par_z[j] - h_par_z[i];

      double vx = h_vel_x[j] - h_vel_x[i];
      double vy = h_vel_y[j] - h_vel_y[i];
      double vz = h_vel_z[j] - h_vel_z[i];

      double s = dx * dx + dy * dy + dz * dz + EPS2;
      double vr = dx * vx + dy * vy + dz * vz;

      double invss = InvSqrt2D(s);
      double invss2 = invss * invss;
      double q = invss * invss2;
      double qdotq = -3.0 * vr * invss2;

      double ti = h_par_w[i] * q;
      double tj = h_par_w[j] * q;

      h_acc_x[i] += dx * tj;
      h_acc_y[i] += dy * tj;
      h_acc_z[i] += dz * tj;
      h_acc_x[j] -= dx * ti;
      h_acc_y[j] -= dy * ti;
      h_acc_z[j] -= dz * ti;

      h_adot_x[i] += tj * (vx + qdotq * dx);
      h_adot_y[i] += tj * (vy + qdotq * dy);
      h_adot_z[i] += tj * (vz + qdotq * dz);

      h_adot_x[j] -= ti * (vx + qdotq * dx);
      h_adot_y[j] -= ti * (vy + qdotq * dy);
      h_adot_z[j] -= ti * (vz + qdotq * dz);
    }
  }
  } // end of pragma
}
#endif
