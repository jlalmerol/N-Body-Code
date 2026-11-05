#include "utils.h"
#include "kernel.h"
#include <cstdio>
#include <cstring>
#include <array>
#include <cmath>
#include <stdexcept>

#if defined(_GEMM_BASED)
#include <cblas.h>
#endif

#if defined(_WH_MULTIHOST) || defined(_WH_GEMM_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH)
#include "nbody_profiler.h"
#include <tracy/Tracy.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>

#if __has_include("hostdevcommon/profiler_common.h")
#include "hostdevcommon/profiler_common.h"
#else
#include "hostdev/profiler_common.h"
#endif
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
#include <map>
#include <random>
#include <string_view>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace NBodyProject; 
using namespace tt;
using namespace tt::tt_metal;

#endif

#if defined(_WH_MULTIHOST) || defined(_WH_GEMM_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH)
using namespace tt::tt_metal::distributed;

namespace {
// Timed builds fence phase boundaries; untimed builds rely on queue ordering
// until result readback. Host input vectors stay alive through that readback.
#ifdef NBODY_PHASE_TIMERS
constexpr bool block_for_phase_timers = true;
#else
constexpr bool block_for_phase_timers = false;
#endif
}  // namespace
#endif

#define BLKSIZE 32

#if defined(_WH_GEMM_MULTIHOST)
#include "wh_gemm_multihost.inc"
#elif defined(_WH_MULTIHOST) && 0  // Pre-Metalium-0.74 implementation retained for reference.
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
#elif defined(_WH_MULTICHIP) && 0  // Pre-Metalium-0.74 implementation retained for reference.
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
        EnqueueProgram(cq, program, block_for_phase_timers);
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

#elif defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH)
namespace {
struct WhSfpuPersistentContext {
    int global_n = 0;
    int local_n = 0;
    int rank_start = 0;
    uint32_t target_tiles = 0;
    uint32_t source_tiles = 0;
    uint32_t active_cores = 0;
    std::shared_ptr<MeshDevice> mesh_device;
    std::shared_ptr<MeshBuffer> px_j, px_i, py_j, py_i, pz_j, pz_i;
    std::shared_ptr<MeshBuffer> vx_j, vx_i, vy_j, vy_i, vz_j, vz_i, pwj;
    std::shared_ptr<MeshBuffer> ax, ay, az, adx, ady, adz;
    MeshWorkload workload;
    bool workload_ready = false;
};

#if defined(_WH_MULTIHOST)
std::unique_ptr<WhSfpuPersistentContext> wh_sfpu_context;
#endif
}  // namespace

void shutdown_force_calculation_backend() {
#if defined(_WH_MULTIHOST)
    if (wh_sfpu_context && wh_sfpu_context->mesh_device) {
        nbody_profiling::flush_device(*wh_sfpu_context->mesh_device);
        wh_sfpu_context->mesh_device->close();
    }
    wh_sfpu_context.reset();
#endif
}

void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y,
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y,
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y,
                      double* __restrict h_adot_z, Sys& sys) {
    ZoneScopedN("NBody force calculation");

    using PhaseClock = std::chrono::steady_clock;
    const auto phase_setup_begin = PhaseClock::now();

#if defined(_WH_MULTIHOST)
    if (!wh_sfpu_context) wh_sfpu_context = std::make_unique<WhSfpuPersistentContext>();
    auto& persistent = *wh_sfpu_context;
    if (!persistent.mesh_device) persistent.mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    auto mesh_device = persistent.mesh_device;
#else
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
    WhSfpuPersistentContext transient;
    auto& persistent = transient;
    persistent.mesh_device = mesh_device;
#endif

    constexpr uint32_t TILE_HEIGHT = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_WIDTH = tt::constants::TILE_WIDTH;
    const uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
    
    const uint32_t local_N_per_rank = static_cast<uint32_t>(sys.end - sys.start);
#if defined(_WH_MULTIHOST) || defined(_WH_MESH)
    // One tile is a 32-target x 32-source interaction block. Target values are
    // repeated across columns and source values across rows.
    const uint32_t total_num_tiles = (local_N_per_rank + TILE_HEIGHT - 1) / TILE_HEIGHT;
#else
    const uint32_t total_num_tiles = local_N_per_rank / tile_size;
#endif
    const uint32_t tiles_per_device = total_num_tiles / mesh_device->num_devices();
#if defined(_WH_MULTIHOST) || defined(_WH_MESH)
    const uint32_t n_jtiles = (static_cast<uint32_t>(sys.N) + TILE_WIDTH - 1) / TILE_WIDTH;
#else
    const uint32_t n_jtiles = sys.N;
#endif
    auto tile_size_bytes = tile_size * sizeof(float);
    if (total_num_tiles == 0 || total_num_tiles % mesh_device->num_devices() != 0) {
        throw std::runtime_error("SFPU MeshDevice backend requires at least one target tile per device and an even tile split");
    }
    if (!persistent.workload_ready) {
        persistent.global_n = sys.N; persistent.local_n = local_N_per_rank; persistent.rank_start = sys.start;
        persistent.target_tiles = total_num_tiles; persistent.source_tiles = n_jtiles;
        persistent.px_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.px_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.py_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.py_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.pz_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.pz_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.vx_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.vx_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.vy_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.vy_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.vz_j = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.vz_i = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.pwj = MakeReplicatedBuffer(mesh_device, n_jtiles);
        persistent.ax = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.ay = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.az = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.adx = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.ady = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
        persistent.adz = MakeMeshBuffer(mesh_device, tiles_per_device, total_num_tiles);
    } else if (persistent.global_n != sys.N || persistent.local_n != static_cast<int>(local_N_per_rank) ||
               persistent.rank_start != sys.start || persistent.target_tiles != total_num_tiles ||
               persistent.source_tiles != n_jtiles) {
        throw std::runtime_error("persistent SFPU context cannot change problem or MPI partition size");
    }
    auto& px_j=persistent.px_j; auto& px_i=persistent.px_i; auto& py_j=persistent.py_j; auto& py_i=persistent.py_i;
    auto& pz_j=persistent.pz_j; auto& pz_i=persistent.pz_i; auto& vx_j=persistent.vx_j; auto& vx_i=persistent.vx_i;
    auto& vy_j=persistent.vy_j; auto& vy_i=persistent.vy_i; auto& vz_j=persistent.vz_j; auto& vz_i=persistent.vz_i;
    auto& pwj=persistent.pwj; auto& ax=persistent.ax; auto& ay=persistent.ay; auto& az=persistent.az;
    auto& adx=persistent.adx; auto& ady=persistent.ady; auto& adz=persistent.adz;

    auto& cq = mesh_device->mesh_command_queue();
    if (!persistent.workload_ready) {
    Program program = CreateProgram();
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    constexpr bool row_major = true;
    auto [num_active_cores, active_cores, core_group_1, core_group_2,
          tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_grid, tiles_per_device, row_major);
    auto all_device_cores = active_cores;
    sys.num_cores = num_active_cores * mesh_device->num_devices();

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
#ifdef NBODY_DEVICE_ROW_REDUCE
    CBHandle cb_row_scale = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_30, 1);
#endif

    std::vector<uint32_t> reader_compile_time_args;
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
    reader_compile_time_args = {
        static_cast<uint32_t>(tt::CBIndex::c_0), static_cast<uint32_t>(tt::CBIndex::c_1),
        static_cast<uint32_t>(tt::CBIndex::c_2), static_cast<uint32_t>(tt::CBIndex::c_3),
        static_cast<uint32_t>(tt::CBIndex::c_4), static_cast<uint32_t>(tt::CBIndex::c_5),
        static_cast<uint32_t>(tt::CBIndex::c_6), static_cast<uint32_t>(tt::CBIndex::c_7),
        static_cast<uint32_t>(tt::CBIndex::c_8), static_cast<uint32_t>(tt::CBIndex::c_9),
        static_cast<uint32_t>(tt::CBIndex::c_10), static_cast<uint32_t>(tt::CBIndex::c_11),
        static_cast<uint32_t>(tt::CBIndex::c_12)};
#else
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
#endif

    auto reader = CreateKernel(
      program,
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
      "N-Body-Code/kernels/nb_read.cpp",
#else
      "N-Body-Code/kernels/"
      "mesh_read.cpp",
#endif
      all_device_cores,
      DataMovementConfig{
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
#else
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
#endif
        .compile_args = reader_compile_time_args
      });

    std::vector<uint32_t> writer_compile_time_args;
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
    writer_compile_time_args = {
        static_cast<uint32_t>(tt::CBIndex::c_13), static_cast<uint32_t>(tt::CBIndex::c_14),
        static_cast<uint32_t>(tt::CBIndex::c_15), static_cast<uint32_t>(tt::CBIndex::c_16),
        static_cast<uint32_t>(tt::CBIndex::c_17), static_cast<uint32_t>(tt::CBIndex::c_18)};
#else
    TensorAccessorArgs(ax->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(ay->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(az->get_reference_buffer()).append_to(writer_compile_time_args);

    TensorAccessorArgs(adx->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(ady->get_reference_buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(adz->get_reference_buffer()).append_to(writer_compile_time_args);
#endif

    KernelHandle writer = CreateKernel(
      program,
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
      "N-Body-Code/kernels/nb_write.cpp",
#else
      "N-Body-Code/kernels/"
      "mesh_write.cpp",
#endif
      all_device_cores,
      DataMovementConfig{
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
#else
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
#endif
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

    std::vector<uint32_t> compute_compile_time_args;
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
    for (uint32_t cb = static_cast<uint32_t>(tt::CBIndex::c_0);
         cb <= static_cast<uint32_t>(tt::CBIndex::c_18); ++cb) {
        compute_compile_time_args.push_back(cb);
    }
#endif

    std::map<std::string, std::string> compute_defines;
#ifdef NBODY_DEVICE_PROFILE
    compute_defines["NBODY_DEVICE_PROFILE"] = "1";
#endif
#ifdef NBODY_DEVICE_ROW_REDUCE
    compute_defines["NBODY_DEVICE_ROW_REDUCE"] = "1";
#endif
    auto compute_config = ComputeConfig{
      .math_fidelity = MathFidelity::HiFi4,
      .fp32_dest_acc_en = true,
      .dst_full_sync_en = true,
      .unpack_to_dest_mode = unpack_modes,
      .math_approx_mode = false,
      .compile_args = compute_compile_time_args,
      .defines = compute_defines,
      .opt_level = KernelBuildOptLevel::O3
    };
    auto compute = CreateKernel(
        program,
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
        "N-Body-Code/kernels/nb_compute.cpp",
#else
        "N-Body-Code/kernels/"
        "mesh_compute.cpp",
#endif
        all_device_cores,
	      compute_config
		  );

    const auto cores = grid_to_cores(
        num_active_cores, core_grid.x, core_grid.y, row_major);

    uint32_t first_tile = 0;
    for (const auto& core : cores) {
        uint32_t tile_count = 0;
        if (core_group_1.contains(core)) {
            tile_count = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            tile_count = tiles_per_core_group_2;
        }

        SetRuntimeArgs(program, reader, core, {
            tile_count, first_tile, n_jtiles,
            static_cast<uint32_t>(px_i->address()), static_cast<uint32_t>(px_j->address()),
            static_cast<uint32_t>(py_i->address()), static_cast<uint32_t>(py_j->address()),
            static_cast<uint32_t>(pz_i->address()), static_cast<uint32_t>(pz_j->address()),
            static_cast<uint32_t>(vx_i->address()), static_cast<uint32_t>(vx_j->address()),
            static_cast<uint32_t>(vy_i->address()), static_cast<uint32_t>(vy_j->address()),
            static_cast<uint32_t>(vz_i->address()), static_cast<uint32_t>(vz_j->address()),
            static_cast<uint32_t>(pwj->address())
        });

        SetRuntimeArgs(program, writer, core, {
            tile_count, first_tile,
            static_cast<uint32_t>(ax->address()), static_cast<uint32_t>(ay->address()),
            static_cast<uint32_t>(az->address()), static_cast<uint32_t>(adx->address()),
            static_cast<uint32_t>(ady->address()), static_cast<uint32_t>(adz->address())
        });

#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP)
        SetRuntimeArgs(program, compute, core, {tile_count, first_tile, n_jtiles});
#else
        SetRuntimeArgs(program, compute, core, {tile_count, n_jtiles});
#endif
        first_tile += tile_count;
    }

    auto device_range = MeshCoordinateRange(mesh_device->shape());
    persistent.workload.add_program(device_range, std::move(program));
    persistent.active_cores = sys.num_cores;
    persistent.workload_ready = true;
    }
    sys.num_cores = persistent.active_cores;
    const auto phase_pack_begin = PhaseClock::now();
    sys.phase_setup = std::chrono::duration<double>(phase_pack_begin - phase_setup_begin).count();

    std::vector<float> px_data, py_data, pz_data, vx_data, vy_data, vz_data;
    std::vector<float> px_j_data, py_j_data, pz_j_data, vx_j_data, vy_j_data, vz_j_data, pw_j_data;

#if defined(_WH_MULTIHOST) || defined(_WH_MESH)
    px_data.resize(total_num_tiles * tile_size); py_data.resize(total_num_tiles * tile_size);
    pz_data.resize(total_num_tiles * tile_size); vx_data.resize(total_num_tiles * tile_size);
    vy_data.resize(total_num_tiles * tile_size); vz_data.resize(total_num_tiles * tile_size);
    px_j_data.resize(n_jtiles * tile_size); py_j_data.resize(n_jtiles * tile_size);
    pz_j_data.resize(n_jtiles * tile_size); vx_j_data.resize(n_jtiles * tile_size);
    vy_j_data.resize(n_jtiles * tile_size); vz_j_data.resize(n_jtiles * tile_size);
    pw_j_data.resize(n_jtiles * tile_size);

    auto physical_index = [tile_size](uint32_t tile, uint32_t row, uint32_t column) {
        const uint32_t face = (row / 16) * 2 + column / 16;
        return tile * tile_size + face * 256 + (row % 16) * 16 + column % 16;
    };
    auto set_tile = [&](std::vector<float>& data, uint32_t tile, uint32_t row, uint32_t column, float value) {
        data[physical_index(tile, row, column)] = value;
    };

    #pragma omp parallel for
    for (uint32_t local_i = 0; local_i < local_N_per_rank; ++local_i) {
        const uint32_t particle = static_cast<uint32_t>(sys.start) + local_i;
        const uint32_t tile = local_i / TILE_HEIGHT, row = local_i % TILE_HEIGHT;
        for (uint32_t column = 0; column < TILE_WIDTH; ++column) {
            set_tile(px_data,tile,row,column,h_par_x[particle]); set_tile(py_data,tile,row,column,h_par_y[particle]);
            set_tile(pz_data,tile,row,column,h_par_z[particle]); set_tile(vx_data,tile,row,column,h_vel_x[particle]);
            set_tile(vy_data,tile,row,column,h_vel_y[particle]); set_tile(vz_data,tile,row,column,h_vel_z[particle]);
        }
    }

    #pragma omp parallel for
    for (uint32_t particle = 0; particle < static_cast<uint32_t>(sys.N); ++particle) {
        const uint32_t tile = particle / TILE_WIDTH, column = particle % TILE_WIDTH;
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row) {
            set_tile(px_j_data,tile,row,column,h_par_x[particle]); set_tile(py_j_data,tile,row,column,h_par_y[particle]);
            set_tile(pz_j_data,tile,row,column,h_par_z[particle]); set_tile(vx_j_data,tile,row,column,h_vel_x[particle]);
            set_tile(vy_j_data,tile,row,column,h_vel_y[particle]); set_tile(vz_j_data,tile,row,column,h_vel_z[particle]);
            set_tile(pw_j_data,tile,row,column,h_par_w[particle]);
        }
    }
#else
    px_data.assign(h_par_x + sys.start, h_par_x + sys.end);
    py_data.assign(h_par_y + sys.start, h_par_y + sys.end);
    pz_data.assign(h_par_z + sys.start, h_par_z + sys.end);
    vx_data.assign(h_vel_x + sys.start, h_vel_x + sys.end);
    vy_data.assign(h_vel_y + sys.start, h_vel_y + sys.end);
    vz_data.assign(h_vel_z + sys.start, h_vel_z + sys.end);
    px_j_data.resize(sys.N * tile_size); py_j_data.resize(sys.N * tile_size);
    pz_j_data.resize(sys.N * tile_size); vx_j_data.resize(sys.N * tile_size);
    vy_j_data.resize(sys.N * tile_size); vz_j_data.resize(sys.N * tile_size);
    pw_j_data.resize(sys.N * tile_size);
    #pragma omp parallel for
    for (int particle_idx = 0; particle_idx < sys.N; ++particle_idx) {
        for (uint32_t j = 0; j < tile_size; ++j) {
            const uint32_t k = particle_idx * tile_size + j;
            px_j_data[k]=h_par_x[particle_idx]; py_j_data[k]=h_par_y[particle_idx]; pz_j_data[k]=h_par_z[particle_idx];
            vx_j_data[k]=h_vel_x[particle_idx]; vy_j_data[k]=h_vel_y[particle_idx]; vz_j_data[k]=h_vel_z[particle_idx];
            pw_j_data[k]=h_par_w[particle_idx];
        }
    }
#endif

    const auto phase_h2d_begin = PhaseClock::now();
    sys.phase_host_pack = std::chrono::duration<double>(phase_h2d_begin - phase_pack_begin).count();

    {
        ZoneScopedN("NBody H2D");
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
        EnqueueWriteMeshBuffer(cq, pwj, pw_j_data, block_for_phase_timers);
    }

    const auto phase_workload_begin = PhaseClock::now();
    sys.phase_h2d = std::chrono::duration<double>(phase_workload_begin - phase_h2d_begin).count();
    {
        ZoneScopedN("NBody workload");
        const auto device_count = mesh_device->num_devices();
        const auto target_elements = px_data.size() / device_count;
        nbody_profiling::record_workload(
            *persistent.mesh_device, persistent.workload, "NBodySFPU", sys.N,
            sys.num_cores / device_count,
            {target_elements, px_j_data.size(), target_elements, py_j_data.size(),
             target_elements, pz_j_data.size(), target_elements, vx_j_data.size(),
             target_elements, vy_j_data.size(), target_elements, vz_j_data.size(), pw_j_data.size()},
            {target_elements, target_elements, target_elements,
             target_elements, target_elements, target_elements});
        EnqueueMeshWorkload(cq, persistent.workload, block_for_phase_timers);
    }
    const auto phase_d2h_begin = PhaseClock::now();
    // Without a workload fence this interval measures submission, not execution.
    sys.phase_workload = block_for_phase_timers
        ? std::chrono::duration<double>(phase_d2h_begin - phase_workload_begin).count() : 0.0;
    sys.kernel_time = sys.phase_workload;

    std::vector<float> ax_data;
    std::vector<float> ay_data;
    std::vector<float> az_data;
    std::vector<float> adx_data;
    std::vector<float> ady_data;
    std::vector<float> adz_data;
    ax_data.resize(px_data.size()); ay_data.resize(px_data.size()); az_data.resize(px_data.size());
    adx_data.resize(px_data.size()); ady_data.resize(px_data.size()); adz_data.resize(px_data.size());

    {
        ZoneScopedN("NBody D2H");
        EnqueueReadMeshBuffer(cq, ax_data, ax, true /* blocking */);
        EnqueueReadMeshBuffer(cq, ay_data, ay, true /* blocking */);
        EnqueueReadMeshBuffer(cq, az_data, az, true /* blocking */);

        EnqueueReadMeshBuffer(cq, adx_data, adx, true /* blocking */);
        EnqueueReadMeshBuffer(cq, ady_data, ady, true /* blocking */);
        EnqueueReadMeshBuffer(cq, adz_data, adz, true /* blocking */);
    }
    const auto phase_post_begin = PhaseClock::now();
    sys.phase_d2h = std::chrono::duration<double>(phase_post_begin - phase_d2h_begin).count();

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
    for (uint32_t i = 0; i < local_N_per_rank; ++i) {
#if defined(_WH_MULTIHOST) || defined(_WH_MESH)
      const uint32_t tile=i/TILE_HEIGHT, row=i%TILE_HEIGHT;
      double sax=0.0, say=0.0, saz=0.0, sadx=0.0, sady=0.0, sadz=0.0;
#ifdef NBODY_DEVICE_ROW_REDUCE
      constexpr uint32_t output_columns = 1;
#else
      constexpr uint32_t output_columns = TILE_WIDTH;
#endif
      for (uint32_t column=0; column<output_columns; ++column) {
          const uint32_t face=(row/16)*2+column/16;
          const uint32_t k=tile*tile_size+face*256+(row%16)*16+column%16;
          sax+=ax_data[k]; say+=ay_data[k]; saz+=az_data[k];
          sadx+=adx_data[k]; sady+=ady_data[k]; sadz+=adz_data[k];
      }
      h_acc_x[i+sys.start]=sax; h_acc_y[i+sys.start]=say; h_acc_z[i+sys.start]=saz;
      h_adot_x[i+sys.start]=sadx; h_adot_y[i+sys.start]=sady; h_adot_z[i+sys.start]=sadz;
#else
      h_acc_x[i+sys.start]=static_cast<double>(ax_data[i]); h_acc_y[i+sys.start]=static_cast<double>(ay_data[i]);
      h_acc_z[i+sys.start]=static_cast<double>(az_data[i]); h_adot_x[i+sys.start]=static_cast<double>(adx_data[i]);
      h_adot_y[i+sys.start]=static_cast<double>(ady_data[i]); h_adot_z[i+sys.start]=static_cast<double>(adz_data[i]);
#endif
    }

    sys.phase_host_postprocess = std::chrono::duration<double>(PhaseClock::now() - phase_post_begin).count();

#if !defined(_WH_MULTIHOST)
  nbody_profiling::flush_device(*mesh_device);
  mesh_device->close();
#endif
}
#elif defined(_GEMM_BASED)
namespace {

struct alignas(64) NBodyTile32 {
  std::array<double, BLKSIZE * BLKSIZE> values{};

  double* data() { return values.data(); }
  const double* data() const { return values.data(); }
};

struct NBodyTargetTile {
  NBodyTile32 Ar;
  NBodyTile32 Av;
  int start = 0;
  int count = 0;
};

struct NBodyForceWorkspace {
  NBodyTile32 Br;
  NBodyTile32 Bv;
  NBodyTile32 P;
  NBodyTile32 D;
  NBodyTile32 C;
  NBodyTile32 W;
  NBodyTile32 U;
  NBodyTile32 HW;
  NBodyTile32 HU;
};

// Row-major GEMM over physical 32x32 tiles. The full result tile is cleared,
// while active_k and active_n skip padding known to be identically zero on CPU.
void nbody_tile_gemm(const double* A, const double* B, double* C,
                     int active_k, int active_n) {
  constexpr int TILE = BLKSIZE;
  std::fill(C, C + TILE * TILE, 0.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              TILE, active_n, active_k,
              1.0, A, TILE, B, TILE,
              0.0, C, TILE);
}

}  // namespace

void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y,
                       double* __restrict h_par_z, double* __restrict h_par_w,
                       double* __restrict h_vel_x, double* __restrict h_vel_y,
                       double* __restrict h_vel_z, double* __restrict h_acc_x,
                       double* __restrict h_acc_y, double* __restrict h_acc_z,
                       double* __restrict h_adot_x, double* __restrict h_adot_y,
                       double* __restrict h_adot_z, Sys& sys) {
  constexpr int B = BLKSIZE;

  #pragma omp parallel for
  for (int i = 0; i < sys.N; ++i) {
    h_acc_x[i] = h_acc_y[i] = h_acc_z[i] = 0.0;
    h_adot_x[i] = h_adot_y[i] = h_adot_z[i] = 0.0;
  }

  const int target_blocks = (sys.end - sys.start + B - 1) / B;
  const int source_blocks = (sys.N + B - 1) / B;
  std::vector<NBodyTargetTile> targets(target_blocks);

  // Targets remain packed for the force evaluation. Source tiles are packed
  // into per-thread storage immediately before use so their padded matrices
  // are hot in cache during the four GEMMs.
  #pragma omp parallel for schedule(static)
  for (int iblock = 0; iblock < target_blocks; ++iblock) {
    NBodyTargetTile& target = targets[iblock];
    target.start = sys.start + iblock * B;
    target.count = std::min(B, sys.end - target.start);

    for (int ii = 0; ii < target.count; ++ii) {
      const int i = target.start + ii;
      const double x = h_par_x[i], y = h_par_y[i], z = h_par_z[i];
      const double vx = h_vel_x[i], vy = h_vel_y[i], vz = h_vel_z[i];

      target.Ar.values[ii * B + 0] = x*x + y*y + z*z;
      target.Ar.values[ii * B + 1] = x;
      target.Ar.values[ii * B + 2] = y;
      target.Ar.values[ii * B + 3] = z;
      target.Ar.values[ii * B + 4] = 1.0;

      target.Av.values[ii * B + 0] = 1.0;
      target.Av.values[ii * B + 1] = vx;
      target.Av.values[ii * B + 2] = vy;
      target.Av.values[ii * B + 3] = vz;
      target.Av.values[ii * B + 4] = x;
      target.Av.values[ii * B + 5] = y;
      target.Av.values[ii * B + 6] = z;
      target.Av.values[ii * B + 7] = x*vx + y*vy + z*vz;
    }
  }

  #pragma omp parallel
  {
    NBodyForceWorkspace workspace;

    #pragma omp for schedule(static)
    for (int iblock = 0; iblock < target_blocks; ++iblock) {
      const NBodyTargetTile& target = targets[iblock];

      for (int jblock = 0; jblock < source_blocks; ++jblock) {
        const int source_start = jblock * B;
        const int source_count = std::min(B, sys.N - source_start);
        workspace.Br = NBodyTile32{};
        workspace.Bv = NBodyTile32{};
        workspace.P = NBodyTile32{};

        for (int jj = 0; jj < source_count; ++jj) {
          const int j = source_start + jj;
          const double x = h_par_x[j], y = h_par_y[j], z = h_par_z[j];
          const double vx = h_vel_x[j], vy = h_vel_y[j], vz = h_vel_z[j];

          workspace.Br.values[0 * B + jj] = 1.0;
          workspace.Br.values[1 * B + jj] = -2.0*x;
          workspace.Br.values[2 * B + jj] = -2.0*y;
          workspace.Br.values[3 * B + jj] = -2.0*z;
          workspace.Br.values[4 * B + jj] = x*x + y*y + z*z;

          workspace.Bv.values[0 * B + jj] = x*vx + y*vy + z*vz;
          workspace.Bv.values[1 * B + jj] = -x;
          workspace.Bv.values[2 * B + jj] = -y;
          workspace.Bv.values[3 * B + jj] = -z;
          workspace.Bv.values[4 * B + jj] = -vx;
          workspace.Bv.values[5 * B + jj] = -vy;
          workspace.Bv.values[6 * B + jj] = -vz;
          workspace.Bv.values[7 * B + jj] = 1.0;

          workspace.P.values[jj * B + 0] = 1.0;
          workspace.P.values[jj * B + 1] = x;
          workspace.P.values[jj * B + 2] = y;
          workspace.P.values[jj * B + 3] = z;
          workspace.P.values[jj * B + 4] = vx;
          workspace.P.values[jj * B + 5] = vy;
          workspace.P.values[jj * B + 6] = vz;
        }

        nbody_tile_gemm(
            target.Ar.data(), workspace.Br.data(), workspace.D.data(), 5, B);
        nbody_tile_gemm(
            target.Av.data(), workspace.Bv.data(), workspace.C.data(), 8, B);

        for (int ii = 0; ii < B; ++ii) {
          for (int jj = 0; jj < B; ++jj) {
            const bool valid = ii < target.count && jj < source_count;
            const int i = target.start + ii;
            const int j = source_start + jj;
            if (!valid || i == j) {
              workspace.W.values[ii * B + jj] = 0.0;
              workspace.U.values[ii * B + jj] = 0.0;
              continue;
            }

            const double s =
                std::max(0.0, workspace.D.values[ii * B + jj]) + EPS2;
            const double inv_r = 1.0 / std::sqrt(s);
            const double w = h_par_w[j] * inv_r * inv_r * inv_r;
            workspace.W.values[ii * B + jj] = w;
            workspace.U.values[ii * B + jj] =
                -3.0 * w * workspace.C.values[ii * B + jj] / s;
          }
        }

        nbody_tile_gemm(
            workspace.W.data(), workspace.P.data(), workspace.HW.data(), B, 7);
        nbody_tile_gemm(
            workspace.U.data(), workspace.P.data(), workspace.HU.data(), B, 7);

        for (int ii = 0; ii < target.count; ++ii) {
          const int i = target.start + ii;
          const double sw = workspace.HW.values[ii * B];
          const double su = workspace.HU.values[ii * B];
          h_acc_x[i] += workspace.HW.values[ii * B + 1] - h_par_x[i] * sw;
          h_acc_y[i] += workspace.HW.values[ii * B + 2] - h_par_y[i] * sw;
          h_acc_z[i] += workspace.HW.values[ii * B + 3] - h_par_z[i] * sw;
          h_adot_x[i] += workspace.HW.values[ii * B + 4] - h_vel_x[i] * sw
                       + workspace.HU.values[ii * B + 1] - h_par_x[i] * su;
          h_adot_y[i] += workspace.HW.values[ii * B + 5] - h_vel_y[i] * sw
                       + workspace.HU.values[ii * B + 2] - h_par_y[i] * su;
          h_adot_z[i] += workspace.HW.values[ii * B + 6] - h_vel_z[i] * sw
                       + workspace.HU.values[ii * B + 3] - h_par_z[i] * su;
        }
      }
    }
  }
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
