#pragma once
#include "utils.h"
#include "kernel.h"
#include <cstdio>
#include <cstring>

#if defined(_WH)
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/work_split.hpp>

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
using namespace tt::tt_metal::v0;

#endif

#define BLKSIZE 32

#if defined(_WH)
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y,
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y,
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y,
                      double* __restrict h_adot_z, Sys sys) {
    int seed = 0x1234567;
    int device_id = sys.rank;
    
    auto* device = CreateDevice(device_id);
    Program program = CreateProgram();

    CommandQueue& cq = device->command_queue();

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::vector<uint32_t> tiles_per_core;
    const uint32_t core_to_print = 4;
    const uint32_t local_N = sys.N / sys.size;
    const uint32_t n_tiles = local_N / tile_size;
    const uint32_t n_jtiles = n_tiles * tile_size;

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

    auto eps = MakeBufferFP32(device, n_tiles, false);
    auto pwj = MakeBufferFP32(device, n_jtiles, false);
    auto cns = MakeBufferFP32(device, n_tiles, false);

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

    CBHandle cb_eps = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_12, cir_buf_num_tile);
    CBHandle cb_pwj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_13, cir_buf_num_tile);
    CBHandle cb_cns = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_14, cir_buf_num_tile);

    CBHandle cb_ax = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_15, cir_buf_num_tile);
    CBHandle cb_ay = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_16, cir_buf_num_tile);
    CBHandle cb_az = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_17, cir_buf_num_tile);

    CBHandle cb_adx = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_18, cir_buf_num_tile);
    CBHandle cb_ady = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_19, cir_buf_num_tile);
    CBHandle cb_adz = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_20, cir_buf_num_tile);

    CBHandle cb_axt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_21, cir_buf_num_tile);
    CBHandle cb_ayt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_22, cir_buf_num_tile);
    CBHandle cb_azt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_23, cir_buf_num_tile);

    CBHandle cb_adxt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_26, cir_buf_num_tile);
    CBHandle cb_adyt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_27, cir_buf_num_tile);
    CBHandle cb_adzt = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_28, cir_buf_num_tile);

    CBHandle cb_tmp = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_24, cir_buf_num_tile);
    CBHandle cb_tj = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_25, cir_buf_num_tile);

    CBHandle cb_tmp0 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_29, cir_buf_num_tile);
    CBHandle cb_tmp1 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_30, cir_buf_num_tile);
    CBHandle cb_tmp2 = MakeCircularBufferFP32(program, all_device_cores, tt::CBIndex::c_31, cir_buf_num_tile);

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
        (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
        (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
        (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
        (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
        (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
        (std::uint32_t)tt::CBIndex::c_12, (std::uint32_t)tt::CBIndex::c_13,
        (std::uint32_t)tt::CBIndex::c_14,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_15,
        (std::uint32_t)tt::CBIndex::c_16,
        (std::uint32_t)tt::CBIndex::c_17,
        (std::uint32_t)tt::CBIndex::c_18,
        (std::uint32_t)tt::CBIndex::c_19,
        (std::uint32_t)tt::CBIndex::c_20,
    };

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1,
        (std::uint32_t)tt::CBIndex::c_2, (std::uint32_t)tt::CBIndex::c_3,
        (std::uint32_t)tt::CBIndex::c_4, (std::uint32_t)tt::CBIndex::c_5,
        (std::uint32_t)tt::CBIndex::c_6, (std::uint32_t)tt::CBIndex::c_7,
        (std::uint32_t)tt::CBIndex::c_8, (std::uint32_t)tt::CBIndex::c_9,
        (std::uint32_t)tt::CBIndex::c_10, (std::uint32_t)tt::CBIndex::c_11,
        (std::uint32_t)tt::CBIndex::c_12, (std::uint32_t)tt::CBIndex::c_13,
        (std::uint32_t)tt::CBIndex::c_14,
        (std::uint32_t)tt::CBIndex::c_15,
        (std::uint32_t)tt::CBIndex::c_16,
        (std::uint32_t)tt::CBIndex::c_17,
        (std::uint32_t)tt::CBIndex::c_18,
        (std::uint32_t)tt::CBIndex::c_19,
        (std::uint32_t)tt::CBIndex::c_20,
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
        modes[static_cast<uint32_t>(tt::CBIndex::c_21)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_22)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_23)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_24)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_25)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_26)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_27)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_28)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_29)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_30)] = UnpackToDestMode::UnpackToDestFp32;
        modes[static_cast<uint32_t>(tt::CBIndex::c_31)] = UnpackToDestMode::UnpackToDestFp32;
        return modes;
    }(),
    .math_approx_mode = false,
    .compile_args = compute_compile_time_args,
    .defines = {},
    .opt_level = KernelBuildOptLevel::O3
};

    auto reader = CreateKernel(
        program,
        "n_body/kernels/"
        "nb_read.cpp",
        all_device_cores,
        DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args});
    auto writer = CreateKernel(
        program,
        "n_body/kernels/"
        "nb_write.cpp",
        all_device_cores,
        DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args});
    auto compute = CreateKernel(
        program,
        "n_body/kernels/"
        "nb_compute.cpp",
        all_device_cores,
	compute_config
		);

    constexpr bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, n_tiles, row_major);

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
            eps->address(),
            pwj->address(), cns->address(),
        });
        SetRuntimeArgs(program, writer, core, {
            num_tiles_per_core, start_tile_id,
            ax->address(),
            ay->address(),
            az->address(),
            adx->address(),
            ady->address(),
            adz->address(),
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

    std::vector<float> eps_data, cns_data;
    eps_data.resize(tile_size * n_tiles);
    cns_data.resize(tile_size * n_tiles);

    int nsize = px_data.size();
    std::fill(eps_data.begin(), eps_data.end(), EPS2);
    std::fill(cns_data.begin(), cns_data.end(), -3.0f);
    
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

    for (int i = 0; i < n_jtiles; ++i) {
        for (int j = 0; j < tile_size; ++j) {
            int k = i * tile_size + j;
            px_j_data[k] = px_data[i + sys.start];
            py_j_data[k] = py_data[i + sys.start];
            pz_j_data[k] = py_data[i + sys.start];
            vx_j_data[k] = vx_data[i + sys.start];
            vy_j_data[k] = vy_data[i + sys.start];
            vz_j_data[k] = vz_data[i + sys.start];
	          pw_j_data[k] = h_par_w[i + sys.start];
        }
    }

    std::vector<float> ax_data;
    std::vector<float> ay_data;
    std::vector<float> az_data;
    std::vector<float> adx_data;
    std::vector<float> ady_data;
    std::vector<float> adz_data;

    //std::cout << "Start execution!" << std::endl;
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

    EnqueueWriteBuffer(cq, eps, eps_data, false);
    EnqueueWriteBuffer(cq, pwj, pw_j_data, false);
    EnqueueWriteBuffer(cq, cns, cns_data, false);
    
    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);
    tt::tt_metal::Synchronize(device);
    auto end = std::chrono::high_resolution_clock::now(); 
    
    EnqueueReadBuffer(cq, ax, ax_data, true);
    EnqueueReadBuffer(cq, ay, ay_data, true);
    EnqueueReadBuffer(cq, az, az_data, true);

    EnqueueReadBuffer(cq, adx, adx_data, true);
    EnqueueReadBuffer(cq, ady, ady_data, true);
    EnqueueReadBuffer(cq, adz, adz_data, true);

    std::chrono::duration<double, std::milli> elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    double milliseconds = elapsed_ms.count();
    double seconds = milliseconds / 1000.0;

    //std::cout << "Kernel execution finished!" << std::endl;
    sys.kernel_time = seconds;

    for (size_t i = 0; i < n_jtiles; ++i) {
    	h_acc_x[i + sys.start] = static_cast<double>(ax_data[i]);
    	h_acc_y[i + sys.start] = static_cast<double>(ay_data[i]);
    	h_acc_z[i + sys.start] = static_cast<double>(az_data[i]);
    	h_adot_x[i + sys.start] = static_cast<double>(adx_data[i]);
    	h_adot_y[i + sys.start] = static_cast<double>(ady_data[i]);
    	h_adot_z[i + sys.start] = static_cast<double>(adz_data[i]);
	}

    //std::cout << "Time taken: " << seconds << " seconds for N = " << nsize << " bodies." << std::endl;
    
    CloseDevice(device);
}
#else
void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z, 
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y, 
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y, 
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y, 
                      double* __restrict h_adot_z, Sys sys) {
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
