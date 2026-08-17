#include "utils.h"
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <limits>

#if defined(_WH_MULTIHOST) || defined(_WH_GEMM_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH)
#include "nbody_tt_integration.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace NBodyProject {

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

std::shared_ptr<Buffer> MakeBufferFP32(IDevice* device, uint32_t n_tiles, bool sram) {
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    // constexpr uint32_t tile_size = 4 * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_bytes * n_tiles, page_tiles * tile_bytes, sram);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    // constexpr uint32_t tile_size = 4 * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_bytes, tile_bytes, tt::DataFormat::Float32);
}

std::shared_ptr<MeshBuffer> MakeMeshBuffer(const std::shared_ptr<MeshDevice>& mesh, const uint32_t tiles_per_device, const uint32_t total_num_tiles) { 
    constexpr uint32_t TILE_HEIGHT = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_WIDTH = tt::constants::TILE_WIDTH;
    auto tile_size = TILE_WIDTH * TILE_HEIGHT;
    auto tile_size_bytes = tile_size * sizeof(float);
    auto buffer_size_bytes = total_num_tiles * tile_size_bytes;

    auto shard_shape = Shape2D{
        TILE_HEIGHT,
        TILE_WIDTH * tiles_per_device
    };
    
    auto distributed_buffer_shape = Shape2D{
        TILE_HEIGHT,
        TILE_WIDTH * total_num_tiles 
    };

    auto local_buffer_config = DeviceLocalBufferConfig{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
        // .bottom_up = false
    };

    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    return MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh.get());
}

std::shared_ptr<MeshBuffer> MakeReplicatedBuffer(const std::shared_ptr<MeshDevice>& mesh, const uint32_t rep_num_tiles) {
    constexpr uint32_t TILE_HEIGHT = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_WIDTH = tt::constants::TILE_WIDTH;
    auto tile_size = TILE_WIDTH * TILE_HEIGHT;
    auto tile_size_bytes = tile_size * sizeof(float);

    ReplicatedBufferConfig replicated_config{
        .size = tile_size_bytes * rep_num_tiles
    };

    DeviceLocalBufferConfig device_local_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
        // .bottom_up = false
    };

    return MeshBuffer::create(replicated_config, device_local_config, mesh.get());
}

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
return argv[++i];
}

void help(std::string_view program_name) {
	std::cout << "Usage: " << program_name << " [options]\n";
	std::cout << "This program demonstrates how to add two vectors using "
	            "tt-Metalium.\n";
	std::cout << "\n";
	std::cout << "Options:\n";
	std::cout << "  --device, -d <device_id>  Specify the device to run the "
	            "program on. Default is 0.\n";
	std::cout << "  --seed, -s <seed>         Specify the seed for the random "
	            "number generator. Default is random.\n";
	exit(0);
}

} // namespace NBodyProject

#endif

double timer() {
    struct timeval tmp;
    double sec;
    gettimeofday(&tmp, nullptr);
    sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;
    return sec;
}

double compute_kinetic(double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, int N) {
    double T = 0.0;
    #pragma omp parallel for reduction(+:T)
    for (int i = 0; i < N; i++) {
        double vx2 = h_vel_x[i] * h_vel_x[i];
        double vy2 = h_vel_y[i] * h_vel_y[i];
        double vz2 = h_vel_z[i] * h_vel_z[i];
        T += 0.5 * h_par_w[i] * (vx2 + vy2 + vz2);
    }
    return T;
}

double compute_potential(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, int N) {
    double U = 0.0;
    #pragma omp parallel for reduction(+:U) schedule(static)
    for (int i = 0; i < N; i++) {
        double tmp = 0.0;
        for (int j = i + 1; j < N; j++) {
            double rx2 = (h_par_x[i] - h_par_x[j]) * (h_par_x[i] - h_par_x[j]);
            double ry2 = (h_par_y[i] - h_par_y[j]) * (h_par_y[i] - h_par_y[j]);
            double rz2 = (h_par_z[i] - h_par_z[j]) * (h_par_z[i] - h_par_z[j]);
            double rij2 = rx2 + ry2 + rz2 + EPS2;
            double invrij = 1. / sqrt(rij2);
            tmp -= h_par_w[j] * h_par_w[j] * invrij;
        }
        U += tmp;
    }
    return U;
}

double InvSqrt2D(double x) {
    double halfx = 0.5 * x;
    long long i = *(long long*) &x;
    i = 0x5FE6ED2102DCBFDA - (i >> 1);
    double y = *(double*) &i;
    y *= 1.50087895511633457 - halfx * y * y; 
    y *= 1.50000057967625766 - halfx * y * y; 
    y *= 1.5000000000002520  - halfx * y * y; 
    y *= 1.5000000000000000  - halfx * y * y; 
    return y;
}

float InvSqrt2(float x) {
    float halfx = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5F376908 - (i >> 1);
    float y = *(float*)&i;
    y *= 1.50087896f - halfx * y * y;
    y *= 1.50000057f - halfx * y * y;
    return y;
}

double virial_ratio(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, int N) {    
    double T = compute_kinetic(h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
    double U = compute_potential(h_par_x, h_par_y, h_par_z, h_par_w, N);
    return -2.0 * T / U;
}

double estimate_err(double En, double E0) {
    return std::abs(En - E0) / std::abs(E0);
}

namespace {

struct ValidationMetrics {
    double relative_l2 = 0.0;
    double rms_absolute = 0.0;
    double max_absolute = 0.0;
    double max_robust_relative = 0.0;
    double reference_rms = 0.0;
    double relative_floor = 0.0;
    int max_absolute_particle = -1;
    int max_absolute_component = -1;
    int max_relative_particle = -1;
    int max_relative_component = -1;
    double max_absolute_reference = 0.0;
    double max_absolute_result = 0.0;
    double max_relative_reference = 0.0;
    double max_relative_result = 0.0;
    std::size_t non_finite = 0;
};

ValidationMetrics calculate_validation_metrics(
    const std::array<const double*, 3>& reference,
    const std::array<const double*, 3>& result,
    int n) {
    ValidationMetrics metrics;
    const std::size_t count = static_cast<std::size_t>(n) * reference.size();
    if (count == 0) return metrics;

    long double reference_squared_sum = 0.0L;
    long double error_squared_sum = 0.0L;
    for (int i = 0; i < n; ++i) {
        for (std::size_t component = 0; component < reference.size(); ++component) {
            const double ref = reference[component][i];
            const double got = result[component][i];
            if (!std::isfinite(ref) || !std::isfinite(got)) {
                ++metrics.non_finite;
                continue;
            }
            const long double error = static_cast<long double>(got) - ref;
            reference_squared_sum += static_cast<long double>(ref) * ref;
            error_squared_sum += error * error;
            const double absolute_error = std::abs(got - ref);
            if (absolute_error > metrics.max_absolute) {
                metrics.max_absolute = absolute_error;
                metrics.max_absolute_particle = i;
                metrics.max_absolute_component = static_cast<int>(component);
                metrics.max_absolute_reference = ref;
                metrics.max_absolute_result = got;
            }
        }
    }

    metrics.reference_rms = std::sqrt(static_cast<double>(reference_squared_sum / count));
    metrics.rms_absolute = std::sqrt(static_cast<double>(error_squared_sum / count));
    metrics.relative_l2 = reference_squared_sum > 0.0L
        ? std::sqrt(static_cast<double>(error_squared_sum / reference_squared_sum))
        : (error_squared_sum == 0.0L ? 0.0 : std::numeric_limits<double>::infinity());

    // A component close to zero must not turn harmless roundoff into an
    // enormous relative error. Scale the floor to the RMS reference magnitude,
    // while retaining an absolute lower bound for an all-zero reference.
    metrics.relative_floor = std::max(1.0e-12, metrics.reference_rms * 1.0e-6);
    for (int i = 0; i < n; ++i) {
        for (std::size_t component = 0; component < reference.size(); ++component) {
            const double ref = reference[component][i];
            const double got = result[component][i];
            if (!std::isfinite(ref) || !std::isfinite(got)) continue;
            const double robust_relative =
                std::abs(got - ref) / std::max(std::abs(ref), metrics.relative_floor);
            if (robust_relative > metrics.max_robust_relative) {
                metrics.max_robust_relative = robust_relative;
                metrics.max_relative_particle = i;
                metrics.max_relative_component = static_cast<int>(component);
                metrics.max_relative_reference = ref;
                metrics.max_relative_result = got;
            }
        }
    }
    return metrics;
}

void print_validation_metrics(const char* quantity, const ValidationMetrics& metrics) {
    static constexpr std::array<const char*, 3> component_names = {"x", "y", "z"};
    const auto component_name = [&](int component) {
        return component >= 0 && component < static_cast<int>(component_names.size())
            ? component_names[component]
            : "n/a";
    };

    std::cout << std::left << std::setw(14) << quantity
              << std::right << std::scientific << std::setprecision(6)
              << " rel_L2=" << std::setw(13) << metrics.relative_l2
              << " rms_abs=" << std::setw(13) << metrics.rms_absolute
              << " ref_rms=" << std::setw(13) << metrics.reference_rms << '\n'
              << std::setw(14) << ""
              << " max_abs=" << std::setw(13) << metrics.max_absolute
              << " at=" << metrics.max_absolute_particle << ':'
              << component_name(metrics.max_absolute_component)
              << " (ref=" << metrics.max_absolute_reference
              << ", got=" << metrics.max_absolute_result << ')'
              << '\n'
              << std::setw(14) << ""
              << " max_rel*=" << std::setw(13) << metrics.max_robust_relative
              << " floor=" << metrics.relative_floor
              << " at=" << metrics.max_relative_particle << ':'
              << component_name(metrics.max_relative_component)
              << " (ref=" << metrics.max_relative_reference
              << ", got=" << metrics.max_relative_result << ')'
              << " non_finite=" << metrics.non_finite << '\n';
}

}  // namespace

void test(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, double *h_acc_x, double *h_acc_y, double *h_acc_z, double *h_adot_x, double *h_adot_y, double *h_adot_z, int N) {
    std::vector<double> ax(N, 0.0), ay(N, 0.0), az(N, 0.0);
    std::vector<double> a1x(N, 0.0), a1y(N, 0.0), a1z(N, 0.0);

    // Each target is independent, matching the device's all-sources-per-target
    // formulation and avoiding races in the parallel reference calculation.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double ref_ax = 0.0, ref_ay = 0.0, ref_az = 0.0;
        double ref_jx = 0.0, ref_jy = 0.0, ref_jz = 0.0;
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            double dx = h_par_x[j] - h_par_x[i];
            double dy = h_par_y[j] - h_par_y[i];
            double dz = h_par_z[j] - h_par_z[i];
            double dvx = h_vel_x[j] - h_vel_x[i];
            double dvy = h_vel_y[j] - h_vel_y[i];
            double dvz = h_vel_z[j] - h_vel_z[i];
            double distance = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + EPS2);
            double sqrdistj = h_par_w[j] * distance * distance * distance;
            double rinv2 = -3.0 * distance * distance;
            double q1 = rinv2 * (dx * dvx + dy * dvy + dz * dvz);
            ref_ax += dx * sqrdistj;
            ref_ay += dy * sqrdistj;
            ref_az += dz * sqrdistj;
            ref_jx += sqrdistj * (q1 * dx + dvx);
            ref_jy += sqrdistj * (q1 * dy + dvy);
            ref_jz += sqrdistj * (q1 * dz + dvz);
        }
        ax[i] = ref_ax; ay[i] = ref_ay; az[i] = ref_az;
        a1x[i] = ref_jx; a1y[i] = ref_jy; a1z[i] = ref_jz;
    }
    
    const auto saved_flags = std::cout.flags();
    const auto saved_precision = std::cout.precision();
    std::cout << "\nDEBUG force validation (double host reference, N=" << N << ")\n";
    std::cout << "max_rel* uses max(|reference|, floor); floor = ref_rms * 1e-6, minimum 1e-12\n";
    print_validation_metrics(
        "acceleration",
        calculate_validation_metrics(
            {ax.data(), ay.data(), az.data()}, {h_acc_x, h_acc_y, h_acc_z}, N));
    print_validation_metrics(
        "jerk",
        calculate_validation_metrics(
            {a1x.data(), a1y.data(), a1z.data()}, {h_adot_x, h_adot_y, h_adot_z}, N));
    std::cout.flags(saved_flags);
    std::cout.precision(saved_precision);
}
