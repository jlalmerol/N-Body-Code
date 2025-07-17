#include "utils.h"
#include <iostream>

#if defined(_WH)
#include "nbody_tt_integration.h"

using namespace tt;
using namespace tt::tt_metal;

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
    constexpr uint32_t tile_size = 4 * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = 4 * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
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
    for (int i = 0; i < N; i++) {
        double tmp = 0.0;
        #pragma omp parallel for reduction(+:tmp)
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

void test(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, double *h_acc_x, double *h_acc_y, double *h_acc_z, double *h_adot_x, double *h_adot_y, double *h_adot_z, int N) {
    double *ax = new double [N];
    double *ay = new double [N];
    double *az = new double [N];
    double *a1x = new double [N];
    double *a1y = new double [N];
    double *a1z = new double [N];
    double energy = 0.0;

    // Variables to store the maximum values
    double max_ax = 0, max_ay = 0, max_az = 0;
    double max_a1x = 0, max_a1y = 0, max_a1z = 0;

    for(int i = 0; i < N; i++){
        ax[i] = 0.; az[i] = 0.;
        ay[i] = 0.; a1x[i] = 0.; a1y[i] = 0.; a1z[i] = 0.;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double dx = h_par_x[j] - h_par_x[i];
            double dy = h_par_y[j] - h_par_y[i];
            double dz = h_par_z[j] - h_par_z[i];
	    if (i == 0 && j < 5){
		    std::cout << "dx["<< i << "," << j << "] : " << dx << std::endl;
		    std::cout << "dy["<< i << "," << j << "] : " << dy << std::endl;
		    std::cout << "dz["<< i << "," << j << "] : " << dz << std::endl;
	    }
            double dvx = h_vel_x[j] - h_vel_x[i];
            double dvy = h_vel_y[j] - h_vel_y[i];
            double dvz = h_vel_z[j] - h_vel_z[i];
            double distance = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + EPS2);
            double sqrdistj = h_par_w[j] * distance * distance * distance;
            double sqrdisti = h_par_w[i] * distance * distance * distance;
            energy -= h_par_w[j] * h_par_w[i] * distance;
            double rinv2 = -3.0f * distance * distance;
            double q1 = rinv2 * (dx * dvx + dy * dvy + dz * dvz);
            ax[i] += dx * sqrdistj;
            ay[i] += dy * sqrdistj;
            az[i] += dz * sqrdistj;
            ax[j] -= dx * sqrdisti;
            ay[j] -= dy * sqrdisti;
            az[j] -= dz * sqrdisti;
            a1x[i] += sqrdistj * (q1 * dx + dvx);
            a1y[i] += sqrdistj * (q1 * dy + dvy);
            a1z[i] += sqrdistj * (q1 * dz + dvz);
            a1x[j] -= sqrdisti * (q1 * dx + dvx);
            a1y[j] -= sqrdisti * (q1 * dy + dvy);
            a1z[j] -= sqrdisti * (q1 * dz + dvz);
        }
    }
    
    std::cout << "Relative difference of acc from reference code:" << std::endl;
    for (int i = 0; i < N; i++) {
        double value_ax = std::abs((ax[i] - h_acc_x[i]) / ax[i]);
        double value_ay = std::abs((ay[i] - h_acc_y[i]) / ay[i]);
        double value_az = std::abs((az[i] - h_acc_z[i]) / az[i]);

        if (i < PRT) {
            printf("%.12g  %.12g  %.12g\n", value_ax, value_ay, value_az);
        }

        if (value_ax > max_ax) max_ax = value_ax;
        if (value_ay > max_ay) max_ay = value_ay;
        if (value_az > max_az) max_az = value_az;
    }

    std::cout << "Relative difference of adot from reference code:" << std::endl;
    for (int i = 0; i < N; i++) {
        double value_a1x = std::abs((a1x[i] - h_adot_x[i]) / a1x[i]);
        double value_a1y = std::abs((a1y[i] - h_adot_y[i]) / a1y[i]);
        double value_a1z = std::abs((a1z[i] - h_adot_z[i]) / a1z[i]);

        if (i < PRT) {
            printf("%.12g  %.12g  %.12g\n", value_a1x, value_a1y, value_a1z);
        }

        if (value_a1x > max_a1x) max_a1x = value_a1x;
        if (value_a1y > max_a1y) max_a1y = value_a1y;
        if (value_a1z > max_a1z) max_a1z = value_a1z;
    }

    delete[] ax;
    delete[] ay;
    delete[] az;
    delete[] a1x;
    delete[] a1y;
    delete[] a1z;
}
