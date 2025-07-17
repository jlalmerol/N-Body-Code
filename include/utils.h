#pragma once
#include <vector>
#include <cmath>
#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>

#if defined(_WH)
#include "nbody_tt_integration.h"
#endif

#define PRT 5
#define EPS2 1.0e-8

#if defined(_WH) 
namespace NBodyProject {
	std::shared_ptr<Buffer> MakeBuffer(tt::tt_metal::v0::IDevice* device, uint32_t size, uint32_t page_size, bool sram);
	CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);
	std::shared_ptr<Buffer> MakeBufferFP32(tt::tt_metal::v0::IDevice* device, uint32_t n_tiles, bool sram);
	CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles);
	std::string next_arg(int& i, int argc, char** argv);
	void help(std::string_view program_name);
} // namespace NBodyProject
#endif

double timer();

struct Sys {
    int N;
    int start;
    int end;
    int rank;
    int size;
    int nthreads = 1;
    int thid = 0;
    double t_eval = 0.0;
    double t_pred = 0.0;
    double t_corr = 0.0;
    double t_eval_init = 0.0;
    double kernel_time = 0.0;
    double kernel_time_init = 0.0;
    std::vector<double> t_evalv;
    std::vector<double> t_predv;
    std::vector<double> t_corrv;
    double initE;
    double finE;
    double Q;
    double Err;
};

double InvSqrt2D(double x);
float InvSqrt2(float x);

double compute_kinetic(double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, int N);
double compute_potential(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, int N);
double virial_ratio(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, int N);
double estimate_err(double En, double E0);

void test(double* h_par_x, double* h_par_y, double* h_par_z, double* h_par_w, double* h_vel_x, double* h_vel_y, double* h_vel_z, double *h_acc_x, double *h_acc_y, double *h_acc_z, double *h_adot_x, double *h_adot_y, double *h_adot_z, int N);
