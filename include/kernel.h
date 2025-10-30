#pragma once
#include "utils.h"
#include "omp.h"
#include <stdlib.h>

#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH) 
#pragma message("Compiling for Wormhole")
#else
#pragma message("Using default force_calculation declaration")
#endif

#define THREADS_PER_BLOCK 128
#define EPS2 1.0e-7

void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y, double* __restrict h_par_z,  
                      double* __restrict h_par_w, double* __restrict h_vel_x, double* __restrict h_vel_y, 
                      double* __restrict h_vel_z, double* __restrict h_acc_x, double* __restrict h_acc_y, 
                      double* __restrict h_acc_z, double* __restrict h_adot_x, double* __restrict h_adot_y, 
                      double* __restrict h_adot_z, Sys& sys);
