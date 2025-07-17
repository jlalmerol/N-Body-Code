#pragma once

#include <iostream>
#include <iomanip>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <mpi.h>
#include "utils.h"
#include "kernel.h"

#define THREADS_PER_BLOCK 128
#define G 1.0
#define ETA0 1.0e-4
#define ETA 1.0e-2
#define ONESIXTH 1.0/6.0
#define ONE24TH 1.0/24.0
#define ONE120TH 1.0/120.0
#define PRNT 3
#define COLWIDTH 15
#define COLWIDTH 15

class NBody {
public:
  double alpha;
  double n = -2.35; // salpeter distribution power-law index
  double T, U;
  int size, rank;
  Sys sys;

  double *h_par_x, *h_par_y,*h_par_z,*h_par_w;
  double *h_vel_x, *h_vel_y, *h_vel_z;
  double *h_acc_x, *h_acc_y, *h_acc_z;
  double *h_adot_x, *h_adot_y, *h_adot_z;
  double *h_acc1_x, *h_acc1_y, *h_acc1_z;
  double *h_a1dot_x, *h_a1dot_y, *h_a1dot_z;
  double *h_a2dot_x, *h_a2dot_y, *h_a2dot_z;
  double *h_a3dot_x, *h_a3dot_y, *h_a3dot_z;
  double dt;

  NBody(int Nparticles, double v_max, double r_max, double m_min, double m_max);
  ~NBody();
  void initialize(); void initializewithzeros();
  double initialdt();
  void prediction();
  void snapandcrackle();
  void correction();
  double updatedt();
  void timestepping();
  void cycle(int ncycle);
  
private:
    int N;
    double vmax;
    double rmax;
    double mmin;
    double mmax;
};
