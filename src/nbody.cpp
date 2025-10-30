#include "nbody.h"
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

NBody::NBody(int Nparticles, double v_max, double r_max, double m_min, double m_max)
    : N(Nparticles), vmax(v_max), rmax(r_max), mmin(m_min), mmax(m_max) 
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    alpha = n + 1.0;

    int nlocal = N / size;
    int rest = N % size;
    nlocal = (rank < rest) ? nlocal + 1 : nlocal;
    int offset = (rank < rest) ? 0 : rest;

    sys.size = size;
    sys.rank = rank;

    sys.N = N;
    sys.start = rank * nlocal + offset;
    sys.end = (rank + 1) * nlocal + offset;

    h_par_x = (double*)malloc(N * sizeof(double));
    h_par_y = (double*)malloc(N * sizeof(double));
    h_par_z = (double*)malloc(N * sizeof(double));
    h_par_w = (double*)malloc(N * sizeof(double));

    h_vel_x = (double*)malloc(N * sizeof(double));
    h_vel_y = (double*)malloc(N * sizeof(double));
    h_vel_z = (double*)malloc(N * sizeof(double));

    h_acc_x = (double*)malloc(N * sizeof(double));
    h_acc_y = (double*)malloc(N * sizeof(double));
    h_acc_z = (double*)malloc(N * sizeof(double));

    h_adot_x = (double*)malloc(N * sizeof(double));
    h_adot_y = (double*)malloc(N * sizeof(double));
    h_adot_z = (double*)malloc(N * sizeof(double));

    h_acc1_x = (double*)malloc(N * sizeof(double));
    h_acc1_y = (double*)malloc(N * sizeof(double));
    h_acc1_z = (double*)malloc(N * sizeof(double));

    h_a1dot_x = (double*)malloc(N * sizeof(double));
    h_a1dot_y = (double*)malloc(N * sizeof(double));
    h_a1dot_z = (double*)malloc(N * sizeof(double));

    h_a2dot_x = (double*)malloc(N * sizeof(double));
    h_a2dot_y = (double*)malloc(N * sizeof(double));
    h_a2dot_z = (double*)malloc(N * sizeof(double));

    h_a3dot_x = (double*)malloc(N * sizeof(double));
    h_a3dot_y = (double*)malloc(N * sizeof(double));
    h_a3dot_z = (double*)malloc(N * sizeof(double));
}


void NBody::initialize() {
  // std::mt19937 engine(std::random_device{}());
  std::mt19937 engine(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  int i = 0;
  while(i < N){
    double x_tmp = dist(engine) * rmax;
    double y_tmp = dist(engine) * rmax;
    double z_tmp = dist(engine) * rmax;
    double r2 = x_tmp * x_tmp + y_tmp * y_tmp + z_tmp * z_tmp;
    double r2_max = rmax * rmax;
    if (r2 < r2_max) {
      h_par_x[i] = x_tmp;
      h_par_y[i] = y_tmp;
      h_par_z[i] = z_tmp;
      i++;
    }
  }

  double mmin2 = std::pow(mmin, alpha);
  double mmax2 = std::pow(mmax, alpha);
  double invalpha = 1.0/alpha;
// #pragma omp parallel for
// #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    h_vel_x[i] = dist(engine) * vmax;
    h_vel_y[i] = dist(engine) * vmax;
    h_vel_z[i] = dist(engine) * vmax;
    double tmp_w = (mmax2 - mmin2) * dist(engine) + mmin2;
    h_par_w[i] = std::pow(tmp_w, invalpha);
  }

  T = compute_kinetic(h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
  U = compute_potential(h_par_x, h_par_y, h_par_z, h_par_w, N);
  double Q = virial_ratio(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
  float invsqrtQ = 1.0/std::sqrt(Q);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    h_vel_x[i] = h_vel_x[i] * invsqrtQ;
    h_vel_y[i] = h_vel_y[i] * invsqrtQ;
    h_vel_z[i] = h_vel_z[i] * invsqrtQ;
  }
  T = compute_kinetic(h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
  Q = virial_ratio(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
}

void NBody::initializewithzeros() {
  for (int i = 0; i < N; i++) {
    h_par_x[i] = 0.0;
    h_par_y[i] = 0.0;
    h_par_z[i] = 0.0;
    h_par_w[i] = 0.0;
  }

  for (int i = 0; i < N; i++) {
    h_vel_x[i] = 0.0;
    h_vel_y[i] = 0.0;
    h_vel_z[i] = 0.0;
  }
}

double NBody::initialdt() {
  double temp_min = std::numeric_limits<double>::max();
  for (int i = 0; i < N; i++) {
    double a2x = h_acc_x[i] * h_acc_x[i];
    double a2y = h_acc_y[i] * h_acc_y[i];
    double a2z = h_acc_z[i] * h_acc_z[i];
    double anorm = sqrt(a2x + a2y + a2z);

    double adot2x = h_adot_x[i] * h_adot_x[i];
    double adot2y = h_adot_y[i] * h_adot_y[i];
    double adot2z = h_adot_z[i] * h_adot_z[i];
    double adotnorm = sqrt(adot2x + adot2y + adot2z);

    double current_dts = ETA0 * anorm / adotnorm;
    if (current_dts < temp_min) {
      temp_min = current_dts;
    }
  }
  return temp_min;
}

void NBody::prediction(){
#pragma omp parallel for
  for (int i = 0; i < N; i++){
    h_par_x[i] += h_vel_x[i] * dt + 0.5 * h_acc_x[i] * dt * dt + ONESIXTH * h_adot_x[i] * dt * dt * dt;
    h_par_y[i] += h_vel_y[i] * dt + 0.5 * h_acc_y[i] * dt * dt + ONESIXTH * h_adot_y[i] * dt * dt * dt;
    h_par_z[i] += h_vel_z[i] * dt + 0.5 * h_acc_z[i] * dt * dt + ONESIXTH * h_adot_z[i] * dt * dt * dt;
    
    h_vel_x[i] += h_acc_x[i] * dt + 0.5 * h_adot_x[i] * dt * dt;
    h_vel_y[i] += h_acc_y[i] * dt + 0.5 * h_adot_y[i] * dt * dt;
    h_vel_z[i] += h_acc_z[i] * dt + 0.5 * h_adot_z[i] * dt * dt;
  }
}

void NBody::snapandcrackle(){
#pragma omp parallel for
  for (int i = 0; i < N; i++){
    double invdt2 = 1.0/(dt*dt); 
    double invdt3 = 1.0/(dt*dt*dt);

    h_a2dot_x[i] = 2.0 * invdt2 * ( -3.0 * (h_acc_x[i] - h_acc1_x[i]) - (2.0 * h_adot_x[i] + h_a1dot_x[i]) * dt);
    h_a2dot_y[i] = 2.0 * invdt2 * ( -3.0 * (h_acc_y[i] - h_acc1_y[i]) - (2.0 * h_adot_y[i] + h_a1dot_y[i]) * dt);
    h_a2dot_z[i] = 2.0 * invdt2 * ( -3.0 * (h_acc_z[i] - h_acc1_z[i]) - (2.0 * h_adot_z[i] + h_a1dot_z[i]) * dt);

    h_a3dot_x[i] = 6.0 * invdt3 * (2.0 * (h_acc_x[i] - h_acc1_x[i]) + (h_adot_x[i] + h_a1dot_x[i]) * dt);
    h_a3dot_y[i] = 6.0 * invdt3 * (2.0 * (h_acc_y[i] - h_acc1_y[i]) + (h_adot_y[i] + h_a1dot_y[i]) * dt);
    h_a3dot_z[i] = 6.0 * invdt3 * (2.0 * (h_acc_z[i] - h_acc1_z[i]) + (h_adot_z[i] + h_a1dot_z[i]) * dt);
  }
}

void NBody::correction(){
#pragma omp parallel for
  for (int i = 0; i < N; i++){
    double dt3 = dt * dt * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;

    h_par_x[i] += ONE24TH * h_a2dot_x[i] * dt4 + ONE120TH * h_a3dot_x[i] * dt5;
    h_par_y[i] += ONE24TH * h_a2dot_y[i] * dt4 + ONE120TH * h_a3dot_y[i] * dt5;
    h_par_z[i] += ONE24TH * h_a2dot_z[i] * dt4 + ONE120TH * h_a3dot_z[i] * dt5;
    
    h_vel_x[i] += ONESIXTH * h_a2dot_x[i] * dt3 + ONE24TH * h_a3dot_x[i] * dt4;
    h_vel_y[i] += ONESIXTH * h_a2dot_y[i] * dt3 + ONE24TH * h_a3dot_y[i] * dt4;
    h_vel_z[i] += ONESIXTH * h_a2dot_z[i] * dt3 + ONE24TH * h_a3dot_z[i] * dt4;
  }
}

double NBody::updatedt(){
  double temp_min = std::numeric_limits<double>::max();
  for (int i = 0; i < N; i++){
    double normA2 = h_acc_x[i] * h_acc_x[i] + h_acc_y[i] * h_acc_y[i] + h_acc_z[i] * h_acc_z[i];
    double normA = sqrt(normA2);

    double normAdot2 = h_adot_x[i] * h_adot_x[i] + h_adot_y[i] * h_adot_y[i] + h_adot_z[i] * h_adot_z[i];
    double normAdot = sqrt(normAdot2);

    double normA2dots2 = h_a2dot_x[i] * h_a2dot_x[i] + h_a2dot_y[i] * h_a2dot_y[i] + h_a2dot_z[i] * h_a2dot_z[i];
    double normA2dots = sqrt(normA2dots2);

    double normA3dots2 = h_a3dot_x[i] * h_a3dot_x[i] + h_a3dot_y[i] * h_a3dot_y[i] + h_a3dot_z[i] * h_a3dot_z[i];
    double normA3dots = sqrt(normA3dots2);

    double num = normA * normA2dots + normAdot2;
    double den = normAdot * normA3dots + normA2dots2;

    double current_dts = sqrt(ETA * num / den);
    if (current_dts < temp_min) {
      temp_min = current_dts;
    }
  }
  return temp_min;
}

void NBody::timestepping(){
  double start, end;
  start =  omp_get_wtime();
  prediction();
  end =  omp_get_wtime();
  sys.t_pred = end - start; 

  start =  omp_get_wtime();
  force_calculation(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, h_acc1_x, h_acc1_y, h_acc1_z, h_a1dot_x, h_a1dot_y, h_a1dot_z, sys);
  end =  omp_get_wtime();
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH) 
  sys.t_eval = sys.kernel_time;
#else
  sys.t_eval = end - start;
#endif

  MPI_Allreduce(MPI_IN_PLACE, h_acc1_x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_acc1_y, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_acc1_z, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE, h_a1dot_x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_a1dot_y, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_a1dot_z, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#if defined(_DEBUG)
  if (rank == 0){
    test(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, h_acc1_x, h_acc1_y, h_acc1_z, h_a1dot_x, h_a1dot_y, h_a1dot_z, N);
  }
#endif

  snapandcrackle();

  start =  omp_get_wtime();
  correction();
  end =  omp_get_wtime();
  sys.t_corr = end - start; 

  dt = updatedt();

  double *tmp_acc_x;
  tmp_acc_x = h_acc_x;
  h_acc_x = h_acc1_x;
  h_acc1_x = tmp_acc_x;

  tmp_acc_x = h_adot_x;
  h_adot_x = h_a1dot_x;
  h_a1dot_x = tmp_acc_x;

  double *tmp_acc_y;
  tmp_acc_y = h_acc_y;
  h_acc_y = h_acc1_y;
  h_acc1_y = tmp_acc_y;

  tmp_acc_y = h_adot_y;
  h_adot_y = h_a1dot_y;
  h_a1dot_y = tmp_acc_y;

  double *tmp_acc_z;
  tmp_acc_z = h_acc_z;
  h_acc_z = h_acc1_z;
  h_acc1_z = tmp_acc_z;

  tmp_acc_z = h_adot_z;
  h_adot_z = h_a1dot_z;
  h_a1dot_z = tmp_acc_z;
}

void NBody::cycle(int ncycle){
  sys.t_evalv.resize(ncycle);
  sys.t_predv.resize(ncycle);
  sys.t_corrv.resize(ncycle);
  double start, end;
  if (rank == 0){
    initialize();
  }

  MPI_Bcast(h_par_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_par_y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_par_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_par_w, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(h_vel_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_vel_y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_vel_z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  start = omp_get_wtime();
  force_calculation(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, h_acc_x, h_acc_y, h_acc_z, h_adot_x, h_adot_y, h_adot_z, sys);
  end = omp_get_wtime();
#if defined(_WH_MULTIHOST) || defined(_WH_MULTICHIP) || defined(_WH_MESH) 
  sys.t_eval_init = sys.kernel_time;
#else
  sys.t_eval_init = end - start;
#endif

  MPI_Allreduce(MPI_IN_PLACE, h_acc_x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_acc_y, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_acc_z, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE, h_adot_x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_adot_y, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, h_adot_z, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#if defined(_DEBUG)
  if (rank == 0){
    test(h_par_x, h_par_y, h_par_z, h_par_w, h_vel_x, h_vel_y, h_vel_z, h_acc_x, h_acc_y, h_acc_z, h_adot_x, h_adot_y, h_adot_z, N);
  }
#endif

  dt = initialdt();
  
  T = compute_kinetic(h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
  U = compute_potential(h_par_x, h_par_y, h_par_z, h_par_w, N);
  sys.initE = T + U;

  for (int i = 0; i < ncycle; i++){
    timestepping();
    T = compute_kinetic(h_par_w, h_vel_x, h_vel_y, h_vel_z, N);
    U = compute_potential(h_par_x, h_par_y, h_par_z, h_par_w, N);
    sys.finE = T + U;
    sys.Q = -2.0 * T / U;

    if (rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &sys.t_pred, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &sys.t_eval, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &sys.t_corr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(&sys.t_pred, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&sys.t_eval, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&sys.t_corr, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      //std::cout << std::setw(COLWIDTH) << N
      //    << std::setw(COLWIDTH) << sys.size
      //    << std::setw(COLWIDTH) << sys.t_pred
      //    << std::setw(COLWIDTH) << sys.t_eval
      //    << std::setw(COLWIDTH) << sys.t_corr
      //    << std::setw(COLWIDTH) << i << std::endl;
      sys.t_evalv[i] = sys.t_eval;
      sys.t_predv[i] = sys.t_pred;
      sys.t_corrv[i] = sys.t_corr;
    }
  }

  if (rank == 0){
     sys.Err = estimate_err(sys.finE, sys.initE);
  }
  
  /*
  if (rank == 0){
      std::cout << "\nInitial Evaluation Time: " << sys.t_eval_init << std::endl;
      std::cout << "Initial Energy: " << sys.initE << std::endl;
      std::cout << "Final Energy: " << sys.finE << std::endl;
      std::cout << "Q = " << sys.Q << std::endl;
      std::cout << "Err = " << sys.Err << std::endl;
  }
  */
}

NBody::~NBody() {
    free(h_par_x); h_par_x = nullptr;
    free(h_par_y); h_par_y = nullptr;
    free(h_par_z); h_par_z = nullptr;
    free(h_par_w); h_par_w = nullptr;

    free(h_vel_x); h_vel_x = nullptr;
    free(h_vel_y); h_vel_y = nullptr;
    free(h_vel_z); h_vel_z = nullptr;

    free(h_acc_x); h_acc_x = nullptr;
    free(h_acc_y); h_acc_y = nullptr;
    free(h_acc_z); h_acc_z = nullptr;

    free(h_adot_x); h_adot_x = nullptr;
    free(h_adot_y); h_adot_y = nullptr;
    free(h_adot_z); h_adot_z = nullptr;

    free(h_acc1_x); h_acc1_x = nullptr;
    free(h_acc1_y); h_acc1_y = nullptr;
    free(h_acc1_z); h_acc1_z = nullptr;

    free(h_a1dot_x); h_a1dot_x = nullptr;
    free(h_a1dot_y); h_a1dot_y = nullptr;
    free(h_a1dot_z); h_a1dot_z = nullptr;

    free(h_a2dot_x); h_a2dot_x = nullptr;
    free(h_a2dot_y); h_a2dot_y = nullptr;
    free(h_a2dot_z); h_a2dot_z = nullptr;

    free(h_a3dot_x); h_a3dot_x = nullptr;
    free(h_a3dot_y); h_a3dot_y = nullptr;
    free(h_a3dot_z); h_a3dot_z = nullptr;
}
