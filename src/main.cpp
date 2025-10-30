#include <mpi.h>
#include "nbody.h"
#include "utils.h"

#include <iomanip>
#include <ctime>
#include <iostream>

#include <omp.h>

#define COLWIDTH 15

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int N = 16384;
  int Ncycle = 10;

  for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--N" && i + 1 < argc) {
          try {
               N = std::stoi(argv[++i]);
               if (N <= 0) throw std::invalid_argument("N must be positive.");
          } catch (const std::exception& e) {
               std::cerr << "Error parsing --N: " << e.what() << std::endl;
               MPI_Finalize();
               return 1;
          }
      }
      else if (arg == "--Ncycle" && i + 1 < argc) {
          try {
               Ncycle = std::stoi(argv[++i]);
               if (Ncycle <= 0) throw std::invalid_argument("Ncycle must be positive.");
           } catch (const std::exception& e) {
               std::cerr << "Error parsing --Ncycle: " << e.what() << std::endl;
               MPI_Finalize();
               return 1;
           }
       }
       else {
             std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
             std::cerr << "Usage: ./program [--N <value>] [--Ncycle <value>]" << std::endl;
             MPI_Finalize();
             return 1;
       }
   }

/*
  if (argc > 1) {
    try {
      N = std::stoi(argv[1]);
      if (N <= 0) {
        throw std::invalid_argument("N must be positive.");
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Invalid argument: " << e.what() << std::endl;
      return 1;
    } catch (const std::out_of_range& e) {
      std::cerr << "Argument out of range: " << e.what() << std::endl;
      return 1;
    }
  }
*/  
  
  int rank;
  if (rank == 0) {
      std::cout << "Using N = " << N << ", Ncycle = " << Ncycle << std::endl;
  }

  //int Ncycle = 10;
  double vmax = 1.0;
  double mmin = 0.01;
  double mmax = 1.0;
  double rmax = 1.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start = MPI_Wtime();
  NBody nbody(N, vmax, rmax, mmin, mmax); 
  nbody.cycle(Ncycle);
  double end = MPI_Wtime();
  double totaltime = end - start;

  if (rank == 0) {
    double now = timer();
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Time stamp(start): " << now << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(COLWIDTH) << "N"
              << std::setw(COLWIDTH) << "NPES"
              << std::setw(COLWIDTH) << "PRED"
              << std::setw(COLWIDTH) << "EVAL"
              << std::setw(COLWIDTH) << "CORR"
#if defined(_PTHD)
              << std::setw(COLWIDTH) << "NPTHD"
              << std::setw(COLWIDTH) << "NCYCLE" << std::endl;
    std::cout << std::string(COLWIDTH * 7, '-') << std::endl;
#else
              << std::setw(COLWIDTH) << "NCYCLE" << std::endl;
    std::cout << std::string(COLWIDTH * 6, '-') << std::endl;
#endif
  }

     if (rank == 0) {
	for (int i = 0; i < Ncycle; i++){
		std::cout << std::setw(COLWIDTH) << N
      		<< std::setw(COLWIDTH) << nbody.sys.size
      		<< std::setw(COLWIDTH) << nbody.sys.t_predv[i]
      		<< std::setw(COLWIDTH) << nbody.sys.t_evalv[i]
      		<< std::setw(COLWIDTH) << nbody.sys.t_corrv[i]
      		<< std::setw(COLWIDTH) << i << std::endl;
	}
    }

  if (rank == 0){
      std::cout << "\nInitial Evaluation Time: " << nbody.sys.t_eval_init << std::endl;
      std::cout << "Initial Energy: " << nbody.sys.initE << std::endl;
      std::cout << "Final Energy: " << nbody.sys.finE << std::endl;
      std::cout << "Q = " << nbody.sys.Q << std::endl;
      std::cout << "Err = " << nbody.sys.Err << std::endl;
  }

  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&totaltime, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::cout << "WALL time: " << totaltime << " seconds" << std::endl;
    std::cout << "Total active cores: " << nbody.sys.num_cores << std::endl;
    std::cout << std::endl;
    double end = timer();
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Time stamp (end): " << end << std::endl;
  }
  
  MPI_Finalize();
  return 0;
}
