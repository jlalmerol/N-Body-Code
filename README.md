# N-Body Simulation

A **toy N-body simulation code** designed for performance exploration and validation on **Tensix cores** using the **TT-Metal** framework.  
This implementation models gravitational interactions among multiple bodies and is optimized for multi-device Wormhole (WH) execution.

---

## 🧩 Features

- Scalable, parallelized N-body simulation on **Tensix cores**
- Multi–**Wormhole (WH)** device execution using **TT-Metal**
- **MPI-based parallelism** for distributed multi-card workloads
- Modular **CMake** build system with configurable debug and precision modes
- Designed for research, benchmarking, and hardware validation

---

## 🛠️ Build Instructions

### 1. Configure the project
```bash
mkdir build && cd build
cmake -DNBODY_BUILD_MODE=<BUILD_MODE> ..
```
Note: Add `-DCMAKE_BUILD_TYPE=Debug` to enable debug prints.
### 2. Compile
```
make
```

---

## 📦 Build Modes

| Mode  | Description                                                                                |
| ----- | ------------------------------------------------------------------------------------------ |
| `None` | Builds the **default** CPU-only implementation in full FP64 precision. |
| `_F32` | Builds the **CPU**-only mixed-precision version (FP32 and FP64). |
| `_WH_MULTIHOST` | Builds the simulation for multiple **Wormhole (WH)** devices using only **L**-chip. |
| `_WH_GEMM_MULTIHOST` | Builds the blocked 32x32 GEMM force kernel for one **Wormhole** per MPI rank. |
| `_WH_MULTICHIP` | Builds the simulation for multiple **Wormhole (WH)** devices using both **L**-chip and **R**-chip. |
| `_WH_MESH` | Builds the simulation for a **mesh configuration** of multiple **Wormhole (WH)** devices. |

For a detailed comparison of the Wormhole data layouts, compute engines,
reductions, device lifecycles, and current benchmark results, see
[COMPUTE_KERNEL_REPORT.md](COMPUTE_KERNEL_REPORT.md).

Note:
Wormhole cards are currently configured in isolation.
Mesh topology per MPI process follows a (1, 2) layout.

The GEMM multi-host mode expects each MPI rank's target range to start on a
32-particle boundary. Choose `N` so that the per-rank particle count is a
multiple of 32. Configure it from a TT-Metal source tree containing this
repository as `N-Body-Code`:

```bash
cmake -S N-Body-Code -B N-Body-Code/build \
  -DNBODY_BUILD_MODE=_WH_GEMM_MULTIHOST
cmake --build N-Body-Code/build -j
```

---

## 🚀 Run Instructions

To execute the N-body simulation using **TT-Metal** and **MPI**, use the `tt-run` launcher with appropriate rank bindings and runtime arguments.

### Syntax
```bash
tt-run --rank-binding [rank-binding-yaml] --mpi-args "--bind-to core" ./build/nbody_app --Ncycle [time-cycles] --N [number-of-bodies]
```
Example:  
```
tt-run --rank-binding yamls/2_wh_rank_bindings.yaml --mpi-args "--bind-to core" ./build/nbody_app --Ncycle 3 --N 102400
```

Argument descriptions:

- --rank-binding [rank-binding-yaml]: Path to the YAML file defining rank-to-device bindings.

- --mpi-args "--bind-to core": Ensures MPI processes are pinned to CPU cores for optimal performance.

- --Ncycle [time-cycles]: Number of simulation time cycles to run.

- --N [number-of-bodies]: Number of bodies to simulate.

Note:
The simulation runs with MPI, where each MPI process is mapped to a single Wormhole (WH) card.
This configuration enables multi-device execution across multiple WH cards through TT-Metal, achieving distributed parallelism across hardware resources.

---

## ⚙️ Software and Environment Specifications
The following software environment was used for development, testing, and validation:

| Component                       | Version / Commit                            |
| ------------------------------- | ------------------------------------------- |
| **TT-Metal**                    | `v0.62.0-rc33-281-gef93c54e01`              |
| **TT-Metal submodule (tt_llk)** | `e95fdff39539fdafc96c84672c7f09b5ee38eff3`  |
| **TT-Metal submodule (umd)**    | `d5196f5ee90f0380848a19e199a68e8dab883eb6`  |
| **MPI**                         | `OpenMPI v5.0.7-ulfm`                       |
| **C++ Compiler**                | `g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)` |
| **Python**                      | `3.12.3`                                    |

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
