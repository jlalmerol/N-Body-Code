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

## 🛠️ Build instructions

Use a different build directory for every mode and configuration. This keeps
CMake and TT-Metal artifacts from incompatible modes from being mixed.

On the validated Wormhole host, first select GCC 12 and the ULFM Open MPI
installation:

```bash
cd ~/tt-metal/N-Body-Code
export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

The general configure/build pattern is:

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
cmake -S . -B <build-directory> \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=<mode> \
  -DMPI_C_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicc \
  -DMPI_CXX_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicxx

cmake --build <build-directory> --target nbody_app -j8
```

Building only `nbody_app` avoids compiling unrelated TT-Metal utilities.

### Build modes

| Mode | Execution and force implementation |
|---|---|
| `DEFAULT` | CPU direct all-pairs implementation using host `double` arithmetic. |
| `_F32` | CPU direct all-pairs implementation with Float32 force arithmetic and `double` host state. |
| `_GEMM_BASED` | CPU blocked 32×32 implementation using OpenBLAS. Set `OPENBLAS_ROOT` if OpenBLAS is not in a standard path. |
| `_WH_MULTIHOST` | One 1×1 `MeshDevice` per MPI rank. Uses `nb_read.cpp`, `nb_compute.cpp`, and `nb_write.cpp`, compact 32-target × 32-source packing, SFPU pair arithmetic, and a persistent device/program lifecycle. |
| `_WH_GEMM_MULTIHOST` | Experimental path with one 1×1 `MeshDevice` per rank. Uses augmented 32×32 matrices, Tensix matrix products, SFPU inverse-distance operations, on-device source-moment reduction, and a persistent lifecycle. Validate numerical accuracy for the intended input distribution; the augmented squared-distance expression can suffer cancellation for close pairs. |
| `_WH_MULTICHIP` | One MPI process controls a 1×2 local Wormhole mesh using the `nb_*` kernel family. This mode retains its legacy expanded layout and is distinct from MPI multi-host execution. |
| `_WH_MESH` | One MPI process controls a 1×2 `MeshDevice` using `mesh_read.cpp`, `mesh_compute.cpp`, and `mesh_write.cpp`. The current port uses the 32×32 target/source block layout and requires two visible Wormhole devices. |

For detailed kernel equations and dataflow, see
[FORCE_KERNEL_DETAILS.md](FORCE_KERNEL_DETAILS.md). For a mode-level comparison,
see [COMPUTE_KERNEL_REPORT.md](COMPUTE_KERNEL_REPORT.md).

### CPU build examples

Default FP64 CPU:

```bash
cmake -S . -B build-cpu-fp64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=DEFAULT
cmake --build build-cpu-fp64 --target nbody_app -j8
```

Mixed-precision CPU:

```bash
cmake -S . -B build-cpu-f32 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=_F32
cmake --build build-cpu-f32 --target nbody_app -j8
```

OpenBLAS CPU GEMM:

```bash
cmake -S . -B build-cpu-gemm \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=_GEMM_BASED \
  -DOPENBLAS_ROOT=/path/to/openblas
cmake --build build-cpu-gemm --target nbody_app -j8
```

### Wormhole release builds

MULTIHOST:

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
cmake -S . -B build-wh-multihost-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=_WH_MULTIHOST \
  -DNBODY_DEBUG_VALIDATION=OFF \
  -DNBODY_PHASE_TIMERS=OFF \
  -DNBODY_DEVICE_PROFILE=OFF \
  -DNBODY_DEVICE_ROW_REDUCE=OFF \
  -DMPI_C_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicc \
  -DMPI_CXX_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicxx
cmake --build build-wh-multihost-release --target nbody_app -j8
```

GEMM MULTIHOST:

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
cmake -S . -B build-wh-gemm-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBODY_BUILD_MODE=_WH_GEMM_MULTIHOST \
  -DNBODY_DEBUG_VALIDATION=OFF \
  -DNBODY_PHASE_TIMERS=OFF \
  -DNBODY_DEVICE_PROFILE=OFF \
  -DMPI_C_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicc \
  -DMPI_CXX_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicxx
cmake --build build-wh-gemm-release --target nbody_app -j8
```

MULTICHIP and MESH use the same command pattern with, respectively:

```text
-DNBODY_BUILD_MODE=_WH_MULTICHIP  and build directory build-wh-multichip-release
-DNBODY_BUILD_MODE=_WH_MESH       and build directory build-wh-mesh-release
```

Both require two local devices visible to the process.

### Validation builds

Use `RelWithDebInfo` plus `NBODY_DEBUG_VALIDATION=ON`. Do not use full
`CMAKE_BUILD_TYPE=Debug` for the embedded TT-Metal build: this revision can
fail while linking auxiliary Tracy/TBB tools or architecture-specific HAL
objects.

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
cmake -S . -B build-wh-multihost-validation \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNBODY_BUILD_MODE=_WH_MULTIHOST \
  -DNBODY_DEBUG_VALIDATION=ON \
  -DNBODY_PHASE_TIMERS=OFF \
  -DNBODY_DEVICE_PROFILE=OFF \
  -DNBODY_DEVICE_ROW_REDUCE=OFF \
  -DMPI_C_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicc \
  -DMPI_CXX_COMPILER=/opt/openmpi-v5.0.7-ulfm/bin/mpicxx
cmake --build build-wh-multihost-validation --target nbody_app -j8
```

For GEMM validation, change the build directory to
`build-wh-gemm-validation` and the mode to `_WH_GEMM_MULTIHOST`.

The validator performs a double-precision host `O(N²)` calculation after each
force evaluation. Start with `N=1024` or `N=4096`, not a production-size run.

### Build options

| Option | Default | Meaning |
|---|---:|---|
| `USE_OPENMP` | `ON` | Enable OpenMP in host calculations. |
| `NBODY_PHASE_TIMERS` | `OFF` | Print packing, setup, H2D, workload, D2H, and host-postprocessing time for the last force call. |
| `NBODY_DEBUG_VALIDATION` | `OFF` | Compare device force results with a robust double-precision host reference and print error norms. |
| `NBODY_DEVICE_PROFILE` | `OFF` | Compile device profiler events into the N-body compute kernels. |
| `NBODY_DEVICE_ROW_REDUCE` | `OFF` | Experimental on-device row reduction for `_WH_MULTIHOST` and `_WH_MESH`; full-size output buffers are retained. |

## 🚀 Run instructions

All modes accept:

```text
--N <number-of-bodies>
--Ncycle <number-of-integration-cycles>
```

CPU modes still initialize MPI and can be run with one process:

```bash
/opt/openmpi-v5.0.7-ulfm/bin/mpirun -np 1 \
  ./build-cpu-fp64/nbody_app --Ncycle 3 --N 4096
```

### `tt-run`: one Wormhole per MPI rank

The rank-binding files remain YAML, but their `mesh_graph_desc_path` values
refer to MGD 2.0 Protobuf text files (`.textproto`). Each MPI rank owns a
separate 1×1 mesh (`mesh_host_rank: 0`); force-result exchange between ranks is
performed by MPI. The supplied descriptors declare the channel configuration
required by MGD 2.0.

Select the ULFM Open MPI installation before invoking `tt-run`:

```bash
export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

For a single-host run, use `--bare`.

Single-rank MULTIHOST:

```bash
tt-run --bare \
  --rank-binding yamls/1_wh_rank_bindings.yaml \
  --mpi-args "--bind-to core" \
  ./build-wh-multihost-release/nbody_app \
  --Ncycle 3 --N 102400
```

Single-rank GEMM MULTIHOST:

```bash
tt-run --bare \
  --rank-binding yamls/1_wh_rank_bindings.yaml \
  --mpi-args "--bind-to core" \
  ./build-wh-gemm-release/nbody_app \
  --Ncycle 3 --N 102400
```

Two-rank execution uses `yamls/2_wh_rank_bindings.yaml`; four ranks use
`yamls/4_wh_rank_bindings.yaml`. Each rank is bound to one Wormhole device:

```bash
tt-run \
  --rank-binding yamls/2_wh_rank_bindings.yaml \
  --mpi-args "--bind-to core" \
  ./build-wh-multihost-release/nbody_app \
  --Ncycle 3 --N 102400
```

For `_WH_GEMM_MULTIHOST`, each rank's target interval must begin on a
32-particle boundary. Choose `N` and rank count so the particles per rank are a
multiple of 32.

### Local 1×2 modes

`_WH_MULTICHIP` and `_WH_MESH` each create a 1×2 local `MeshDevice` inside one
MPI process. They require a host exposing two devices to that process. They are
not the same topology as the examples above, where each MPI rank sees one
device. Use the rank-binding/mesh descriptor appropriate to the installed
two-device system.

### Performance and validation output

See [FORCE_KERNEL_DETAILS.md](FORCE_KERNEL_DETAILS.md) for a detailed account
of host packing, 32×32 all-pairs blocks, device calculations, accumulation,
reduction, and output reconstruction in `_WH_MULTIHOST` and
`_WH_GEMM_MULTIHOST`.

With `NBODY_PHASE_TIMERS=ON`, Wormhole builds report the last force call as:

```text
Last force phases (s): pack=... setup=... H2D=... workload=... D2H=... postprocess=...
```

This output is independent of Debug validation. `NBODY_PHASE_TIMERS` defaults
to `OFF`. When explicitly enabled, the final input write and workload submission block to separate H2D,
workload, and D2H measurements. With `-DNBODY_PHASE_TIMERS=OFF`, all input
writes and workload submissions in the SFPU and GEMM mesh paths request
non-blocking execution on the same command queue. Queue ordering preserves
write-before-execute dependencies; blocking result reads wait before the CPU
consumes results and before host input vectors are destroyed. Untimed builds
do not report enqueue latency as device kernel execution time.

These flags request asynchronous submission; actual overlap depends on the
SDK dispatch backend. In particular, slow-dispatch buffer transfers can still
execute synchronously inside TT-Metal. The force calculation itself still
waits for its results before returning.

Device profiler events are disabled by default. Enable them only for profiling
with `-DNBODY_DEVICE_PROFILE=ON`.

`NBODY_DEVICE_ROW_REDUCE=ON` enables the experimental SFPU row reduction for
`_WH_MULTIHOST` and `_WH_MESH`. It deliberately retains full-tile output so the
device reduction can be validated independently before compact buffers and a
compact writer are introduced. Compare a small run against the default build
before benchmarking it.

---

## Tracy profiling

Wormhole host paths use `ZoneScopedN` for `NBody force calculation`,
`NBody H2D`, `NBody workload`, and `NBody D2H`. TT-Metal supplies the
Tracy client; configure the device build with `-DENABLE_TRACY=ON`.
CPU-only builds do not depend on Tracy.

Use `-DNBODY_PHASE_TIMERS=ON` to fence transfers and workload execution
when measuring separate elapsed phases. With phase timers off, the H2D
and workload scopes measure API submission; D2H can include waiting for
outstanding device work. Profiling fences change overlap and throughput.
The first force call also includes device setup and kernel compilation.

All three device kernel families have `DeviceZoneScopedN` scopes named
`NBody device read`, `NBody device compute`, and `NBody device write`.
These include circular-buffer/NoC waits (compute scopes exclude hardware startup);
reader, compute, and writer can overlap, so their times should not be added.
Set `TT_METAL_DEVICE_PROFILER=1` at runtime to enable TT-Metal device events.
`NBODY_DEVICE_PROFILE=ON` additionally enables the existing per-tile SFPU
compute events. Device events use TT-Metal's device profiler and its
postprocessing workflow; a plain Tracy capture records the host scopes.

For the simulator container, start Colima and the container if needed:

```bash
colima start
docker start gravityx-tt
docker exec -it gravityx-tt bash
```

Inside the container, build the instrumented sources in
`/opt/tt-metal/N-Body-Code`. In one container terminal, start the matching
SDK capture tool before launching the application:

```bash
/opt/tt-metal/N-Body-Code/build-wh-multihost-release/tools/profiler/bin/tracy-capture \
  -o /tmp/nbody.tracy -f
```

In another container terminal:

```bash
cd /opt/tt-metal
export TT_METAL_HOME=/opt/tt-metal
export TT_METAL_SIMULATOR=/opt/mylibrary/ttsim/1.10.1/lib/libttsim_wh_aarch64.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export OMPI_MCA_ess_singleton_isolated=1
OMP_NUM_THREADS=2 ./N-Body-Code/build-wh-multihost-release/nbody_app --N 32 --Ncycle 1
```

Copy `/tmp/nbody.tracy` out with `docker cp` and open it in a matching
Tracy viewer. Simulator runs validate instrumentation and execution;
measure real kernel and transfer performance on hardware.

---

## ⚙️ Software and Environment Specifications
The following software and hardware environment was used for development, testing, and validation.
Collected from tt-smi snapshot.

| Component | Version / Details |
| --------- | ----------------- |
| **Operating system** | `Ubuntu 22.04.5 LTS` |
| **Kernel** | `5.15.0-185-generic` |
| **Platform** | `x86_64` |
| **Host memory** | `30.95 GB` |
| **TT-KMD driver** | `2.8.0` |
| **tt-smi** | `5.3.0` |
| **pyluwen** | `0.8.5` |
| **TT-UMD** | `0.9.5` |
| **Python** | `3.10.12` |
| **MPI** | `Open MPI 5.0.7rc2` (`/opt/openmpi-v5.0.7-ulfm`) |
| **Device** | `Tenstorrent n150 L`, PCIe `0000:01:00.0`, Gen `4` x`16`, DRAM `10G` |
| **Firmware bundle** | `19.6.0.0` |
| **TT flash** | `0.3.6.2` |
| **CM firmware** | `2.39.0.0` (date `2026-02-11`) |
| **Ethernet firmware** | `7.5.0` |
| **DM bootloader firmware** | `129.2.0.0` |
| **DM application firmware** | `5.12.0.0` |
| **TT-Metal** | `v0.74.0-dev20260624-31-g416f59fe33b` |

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
