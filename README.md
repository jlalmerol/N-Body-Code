# N-Body-Simulation

A toy code for n-body simulation.

To compile N-body code for WH (with DEBUG on):

```
mkdir build && cd build
cmake -DNBODY_BUILD_MODE=_WH ..
make
```

To run the code:
```
./nbody_app [num_of_bodies]
```
Example:
```
./nbody_app        # Uses the default: 20480 bodies
./nbody_app 10000  # Simulates 10,000 bodies
```

Note: 

This version of the code currently supports a single Wormhole (WH) device across multiple Tensix cores using TT-Metal (v0.60.1). Multi-device support is under development.