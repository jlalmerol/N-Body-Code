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
mpirun -n 4 ./nbody_app # Uses the default: 20480 bodies
```