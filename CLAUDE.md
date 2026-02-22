# CLAUDE.md — DREAM Codebase Guide

## Project Overview

**DREAM** (Disruption Runaway Electron Analysis Model) is a physics simulation framework for studying relativistic runaway electrons in tokamak fusion devices. It solves coupled nonlinear PDEs describing time evolution of tokamak plasma, with emphasis on runaway electron generation and dynamics.

- **Official paper**: [doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098)
- **Online documentation**: https://ft.nephy.chalmers.se/dream
- **Architecture**: C++17 core solver + Python 3 interface and tooling

---

## Repository Structure

```
MyDREAM/
├── CMakeLists.txt           # Root build configuration
├── testDREAM.sh             # Full test runner (run before PRs)
├── GIT.md                   # Git workflow guidelines
├── CONTRIBUTORS.md          # Contributor list
├── DEVELOPERTIPS.txt        # Developer notes (CMake flags, Valgrind, GDB)
├── src/                     # C++ source code (DREAM library)
│   ├── Equations/
│   │   ├── Fluid/           # Fluid-model equations (~59 files)
│   │   ├── Kinetic/         # Kinetic equations (~15 files)
│   │   └── Scalar/          # Scalar equations
│   ├── EquationSystem/      # Equation system management
│   ├── Settings/            # Settings loading from HDF5
│   ├── Solver/              # Linear/nonlinear solvers (PETSc-backed)
│   ├── TimeStepper/         # Time stepping (constant, adaptive, ionization)
│   ├── Atomics/             # ADAS/AMJUEL/NIST atomic data loaders
│   ├── IonHandler.cpp       # Ion species management
│   ├── OtherQuantityHandler.cpp  # Derived quantity calculations
│   ├── Simulation.cpp       # Top-level simulation orchestrator
│   └── EqsysInitializer.cpp # Equation system construction
├── fvm/                     # Finite Volume Method library
│   ├── Grid/                # Radial and momentum grids, bounce averaging
│   ├── Equation/            # FVM equation infrastructure
│   └── Solvers/             # FVM-level solvers
├── include/
│   ├── DREAM/               # DREAM C++ headers
│   └── FVM/                 # FVM C++ headers
├── iface/                   # C++ entry point (dreami executable)
│   └── Main.cpp             # main(); argument parsing, simulation lifecycle
├── dreampyface/             # Python C extension (optional PyFace build)
├── py/
│   ├── DREAM/               # Main Python package
│   │   ├── DREAMSettings.py # Top-level settings object
│   │   ├── DREAMOutput.py   # Output data reader
│   │   ├── runiface.py      # runiface() helper to launch dreami
│   │   ├── Settings/        # Per-subsystem settings modules
│   │   │   ├── Equations/   # Equation-specific settings (19 modules)
│   │   │   ├── RadialGrid.py, MomentumGrid.py, Solver.py, TimeStepper.py, …
│   │   └── Output/          # Output quantity accessors (~50 modules)
│   ├── Theater/             # Qt5 GUI for real-time visualization
│   └── cli/                 # CLI tools (cli.py, debug.py)
├── tests/
│   ├── cxx/                 # C++ unit tests
│   └── physics/             # Physics validation tests (Python)
├── examples/                # ~45 simulation scenarios
├── extern/softlib/          # Git submodule: hoppe93/softlib
├── setup/                   # Cluster-specific build scripts
├── doc/
│   ├── sphinx/              # Sphinx HTML/PDF documentation sources
│   └── notes/               # LaTeX physics/math notes
├── tools/                   # Utility scripts (ADAS data gen, debugging)
└── cmake/                   # Custom CMake modules
```

---

## Build System

### Requirements

| Dependency | Version | Notes |
|---|---|---|
| CMake | >= 3.12 | Build system |
| GCC / Clang / Intel | C++17 capable | GCC >= 7.0 recommended |
| GNU Scientific Library | >= 2.4 | `libgsl-dev` |
| HDF5 | any | `libhdf5-dev` |
| PETSc | any | Usually manual install (see README) |
| Python 3 | any | Required for ADAS data gen and Python interface |
| MPI (MPICH/OpenMPI) | any | May be required by PETSc |

**Python packages**: `h5py`, `matplotlib`, `numpy`, `scipy`, `packaging`

### Standard Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

This produces `build/iface/dreami` — the main executable.

### CMake Options

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug          # Debug symbols (default: Release)
cmake .. -DDREAM_BUILD_TESTS=ON            # Build C++ tests (default: ON)
cmake .. -DDREAM_BUILD_PYFACE=ON           # Build Python C extension (default: OFF)
cmake .. -DPETSC_DIR=/path/to/petsc -DPETSC_ARCH=linux-c-opt
```

### Out-of-source builds are mandatory

The CMakeLists.txt explicitly rejects in-source builds. Always build in a separate `build/` directory.

### Cluster builds

Pre-made scripts are in `setup/build.<cluster>.sh` and `setup/environment.<cluster>.sh` for HPC environments (cam, cori, engaging, lac10, lotta, mahti, spcsrv26, tok).

---

## Running DREAM

### Workflow

1. **Write a Python settings script** that creates a `DREAMSettings` object and saves it to HDF5:

```python
import sys
sys.path.append('/path/to/MyDREAM/py')
from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Solver as Solver

ds = DREAMSettings()

# Configure physics
ds.eqsys.E_field.setPrescribedData(efield=..., times=..., radius=...)
ds.eqsys.T_cold.setPrescribedData(temperature=..., times=..., radius=...)
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=1e20)

# Configure grids
ds.radialgrid.setB0(5)
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setNr(10)
ds.hottailgrid.setNxi(30)
ds.hottailgrid.setNp(100)

# Configure numerics
ds.solver.setType(Solver.LINEAR_IMPLICIT)
ds.timestep.setTmax(1e-6)
ds.timestep.setNt(4)

# Either save and run manually:
ds.save('dream_settings.h5')
# Then: ./build/iface/dreami dream_settings.h5

# Or run directly via Python:
do = runiface(ds, 'output.h5', quiet=False)
```

2. **Run the executable**:

```bash
./build/iface/dreami dream_settings.h5
# Output written to output.h5 by default
```

3. **Load and analyze output**:

```python
from DREAM.DREAMOutput import DREAMOutput
do = DREAMOutput('output.h5')
# Access quantities: do.eqsys.E_field, do.eqsys.n_re, do.eqsys.T_cold, etc.
```

### Environment Variable

Set `DREAMPATH` to point to the DREAM root directory so `runiface` can locate the `dreami` executable:

```bash
export DREAMPATH=/path/to/MyDREAM
```

### dreami Command-Line Options

```
dreami INPUT        Run simulation from settings file
dreami -a INPUT     Print ADAS element list
dreami -l           List all available settings
dreami -s INPUT     Suppress splash screen
dreami -h           Print help
```

---

## Testing

### Run all tests (required before PRs)

```bash
./testDREAM.sh
```

This script:
1. Rebuilds with `make -j$(nproc)` from the `build/` directory
2. Runs C++ unit tests: `build/tests/cxx/dreamtests all`
3. Runs physics/Python tests: `tests/physics/runtests.py all`

### C++ tests only

```bash
build/tests/cxx/dreamtests all
# Or specific test:
build/tests/cxx/dreamtests <test_name>
```

### Physics tests only

```bash
tests/physics/runtests.py all
# Or specific test:
tests/physics/runtests.py <test_name>
```

### Available physics test suites

- `DREAM_avalanche` — Avalanche generation
- `amperefaraday` — Faraday/Ampere equations
- `code_conductivity` — Conductivity calculations
- `code_runaway` — Runaway electron physics
- `code_synchrotron` — Synchrotron radiation
- `numericmag` — Numerical magnetic field
- `runiface_parallel` — Parallel run infrastructure
- `trapping_conductivity` — Trapping effects on conductivity
- `ts_adaptive` — Adaptive time stepping

---

## Debugging and Profiling

From `DEVELOPERTIPS.txt`:

### GDB

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
gdb build/iface/dreami
# Useful commands:
(gdb) catch throw          # Break on C++ exception thrown
(gdb) catch catch          # Break on C++ exception caught
```

In debug builds on Linux, floating-point exceptions (`NaN`, divide-by-zero, overflow) are enabled and will throw, making them easier to catch.

### Valgrind

```bash
# Memory leak detection
valgrind --leak-check=yes --track-origins=yes --num-callers=30 \
    --log-file=valgrind.log build/iface/dreami settings.h5

# Profiling (visualize with kcachegrind)
valgrind --tool=callgrind build/iface/dreami settings.h5

# Heap profiling (visualize with massif-visualizer)
valgrind --tool=massif build/iface/dreami settings.h5
```

---

## Code Architecture

### Execution Flow

```
dreami [settings.h5]
  └── main()                          (iface/Main.cpp)
       ├── dream_initialize()         (PETSc/DREAM init)
       ├── SimulationGenerator::CreateSettings()
       ├── SettingsSFile::LoadSettings(settings, filename)
       ├── SimulationGenerator::ProcessSettings(settings)
       │    └── Builds EquationSystem with all equations
       ├── sim->Run()
       │    └── eqsys->Solve()        (time loop with PETSc solver)
       ├── sim->Save()                (write output.h5)
       └── dream_finalize()
```

### Key C++ Classes

| Class | File | Purpose |
|---|---|---|
| `Simulation` | `src/Simulation.cpp` | Top-level container; owns EquationSystem and output generator |
| `EquationSystem` | `src/EquationSystem/` | Manages all unknowns, equations, time stepping |
| `EqsysInitializer` | `src/EqsysInitializer.cpp` | Constructs equations from settings |
| `IonHandler` | `src/IonHandler.cpp` | Tracks ion species and charge states |
| `OtherQuantityHandler` | `src/OtherQuantityHandler.cpp` | Computes derived/diagnostic quantities |
| `FVM::Grid` | `fvm/Grid/Grid.cpp` | Radial + momentum grid infrastructure |
| `FVM::UnknownQuantityHandler` | `fvm/UnknownQuantityHandler.cpp` | Registry of all unknown quantities |
| `Solver` (variants) | `src/Solver/` | `SolverLinearlyImplicit`, `SolverNonLinear` |
| `TimeStepper` (variants) | `src/TimeStepper/` | Constant, adaptive, ionization-based |

### FVM vs DREAM separation

- **`fvm/`**: General finite-volume infrastructure (grids, matrices, interpolation, base solvers). Does not know about plasma physics.
- **`src/`**: DREAM-specific physics built on top of FVM (equations, collision models, atomic data, etc.).

### Adding a new equation

1. Add a `.cpp` implementation in `src/Equations/Fluid/` or `src/Equations/Kinetic/`
2. Add the corresponding `.hpp` header in `include/DREAM/Equations/`
3. Register it in `src/EqsysInitializer.cpp` (or relevant `Settings/` loader)
4. Add Python settings support in `py/DREAM/Settings/Equations/`
5. Add output support in `py/DREAM/Output/` if needed
6. Write a physics test in `tests/physics/`

---

## Python Interface

### Package Location

The Python package is at `py/DREAM/`. To use it without installing:

```python
import sys
sys.path.append('/path/to/MyDREAM/py')
import DREAM
```

### Key Modules

| Module | Purpose |
|---|---|
| `DREAM.DREAMSettings` | Root settings object |
| `DREAM.DREAMOutput` | Output data reader |
| `DREAM.runiface` | `runiface()` / `runiface_parallel()` launcher |
| `DREAM.DREAMIO` | Low-level HDF5 I/O |
| `DREAM.ConvergenceScan` | Automated parameter sweeps |
| `DREAM.Settings.EquationSystem` | Equation configuration |
| `DREAM.Settings.RadialGrid` | Radial grid setup |
| `DREAM.Settings.MomentumGrid` | Momentum grid setup |
| `DREAM.Settings.Solver` | Solver type and options |
| `DREAM.Settings.TimeStepper` | Time stepping options |
| `DREAM.Settings.Transport*` | Transport settings |

### Theater (GUI)

A Qt5-based visualization tool for exploring output and running simulations interactively:

```bash
python3 py/Theater/run.py
# or
python3 py/gui.py
```

Requires PyQt5 and matplotlib.

---

## Git Workflow

From `GIT.md` — follow these rules strictly:

- **Use `git add -u`** or add specific files explicitly. Never use `git add -A` or `git add .`
- **Work on feature branches**: one branch per feature/fix
- **Delete merged branches** to avoid clutter
- **No large/binary files** (>1 MiB). HDF5 output files are in `.gitignore`
- **No sensitive data** in the repo directory at any time
- Submit changes via **pull requests** to `master`; reviewed by the DREAM Developer Council

### CI/CD

Two GitHub Actions workflows (`.github/workflows/`):
- `buildAndTest.yml`: GCC build + full test suite on `push`/`PR` to `master`
- `LLVMbuild.yml`: Clang build on `push`/`PR` to `master`

Both use Ubuntu latest, install PETSc via pip, and run `./testDREAM.sh`.

---

## Conventions

### C++

- **Standard**: C++17
- **Warning level**: `-Wall -Wextra -Wpedantic` for all supported compilers
- **Optimization**: `-O3` in Release mode (default)
- **Namespaces**: `DREAM::` for DREAM code, `DREAM::FVM::` for FVM code
- **Exceptions**: Use `DREAMException` (from `include/DREAM/DREAMException.hpp`) and `FVMException`
- **Headers**: `.hpp` extension for C++ headers, `.h` for C-compatible headers
- **Template implementations**: `.tcc` files (e.g., `IO.tcc`)

### Python

- No enforced style tool in the repo, but follow the existing module patterns
- Settings classes generally mirror the C++ settings structure
- Output classes wrap HDF5 group data into Python objects with numpy arrays
- Use `np.ndarray` for grid/field data throughout

### File I/O

- Settings files: HDF5 (`.h5`), written by `DREAMSettings.save()`
- Output files: HDF5 (`.h5`), written by the C++ `OutputGeneratorSFile`
- HDF5 files are in `.gitignore` — never commit them

---

## Common Tasks

### Rebuild after source changes

```bash
cd build && make -j$(nproc)
```

### Run a specific example

```bash
cd examples/basic
python3 basic.py           # generates dream_settings.h5
../../build/iface/dreami dream_settings.h5
python3 plotSolution.py    # visualize output
```

### Check available settings

```bash
build/iface/dreami -l
```

### Count lines of code

```bash
./cloc.sh
```

### Run parallel simulations

```python
from DREAM import runiface_parallel
outputs = runiface_parallel(settings_list, outfile_list, njobs=4)
```

---

## External Dependencies Details

### softlib (submodule)

Located at `extern/softlib/`. Provides `SFile` (unified HDF5/SDF file I/O) and `SOFTLibException`. Initialized automatically during CMake configuration via `git submodule update --init --recursive`.

### PETSc

Critical for all linear and nonlinear solving. Set environment variables:

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=linux-c-opt
```

Or pass to CMake: `cmake .. -DPETSC_DIR=... -DPETSC_ARCH=...`

### ADAS / AMJUEL / NIST

Atomic data used for collision rates and ionization. Data is compiled into the binary from source files in `src/Atomics/`, `src/MeanExcitationEnergyData/`, etc. Generation tools are in `tools/ADAS/`.
