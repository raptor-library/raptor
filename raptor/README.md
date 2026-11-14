# Source layout

To use the library, include `raptor.hpp`.

| Directory | Content |
|---|---|
| `core/` | Basic types (`types.hpp`), vectors (`Vector`, `ParVector`), matrices (`Matrix`, `ParMatrix` and the COO/CSR/CSC and block BCOO/BSR/BSC formats), row partitions (`partition.hpp`), node topology (`topology.hpp`), and communication packages (`comm_pkg.hpp`, `comm_data.hpp`, including node-aware/TAP communication) |
| `aggregation/` | Smoothed aggregation AMG: aggregation, MIS(2), candidate fitting (tentative prolongation), Jacobi prolongation smoothing, and the `SmoothedAggregationSolver` / `ParSmoothedAggregationSolver` hierarchies |
| `ruge_stuben/` | Classical AMG: C/F splitting (RS, CLJP, Falgout, PMIS, HMIS), interpolation, the `RugeStubenSolver` / `ParRugeStubenSolver` hierarchies, and the AIR solver (`ParAIRSolver`) |
| `multilevel/` | Base multilevel hierarchy and level classes (`Multilevel`, `ParMultilevel`, `Level`, `ParLevel`) and sparsification |
| `krylov/` | CG, BiCGStab, and their preconditioned and partial-inner-product variants |
| `gallery/` | Test problems (stencils, 27-point Laplacian, rotated anisotropic diffusion, random matrices) and matrix I/O (Matrix Market and binary formats) |
| `util/linalg/` | SpMV, SpGEMM, matrix addition, relaxation, diagonal scaling, spectral radius estimation, repartitioning, and LAPACK wrappers; `util/linalg/external/` holds the ParMETIS and PT-Scotch wrappers |
| `util/` | Matrix file writer for CSR, BSR, ParCSR, and ParBSR matrices (`writer.hpp`) |
| `external/` | Optional wrappers for hypre, MFEM (with MFEM-based test problems in `external/mfem/`), PETSc, and Trilinos MueLu |
| `profiling/` | Communication profiling |
| `tests/` | Strength-of-connection tests and shared test comparison helpers |

`strength.cpp` and `par_strength.cpp` (strength of connection) are at the top
level.
