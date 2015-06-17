main structure

- types.hpp
  - defines the main typedefs
    - int\_t: integer type (e.g. int)
    - data\_t: data type (e.g. double)

- mulilevel.hpp
  - defines the multilevel hierarchy with A, P, methods

- aggregation
  - main SA routines
    - aggregation
    - aggregate
    - smooth

- core
  - Vector, Matrix, interface, and partitioning headers
    - sequential containers defined by Eigen3
    - vector.hpp: defines ParVector
    - matrix.hpp: defines ParCSRMatrix and SeqCSRMatrix
    - interface.hpp: defines PETSc and Dolfin interfaces
    - partition.hpp: describes the partitioning

- util
  - linalg: ParCSR and ParVector linear algebra
  - utils: helper routines for funny operations
