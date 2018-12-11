// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_HPP
#define RAPTOR_HPP

// Define types such as int and double sizes
#include "core/types.hpp"

// Data about topology and matrix partitions
#ifndef NO_MPI
    #include "core/partition.hpp"
    #include "core/topology.hpp"
#endif 

// Matrix and vector classes
#include "core/matrix.hpp"
#include "core/vector.hpp"
#ifndef NO_MPI
    #include "core/par_matrix.hpp"
    #include "core/par_vector.hpp"
#endif 

// Communication classes
#ifndef NO_MPI
    #include "core/comm_data.hpp"
    #include "core/comm_pkg.hpp"
#endif

// Stencil and diffusion classes
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/random.hpp"
#ifndef NO_MPI
    #include "gallery/par_stencil.hpp"
    #include "gallery/par_random.hpp"
#endif

// Matrix IO
#include "gallery/matrix_IO.hpp"
#include "gallery/matrix_market.hpp"
#ifndef NO_MPI
    #include "gallery/par_matrix_IO.hpp"
    #include "gallery/par_matrix_market.hpp"
#endif

// Matrix external gallery
#ifdef USING_HYPRE
    #include "gallery/external/hypre_wrapper.hpp"
#endif
#ifdef USING_MFEM
    #include "gallery/external/mfem_wrapper.hpp"
#endif

// RugeStuben classes
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "ruge_stuben/ruge_stuben_solver.hpp"
#ifndef NO_MPI
    #include "ruge_stuben/par_cf_splitting.hpp"
    #include "ruge_stuben/par_interpolation.hpp"
    #include "ruge_stuben/par_ruge_stuben_solver.hpp"
#endif

// SmoothedAgg classes
#include "aggregation/mis.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/candidates.hpp"
#include "aggregation/prolongation.hpp"
#include "aggregation/smoothed_aggregation_solver.hpp"
#ifndef NO_MPI
    #include "aggregation/par_mis.hpp"
    #include "aggregation/par_aggregate.hpp"
    #include "aggregation/par_candidates.hpp"
    #include "aggregation/par_prolongation.hpp"
    #include "aggregation/par_smoothed_aggregation_solver.hpp"
#endif

// AMG multilevel classes
#include "multilevel/multilevel.hpp"
#include "multilevel/level.hpp"
#ifndef NO_MPI
    #include "multilevel/par_multilevel.hpp"
    #include "multilevel/par_level.hpp"
#endif 

// Krylov methods
#include "krylov/cg.hpp"
#include "krylov/par_cg.hpp"
#include "krylov/bicgstab.hpp"
#include "krylov/par_bicgstab.hpp"

// Relaxation methods
#include "util/linalg/relax.hpp"
#ifndef NO_MPI
    #include "util/linalg/par_relax.hpp"
#endif

// Repartitioning matrix methods
#ifndef NO_MPI
#include "util/linalg/repartition.hpp"
#endif
#ifdef USING_PTSCOTCH
    #include "util/linalg/external/ptscotch_wrapper.hpp"
#endif
#ifdef USING_PARMETIS
    #include "util/linalg/external/parmetis_wrapper.hpp"
#endif

// Preconditioning Methods
#ifndef NO_MPI
    #include "util/linalg/par_diag_scale.hpp"
#endif


#endif

