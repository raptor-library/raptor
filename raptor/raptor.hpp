// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_HPP
#define RAPTOR_HPP

// Define types such as int and double sizes
#include "core/types.hpp"

// Data about topology and matrix partitions
#ifdef USING_MPI
    #include "core/partition.hpp"
    #include "core/topology.hpp"
#endif 

// Matrix and vector classes
#include "core/matrix.hpp"
#include "core/vector.hpp"
#ifdef USING_MPI
    #include "core/par_matrix.hpp"
    #include "core/par_vector.hpp"
#endif 

// Communication classes
#ifdef USING_MPI
    #include "core/comm_data.hpp"
    #include "core/comm_pkg.hpp"
#endif

// Stencil and diffusion classes
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/random.hpp"
#ifdef USING_MPI
    #include "gallery/par_stencil.hpp"
    #include "gallery/par_random.hpp"
#endif

// Matrix IO
#include "gallery/matrix_IO.hpp"
#ifdef USING_MPI
    #include "gallery/par_matrix_IO.hpp"
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
#ifdef USING_MPI
    #include "ruge_stuben/par_cf_splitting.hpp"
    #include "ruge_stuben/par_interpolation.hpp"
#endif

// AMG multilevel classes
#include "multilevel/multilevel.hpp"
#include "multilevel/level.hpp"
#ifdef USING_MPI
    #include "multilevel/par_multilevel.hpp"
    #include "multilevel/par_level.hpp"
#endif 

// Relaxation methods
#include "util/linalg/seq/relax.hpp"
#ifdef USING_MPI
    #include "util/linalg/relax.hpp"
#endif

// Repartitioning matrix methods
#ifdef USING_MPI
#include "util/linalg/repartition.hpp"
#endif
#ifdef USING_PTSCOTCH
    #include "util/linalg/external/ptscotch.hpp"
#endif


#endif

