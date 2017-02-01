// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TYPES_HPP
#define RAPTOR_CORE_TYPES_HPP

#include "mpi.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//TODO -- should be std::numeric_limits<data_t>::epsilon ...
//#define zero_tol DBL_EPSILON
#define zero_tol 1e-16
#define MPI_INDEX_T MPI_INT
#define MPI_DATA_T MPI_DOUBLE
#define WITH_OMP FALSE 
#define OMP_TPP 16

namespace raptor
{
    using data_t = double;
    using index_t = int;
    enum format_t {CSR, CSC, COO};

    struct coo_data
    {
        index_t row;
        index_t col;
        data_t value;
    };

    struct csr_data
    {
        index_t col;
        data_t value;
    };
}
    static void create_coo_type(MPI_Datatype* coo_type)
    {
        int blocks[2] = {2, 1};
        MPI_Datatype types[2] = {MPI_INDEX_T, MPI_DATA_T};
        MPI_Aint displacements[2];
        MPI_Aint intex;
        MPI_Type_extent(MPI_INDEX_T, &intex);
        displacements[0] = static_cast<MPI_Aint>(0);
        displacements[1] = 2*intex;
        MPI_Type_struct(2, blocks, displacements, types, coo_type);
    }

    static void create_csr_type(MPI_Datatype* csr_type)
    {
        MPI_Datatype oldtypes[2];
        int blockcounts[2];
        MPI_Aint offsets[2], extent, lb;

        offsets[0] = 0;
        oldtypes[0] = MPI_INT;
        blockcounts[0] = 1;
        
        // TODO -- this doesn't work!  Give 4 but offset is actually 8...
        //MPI_Type_get_extent(MPI_INT, &lb, &extent);
        //offsets[1] = extent;
        offsets[1] = sizeof(raptor::csr_data) - sizeof(double);
        oldtypes[1] = MPI_DOUBLE;
        blockcounts[1] = 1;

        MPI_Type_struct(2, blockcounts, offsets, oldtypes, csr_type);
    }

#endif
