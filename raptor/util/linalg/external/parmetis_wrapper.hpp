// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_GALLERY_PARMETIS_HPP
#define RAPTOR_GALLERY_PARMETIS_HPP

#include "raptor.hpp"
#include "parmetis.h"

using namespace raptor;

int* parmetis_partition(ParCSRMatrix* A)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    int start, end;
    int col, global_col;

    // ParMetis Partitioner Variables
    RAPtor_MPI_Comm comm = RAPtor_MPI_COMM_WORLD;
    
    // How vertices of graph are distributed among processes;
    // Array size num_procs+1
    // Range of vertices local to each processor
    int* vtxdist = A->partition->first_cols.data();

    // Local adjacency structure
    aligned_vector<int> xadj(A->local_num_rows+1);
    aligned_vector<int> adjncy(A->local_nnz);
    xadj[0] = 0;
    int nnz = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            global_col = A->on_proc_column_map[col];
            adjncy[nnz++] = global_col;
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            global_col = A->off_proc_column_map[col];
            adjncy[nnz++] = global_col;
        }

        xadj[i+1] = nnz;
    }

    // Weights of vertices and edges
    int* vwgt = NULL;
    int* adjwgt = NULL;

    // Is the graph weighted?
    // 0 - No weighting
    // 1 - Edges only
    // 2 - Vertices only
    // 3 - Both edges and vertices
    int wgtflag = 0;

    // Numbering scheme
    // 0 - Cstyle
    // 1 - Fortran
    int numflag = 0;

    // Number of weights that each vertex has;
    int ncon = 1;

    // Number of sub-domains desired;
    int nparts = num_procs;

    // Fraction of vertex weight distributed to each subdomain
    // Array size ncon x nparts
    // For balanced sub-domains, each part gets 1/nparts
    aligned_vector<float> tpwgts(nparts, 1.0/nparts);

    // Imbalance tolerance for each vertex weight
    // Array size ncon 
    // Perfect balance: 1
    // Perfect imblance: nparts
    // Recommended: 1.05
    aligned_vector<float> ubvec(1, 1.05);

    // Additional Options:
    // Options[0] = 0 (default values) or 1 (specify options[1], options[2])
    // Options[1]: levels of info to be returned (0-default, 1-timing info)
    // Options[2]: random number seed for routine
    aligned_vector<int> options(3, 0);

    // Return value: Number of edges that are cut by partitioning
    int edgecut;

    // Return value: Array (size of local_num_rows) of partition for each row
    int* part = NULL;
    if (A->local_num_rows) 
        part = new int[A->local_num_rows];

    int err = ParMETIS_V3_PartKway(vtxdist, xadj.data(), adjncy.data(), vwgt, adjwgt, 
            &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), options.data(),
            &edgecut, part, &comm);

    return part;
}

#endif
