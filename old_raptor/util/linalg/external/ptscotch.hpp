// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_GALLERY_PTSCOTCH_HPP
#define RAPTOR_GALLERY_PTSCOTCH_HPP

#include <mpi.h>
#include "core/types.hpp"
#include "ptscotch.h"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>

using namespace raptor;

int* ptscotch_partition(ParCSRMatrix* A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Variables for Graph Partitioning
    SCOTCH_Num* partition = new SCOTCH_Num[A->local_num_rows + 2];
    SCOTCH_Num baseval = 0;
    SCOTCH_Num vertlocnbr = A->local_num_rows;
    SCOTCH_Num vertlocmax = A->local_num_rows;
    SCOTCH_Num* vertloctab = new SCOTCH_Num[vertlocnbr + 2];
    SCOTCH_Num* vendloctab = &vertloctab[1];
    SCOTCH_Num* veloloctab = NULL;
    SCOTCH_Num* vlblloctab = NULL;
    SCOTCH_Num edgelocnbr = A->local_nnz;
    SCOTCH_Num edgelocsiz = A->local_nnz;
    SCOTCH_Num* edgeloctab = new SCOTCH_Num[edgelocsiz + 1];
    SCOTCH_Num* edgegsttab = NULL;
    SCOTCH_Num* edloloctab = NULL;

    int row_start, row_end;
    int idx, gbl_idx, ctr;
    int err;

    // Find matrix edge indices for PT Scotch
    ctr = 0;
    vertloctab[0] = 0;
    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_start = A->on_proc->idx1[row];
        row_end = A->on_proc->idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx = A->on_proc->idx2[j];
            if (idx == row) continue;
            gbl_idx = A->on_proc_column_map[idx];
            edgeloctab[ctr] = gbl_idx;
            ctr++;
        }

        if (A->off_proc_num_cols)
        {
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                idx = A->off_proc->idx2[j];
                gbl_idx = A->off_proc_column_map[idx];
                edgeloctab[ctr] = gbl_idx;
                ctr++;
            }
        }
        vertloctab[row+1] = ctr;
    }
    edgelocnbr = ctr;
    edgelocsiz = ctr;


    SCOTCH_Dgraph dgraphdata;
    SCOTCH_Strat stratdata;
    SCOTCH_Arch archdata;

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    SCOTCH_dgraphInit(&dgraphdata, comm);
    SCOTCH_dgraphBuild(&dgraphdata, baseval, vertlocnbr, vertlocmax,
            vertloctab, vendloctab, veloloctab, vlblloctab, edgelocnbr, edgelocsiz,
            edgeloctab, edgegsttab, edloloctab);
    SCOTCH_dgraphCheck(&dgraphdata);

    SCOTCH_stratInit(&stratdata);
    SCOTCH_dgraphPart(&dgraphdata, num_procs, &stratdata, partition);

    SCOTCH_stratExit(&stratdata);
    SCOTCH_dgraphExit(&dgraphdata);

    delete[] vertloctab;
    delete[] edgeloctab;

    MPI_Comm_free(&comm);

    return partition;
}


#endif

