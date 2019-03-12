// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
//
#ifndef RAPTOR_GALLERY_REPARTITION_HPP
#define RAPTOR_GALLERY_REPARTITION_HPP

#include <mpi.h>
#include "core/types.hpp"
#include "core/mpi_types.hpp"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>

using namespace raptor;

CSRMatrix* send_matrix(CSRMatrix* A_on, CSRMatrix* A_off, int* partition, 
        int* local_row_map, int* on_proc_column_map, int* off_proc_column_map, 
	aligned_vector<int>& proc_row_sizes, aligned_vector<int>& new_local_rows);
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, 
        aligned_vector<int>& new_local_rows, bool make_contig = true);
void make_contiguous(ParCSRMatrix* A, bool form_comm = true);

#endif

