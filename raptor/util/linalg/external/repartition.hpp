#ifndef RAPTOR_GALLERY_REPARTITION_HPP
#define RAPTOR_GALLERY_REPARTITION_HPP

#include <mpi.h>
#include "core/types.hpp"
#include "ptscotch.h"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>

using namespace raptor;

int* ptscotch_partition(ParCSRMatrix* A);
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, std::vector<int>& new_local_rows);
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, std::vector<int>& new_local_rows);
void make_noncontiguous(ParCSRMatrix* A, std::vector<int>& column_map);

#endif

