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
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition);
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A);
void make_noncontiguous(ParCSRMatrix* A, std::vector<int>& column_map);

#endif

