#ifndef RAPTOR_GALLERY_REPARTITION_HPP
#define RAPTOR_GALLERY_REPARTITION_HPP

#include <mpi.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/matrix_IO.hpp"
#include "ptscotch.h"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>

using namespace raptor;

int* ptscotch_partition(ParMatrix* A);
void repartition_matrix(ParMatrix* A, std::vector<coo_data>& new_mat, int* partition);

#endif

