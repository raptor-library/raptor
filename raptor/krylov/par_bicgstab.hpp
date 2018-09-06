#ifndef RAPTOR_KRYLOV_PAR_BICGSTAB_HPP
#define RAPTOR_KRYLOV_PAR_BICGSTAB_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_multilevel.hpp"
#include <vector>

using namespace raptor;

void BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void Pre_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqInner_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqInnerSeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void PI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, MPI_Comm &inner_comm, int &my_color,
                 int &first_root, int &second_root, int part_global, int contig, double tol = 1e-05, int max_iter = -1);

void PrePI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, MPI_Comm &inner_comm, int &my_color,
                    int &first_root, int &second_root, int part_global, int contig, double tol = 1e-05, int max_iter = -1);

#endif
