// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "raptor/core/matrix.hpp"

using namespace raptor;

// TODO -- currently assumes partitions are the same 
Matrix* Matrix::add(CSRMatrix* B, bool remove_dup)
{
    CSRMatrix* A = to_CSR();
    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    A->add_append(B, C, remove_dup);
    delete A;
    return C;
}
void Matrix::add_append(CSRMatrix* B, CSRMatrix* C, bool remove_dup)
{
    CSRMatrix* A = to_CSR();
    A->add_append(B, C, remove_dup);
    delete A;
}
Matrix* Matrix::subtract(CSRMatrix* B)
{
    CSRMatrix* A = to_CSR();
    CSRMatrix* C = A->subtract(B);
    delete A;
    return C;
}


CSRMatrix* CSRMatrix::add(CSRMatrix* B, bool remove_dup)
{
    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    add_append(B, C, remove_dup);
    return C;
}

namespace impl {

template <class T> struct is_bsr : std::false_type {};
template <> struct is_bsr<BSRMatrix> : std::true_type {};
template <class T> inline constexpr bool is_bsr_v = is_bsr<T>::value;

template <class T, std::enable_if_t<std::is_same_v<T, CSRMatrix> ||
                                        std::is_same_v<T, BSRMatrix>,
                                    bool> = true>
void add_append(const T & A, const T & B, T & C, bool remove_dup)
{
    auto vals = [](auto & mat) -> auto & {
	    if constexpr (is_bsr_v<T>) return mat.block_vals;
	    else return mat.vals;
    };
    C.resize(A.n_rows, A.n_cols);
    int C_nnz = A.nnz + B.nnz;
    C.idx2.resize(C_nnz);
    vals(C).resize(C_nnz);

    auto copy_vals = [b_size = A.b_size](auto beg, auto end, auto out) {
	    (void) b_size;
	    if constexpr (is_bsr_v<T>) {
		    for (; beg != end; ++beg, ++out) {
			    auto val = new double[b_size];
			    std::copy(val, val + b_size, *beg);
			    *out = val;
		    }
	    } else std::copy(beg, end, out);
    };

    C_nnz = 0;
    C.idx1[0] = 0;
    for (int i = 0; i < A.n_rows; i++)
    {
	    auto add_row = [&](const T & src, T & dst) {
		    auto start = src.idx1[i];
		    auto end = src.idx1[i+1];
		    std::copy(src.idx2.begin() + start,
		              src.idx2.begin() + end,
		              dst.idx2.begin() + C_nnz);
		    copy_vals(vals(src).begin() + start,
		              vals(src).begin() + end,
		              vals(dst).begin() + C_nnz);
		    return end - start;
	    };

	    C_nnz += add_row(A, C);
	    C_nnz += add_row(B, C);

        C.idx1[i+1] = C_nnz;
    }
    C.nnz = C_nnz;
    C.sort();
    if (remove_dup)
        C.remove_duplicates();
}
}

void CSRMatrix::add_append(CSRMatrix* B, CSRMatrix* C, bool remove_dup)
{
	impl::add_append(*this, *B, *C, remove_dup);
}

void BSRMatrix::add_append(BSRMatrix * B, BSRMatrix * C, bool remove_dup)
{
	impl::add_append(*this, *B, *C, remove_dup);
}

CSRMatrix* CSRMatrix::subtract(CSRMatrix* B)
{
    int start, end;

    assert(n_rows == B->n_rows);
    assert(n_cols == B->n_cols);

    CSRMatrix* C = new CSRMatrix(n_rows, n_cols, 2*nnz);
    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.emplace_back(idx2[j]);
            C->vals.emplace_back(vals[j]);
        }
        start = B->idx1[i];
        end = B->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            C->idx2.emplace_back(B->idx2[j]);
            C->vals.emplace_back(-B->vals[j]);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
    C->sort();
    C->remove_duplicates();

    return C;
}


