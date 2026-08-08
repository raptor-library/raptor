#ifndef RAPTOR_MATRIX_TRAITS_HPP
#define RAPTOR_MATRIX_TRAITS_HPP

#include "par_matrix.hpp"

namespace raptor {

template <class T>
using is_bsr_or_csr = std::enable_if_t<std::is_same_v<T, ParCSRMatrix> ||
                                       std::is_same_v<T, ParBSRMatrix>, bool>;
template <class T>
using is_seq_bsr_or_csr = std::enable_if_t<
	std::is_same_v<T, CSRMatrix> || std::is_same_v<T, BSRMatrix>, bool>;

template <class T> struct is_bsr : std::false_type {};
template <> struct is_bsr<ParBSRMatrix> : std::true_type {};
template <> struct is_bsr<BSRMatrix> : std::true_type {};
template <class T> inline constexpr bool is_bsr_v = is_bsr<T>::value;

inline BSRMatrix & bsr_cast(Matrix &mat) { return dynamic_cast<BSRMatrix &>(mat); }
inline const BSRMatrix & bsr_cast(const Matrix & mat) { return dynamic_cast<const BSRMatrix&>(mat); }

template <class T> struct matrix_value;
template <> struct matrix_value<ParCSRMatrix> { using type = double; };
template <> struct matrix_value<ParBSRMatrix> { using type = double*; };
template <class T>
using matrix_value_t = typename matrix_value<T>::type;

template <class T> struct sequential_matrix;
template<> struct sequential_matrix<ParBSRMatrix> { using type = BSRMatrix; };
template<> struct sequential_matrix<ParCSRMatrix> { using type = CSRMatrix; };
template <class T>
using sequential_matrix_t = typename sequential_matrix<T>::type;

template <class T> struct sequential_csc_matrix;
template<> struct sequential_csc_matrix<ParBSRMatrix> { using type = BSCMatrix; };
template<> struct sequential_csc_matrix<ParCSRMatrix> { using type = CSCMatrix; };
template <class T>
using sequential_csc_matrix_t = typename sequential_csc_matrix<T>::type;

template <class T> struct parallel_csc_matrix;
template<> struct parallel_csc_matrix<ParBSRMatrix> { using type = ParBSCMatrix; };
template<> struct parallel_csc_matrix<ParCSRMatrix> { using type = ParCSCMatrix; };
template <class T>
using parallel_csc_matrix_t = typename parallel_csc_matrix<T>::type;

template <class T, is_bsr_or_csr<T> = true>
inline int total_local_num_rows(const T& A){
    if constexpr (is_bsr_v<T>){
        return A.local_num_rows * A.on_proc->b_rows;
    }
    else{
        return A.local_num_rows;
    }
    
}

template <class T, is_bsr_or_csr<T> = true>
inline int total_global_num_rows(const T& A)
{
    if constexpr (is_bsr_v<T>)
    {
        return A.global_num_rows * A.on_proc->b_rows;
    }
    else
    {
        return A.global_num_rows;
    }
}

// map block row indices to global scalar indices
template <class T, is_bsr_or_csr<T> = true>
std::vector<int> total_local_row_map(const T& A)
{
    if constexpr (is_bsr_v<T>)
    {
        std::vector<int> total_rows;
        total_rows.reserve(A.local_row_map.size()*A.on_proc -> b_rows);
        for (int i: A.local_row_map){
            for (int j = 0;j <A.on_proc -> b_rows;j++)
            {
                total_rows.emplace_back(i*A.on_proc->b_rows+j);
            }

        }
        return total_rows;
    }
    else
    {
        return A.local_row_map;
    }
}
}

#endif
