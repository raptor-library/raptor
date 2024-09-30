#ifndef RAPTOR_MATRIX_TRAITS_HPP
#define RAPTOR_MATRIX_TRAITS_HPP

#include "par_matrix.hpp"

namespace raptor {

template <class T>
using is_bsr_or_csr = std::enable_if_t<std::is_same_v<T, ParCSRMatrix> ||
                                       std::is_same_v<T, ParBSRMatrix>, bool>;

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
}

#endif
