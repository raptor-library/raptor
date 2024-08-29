#ifndef RAPTOR_PYAMG_UTILS_HPP
#define RAPTOR_PYAMG_UTILS_HPP

#include <vector>
#include <cmath>

namespace pyamg {

/* Sign-of Function overloaded for int, float and double
 * signof(x) =  1 if x > 0
 * signof(x) = -1 if x < 0
 * signof(0) =  1 if x = 0
 */
inline int signof(int a) { return (a<0 ? -1 : 1); }
inline float signof(float a) { return (a<0.0 ? -1.0 : 1.0); }
inline double signof(double a) { return (a<0.0 ? -1.0 : 1.0); }

/*
 * Return row-major index from 2d array index, A[row,col].
 */
template<class I>
inline I row_major(const I row, const I col, const I num_cols)
{
    return row*num_cols + col;
}

/*
 * Return column-major index from 2d array index, A[row,col].
 */
template<class I>
inline I col_major(const I row, const I col, const I num_rows)
{
    return col*num_rows + row;
}

/* QR-decomposition using Householer transformations on dense
 * 2d array stored in either column- or row-major form.
 *
 * Parameters
 * ----------
 * A : double array
 *     2d matrix A stored in 1d column- or row-major.
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * Q : vector<double>
 *     Matrix Q stored in same format as A.
 * R : in-place
 *     R is stored over A in place, in same format.
 *
 * Notes
 * ------
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
std::vector<T> QR(T A[],
                  const I &m,
                  const I &n,
                  const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Initialize Q to identity
    std::vector<T> Q(m*m,0);
    for (I i=0; i<m; i++) {
        Q[get_ind(i,i,m)] = 1;
    }

    // Loop over columns of A using Householder reflections
    for (I j=0; j<n; j++) {

        // Break loop for short fat matrices
        if (m <= j) {
            break;
        }

        // Get norm of next column of A to be reflected. Choose sign
        // opposite that of A_jj to avoid catastrophic cancellation.
        // Skip loop if norm is zero, as that means column of A is all
        // zero.
        T normx = 0;
        for (I i=j; i<m; i++) {
            T temp = A[get_ind(i,j,*C)];
            normx += temp*temp;
        }
        normx = std::sqrt(normx);
        if (normx < 1e-12) {
            continue;
        }
        normx *= -1*signof(A[get_ind(j,j,*C)]);

        // Form vector v for Householder matrix H = I - tau*vv^T
        // where v = R(j:end,j) / scale, v[0] = 1.
        T scale = A[get_ind(j,j,*C)] - normx;
        T tau = -scale / normx;
        std::vector<T> v(m-j,0);
        v[0] = 1;
        for (I i=1; i<(m-j); i++) {
            v[i] = A[get_ind(j+i,j,*C)] / scale;
        }

        // Modify R in place, R := H*R, looping over columns then rows
        for (I k=j; k<n; k++) {

            // Compute the kth element of v^T * R
            T vtR_k = 0;
            for (I i=0; i<(m-j); i++) {
                vtR_k += v[i] * A[get_ind(j+i,k,*C)];
            }

            // Correction for each row of kth column, given by
            // R_ik -= tau * v_i * (vtR_k)_k
            for (I i=0; i<(m-j); i++) {
                A[get_ind(j+i,k,*C)] -= tau * v[i] * vtR_k;
            }
        }

        // Modify Q in place, Q = Q*H
        for (I i=0; i<m; i++) {

            // Compute the ith element of Q * v
            T Qv_i = 0;
            for (I k=0; k<(m-j); k++) {
                Qv_i += v[k] * Q[get_ind(i,k+j,m)];
            }

            // Correction for each column of ith row, given by
            // Q_ik -= tau * Qv_i * v_k
            for (I k=0; k<(m-j); k++) {
                Q[get_ind(i,k+j,m)] -= tau * v[k] * Qv_i;
            }
        }
    }

    return Q;
}

/* Backward substitution solve on upper-triangular linear system,
 * Rx = rhs, where R is stored in column- or row-major form.
 *
 * Parameters
 * ----------
 * R : double array, length m*n
 *     Upper-triangular array stored in column- or row-major.
 * rhs : double array, length m
 *     Right hand side of linear system
 * x : double array, length n
 *     Preallocated array for solution
 * m : &int
 *     Number of rows in R
 * n : &int
 *     Number of columns in R
 * is_col_major : bool
 *     True if R is stored in column-major, false
 *     if R is stored in row-major.
 *
 * Returns
 * -------
 * Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * R need not be square, the system will be solved over the
 * upper-triangular block of size min(m,n). If remaining entries
 * insolution are unused, they will be set to zero. If a zero
 * is encountered on the ith diagonal, x[i] is set to zero.
 *
 */
template<class I, class T>
void upper_tri_solve(const T R[],
                     const T rhs[],
                     T x[],
                     const I m,
                     const I n,
                     const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Backward substitution
    I rank = std::min(m,n);
    for (I i=(rank-1); i>=0; i--) {
        T temp = rhs[i];
        for (I j=(i+1); j<rank; j++) {
            temp -= R[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(R[get_ind(i,i,*C)]) < 1e-12) {
            x[i] = 0.0;
        }
        else {
            x[i] = temp / R[get_ind(i,i,*C)];
        }
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}

/* Method to solve the linear least squares problem.
 *
 * Parameters
 * ----------
 * A : double array, length m*n
 *     2d array stored in column- or row-major.
 * b : double array, length m
 *     Right hand side of unconstrained problem.
 * x : double array, length n
 *     Container for solution
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * x : vector<double>
 *    Solution to constrained least squares problem.
 *
 * Notes
 * -----
 * If system is under determined, free entries are set to zero.
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
void least_squares(T A[],
                   T b[],
                   T x[],
                   const I &m,
                   const I &n,
                   const I is_col_major=0)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    if (is_col_major) {
        get_ind = &col_major;
    }
    else {
        get_ind = &row_major;
    }

    // Take QR of A
    std::vector<T> Q = QR(A,m,n,is_col_major);

    // Multiply right hand side, b:= Q^T*b. Have to make new vector, rhs.
    std::vector<T> rhs(m,0);
    for (I i=0; i<m; i++) {
        for (I k=0; k<m; k++) {
            rhs[i] += b[k] * Q[get_ind(k,i,m)];
        }
    }

    // Solve upper triangular system, store solution in x.
    upper_tri_solve(A,&rhs[0],x,m,n,is_col_major);
}

}
#endif
