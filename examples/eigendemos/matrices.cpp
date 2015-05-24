/* HOW TO: manipulate data as an Eigen matrix
 * Erin Molloy | emolloy2@illinois.edu
 * May 20, 2013
 */

#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>

void init_array(double* A, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
       for (int j = 0; j < ncols; j++) {
           A[i*ncols + j] = 10.0 * (i*ncols + j) + 10.0;
       }
    }
}

void change_array(double* A, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
           A[i*ncols + j] /= 10.0;
        }
    }
}

void print_array(double* A, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            std::cout << A[i*ncols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // original data, double*
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Original data (double*):" << std::endl;
    const int m = 3, n = 3;
    double* data = (double*) _mm_malloc(m*n*sizeof(double), 8);
    init_array(data, m, n);
    print_array(data, m, n);
    std::cout << "--------------------------------------" << std::endl;
    
    // double* -> Eigen Matrix
    std::cout << "double* -> Eigen Matrix:" << std::endl;
    Eigen::Matrix<double, m, n, Eigen::RowMajor> mat; 
    mat = Eigen::Map<Eigen::Matrix<double,m,n,Eigen::RowMajor> >(data);
    std::cout << mat << std::endl;
    std::cout << "Manipulating Eigen Matrix..." << std::endl;
    mat = mat / 10.0;
    std::cout << mat << std::endl;
    std:: cout << "...does NOT change original data" << std::endl;
    print_array(data, m, n);
    std::cout << "i.e., the data is copied!" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // Eigen matrix -> double*
    std::cout << "Eigen Matrix -> double*:" << std::endl;
    double* ptr = &mat(0);
    print_array(ptr, m, n);
    std::cout << "Manipulating data via ptr (double*)..." << std::endl;
    change_array(ptr, m, n);
    print_array(ptr, m, n);
    std::cout << "...changes the Eigen Matrix" << std::endl;
    std::cout << mat << std::endl;
    std::cout << "... but NOT original data" << std::endl;
    print_array(data, m, n);
    std::cout << "i.e., these ptrs are NOT the same!" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    // double* -> Map to Eigen Matrix
    std::cout << "double* -> Map to Eigen Matrix:" << std::endl;
    Eigen::Map<Eigen::Matrix<double, m, n, Eigen::RowMajor>, 
               Eigen::Aligned> map(data);
    std::cout << map << std::endl;
    std::cout << "Manipulating Map to Eigen Matrix..." << std::endl;
    map = map / 10.0;
    std::cout << map << std::endl;
    std:: cout << "...changes original data" << std::endl;
    print_array(data, m, n);
    std::cout << "i.e., the data is NOT copied!" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    // Map to Eigen Matrix -> double*
    std::cout << "Map to Eigen Matrix -> double*:" << std::endl;
    ptr = &map(0);
    print_array(ptr, m, n);
    std::cout << "Manipulating data via ptr (double*)..." << std::endl;
    change_array(ptr, m, n);
    print_array(ptr, m, n);
    std::cout << "...changes Map to Eigen Matrix" << std::endl;
    std::cout << map << std::endl;
    std::cout << "...AND original data" << std::endl;
    print_array(data, m, n);
    std::cout << "i.e., these ptrs are the same!" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
}
