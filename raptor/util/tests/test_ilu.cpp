#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
#include <mpi.h>

using namespace raptor;

int main(int argc, char* argv[]){

	///For testing small matrix
	COOMatrix * mat = new COOMatrix(4,4);


	mat->add_value(0, 0, 1.0);
	mat->add_value(0, 1, 0);
	mat->add_value(0, 2, 2.0);
	mat->add_value(0, 3, 0.0);

	mat->add_value(1, 0, 0.0);
	mat->add_value(1, 1, 4.0);
	mat->add_value(1, 2, 3.0);
	mat->add_value(1, 3, 6.0);

	mat->add_value(2, 0, 7.0);
	mat->add_value(2, 1, 0);
	mat->add_value(2, 2, 5.0);
	mat->add_value(2, 3, 0);

	mat->add_value(3, 0, 0);
	mat->add_value(3, 1, 9.0);
	mat->add_value(3, 2, 0);
	mat->add_value(3, 3, 8.0);

	mat->sort();

	mat->print();

	//mat->diag_scaling(0);

	//mat->print();

	CSRMatrix* A = new CSRMatrix(mat);

	//Extract attributes for A in CSR
	std::vector<int> & A_rowptr = A->row_ptr();
	std::vector<int> & A_cols = A->cols();
	std::vector<double> & A_data = A->data();

	std::cout << "A csr rowptr =" << " ";
	for (auto i: A_rowptr)
		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "A csr cols =" << " ";
	for (auto i: A_cols)
		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	std::cout << "A csr data =" << " ";
	for (auto i: A_data)
		std::cout << i << ' ';
	std::cout << "\n" <<std::endl;

	COOMatrix* levls = new COOMatrix(4,4);

	A->ilu_k(5,levls);
	
	//levls->sort();

	//CSRMatrix* levls_csr = new CSRMatrix(levls);
		
	//levls_csr->print();

}
