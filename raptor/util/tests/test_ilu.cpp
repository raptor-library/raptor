#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"
//#include <mpi.h>

using namespace raptor;

int main(int argc, char* argv[]){

	///For testing small matrix
	COOMatrix * mat = new COOMatrix(5,5);


	mat->add_value(0, 0, 1.0);
	mat->add_value(0, 1, 2.0);
	mat->add_value(0, 4, 3.0);

	mat->add_value(1, 0, 4.0);
	mat->add_value(1, 1, 3.0);
	mat->add_value(1, 3, 6.0);

	mat->add_value(2, 1, 7.0);
	mat->add_value(2, 2, 5.0);

	mat->add_value(3, 3, 8.0);

	mat->add_value(4, 3, 2.0);
	mat->add_value(4, 4, 1.0);

	mat->sort();

	//mat->print();

	//mat->diag_scaling(0);

	//mat->print();

	CSRMatrix* A = new CSRMatrix(mat);

	//Extract attributes for A in CSR
	std::vector<int>  A_rowptr = A->idx1;
	std::vector<int>  A_cols = A->idx2;
	std::vector<double>  A_data = A->vals;

	A->print();
	
	printf("A csr rowptr = ");
	for (auto i: A_rowptr)
		printf("%d ",i);
	printf("\n");

	printf("A csr cols = ");
	for (auto i: A_cols)
		printf("%d ",i);
	printf("\n");

	printf("A csr data = ");
	for (auto i: A_data)
		printf("%f ", i);
	printf("\n");

	
	//levls->sort();

	//CSRMatrix* levls_csr = new CSRMatrix(levls);
	
	Matrix* factors = A->ilu_levels();
	
	printf("After ilu k funtion returns\n");	
	//printf("ILU factors: \n");
	//factors->print();

}
