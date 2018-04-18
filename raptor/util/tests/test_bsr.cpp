#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor; 

int main(int argc, char** argv){
	std::vector<int> row_ptr = {0,2,3,5};
	std::vector<int> cols = {0,1,1,1,2};
	std::vector<double> vals = {1.0,0.0,2.0,1.0,6.0,7.0,8.0,2.0,1.0,4.0,5.0,1.0,4.0,3.0,0.0,0.0,7.0,2.0,0.0,0.0};

	int rows_in_block = 2;
	int cols_in_block = 2;
	int n = 6;


	BSRMatrix* A_bsr = new BSRMatrix(n, n, rows_in_block, cols_in_block, row_ptr, cols, vals);

	printf("idx 1 = ");
	for(auto i: A_bsr->idx1)
		printf("%d ",i);
	printf("\n");
	
	printf("idx 2 = ");
	for(auto i: A_bsr->idx2)
		printf("%d ",i);
	printf("\n");

	printf("vals = ");
	for(auto i: A_bsr->vals)
		printf("%d ",i);
	printf("\n");
	
	return 0; 

	Vector x(6);
	Vector b(6);
	Vector r(6);
	x.set_const_value(1.0);
	b.set_const_value(0.0);
	r.set_const_value(0.0);

	A_bsr->print();

	printf("\n-----------------------\n");
	printf("A * x = b\n");
	printf("-----------------------\n");
	A_bsr->mult(x, b);
	b.print();

	b.set_const_value(0.0);
	A_bsr->mult_T(x, b);
	printf("\n-----------------------\n");
	printf("A^T * x = b\n");
	printf("-----------------------\n");
	b.print();

	b.set_const_value(2.0);
	A_bsr->residual(x, b, r);
	printf("\n-----------------------\n");
	printf("r = b - A * x\n");
	printf("-----------------------\n");
	r.print();

	return 0;

}
