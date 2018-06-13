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
		printf("%f ",i);
	printf("\n");
	
	//return 0; 

	A_bsr->print();

	int lof = 0;

	Matrix* factors = A_bsr->ilu_k(lof);
	
	factors->print();

	printf("After ilu k funtion returns\n");	
	
	return 0;

}
