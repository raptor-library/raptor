#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>
#include <math.h>

using namespace raptor;


Matrix* COOMatrix::ilu_k(int lof)
{
	printf("Function not implemented for COO type\n");
	return NULL;
}

Matrix* COOMatrix::ilu_levels()
{
	printf("Function not implemented for COO type\n");
	return NULL;
}

Matrix* COOMatrix::ilu_sparsity(Matrix* levls, int lof)
{
	printf("Function not implemented for COO type\n");
	return NULL;
}

Matrix* COOMatrix::ilu_symbolic(int lof)
{
	printf("Function not implemented for COO type\n");
	return NULL;
}

std::vector<double>& COOMatrix::ilu_numeric(Matrix* sparsity)
{
	printf("Function not implemented for COO type\n");
	std::vector<double> emp;
	return emp;
}


Matrix* CSRMatrix::ilu_k(int lof)
{
	printf("Begin ilu symbolic phase \n");
	Matrix* sparsity = ilu_symbolic(lof);

	printf("\n");
	
	printf("Begin ilu numeric phase \n");
	std::vector<double> factors_data = this->ilu_numeric(sparsity);
	printf("After numeric ilu phase completed\n");
	/*
	CSRMatrix* factors = new CSRMatrix(sparsity->n_rows,sparsity->n_cols);
	factors->nnz = sparsity->nnz;
	int factors_nnz_dense = factors->n_rows*factors->n_cols;

	factors->idx1.resize(sparsity->n_rows+1);
	if(factors_nnz_dense){
		factors->idx2.reserve(sparsity->nnz);
		factors->vals.reserve(sparsity->nnz);
	}

	for(int i = 0; i < sparsity->n_rows+1;i++){
		factors->idx1[i] = sparsity->idx1[i];
	}

	for(int i =0; i < sparsity->nnz;i++){
		factors->idx2[i] = sparsity->idx2[i];
		factors->vals[i] = factors_data[i];
	}
	*/
	return sparsity;
	//return factors;
}

Matrix* CSRMatrix::ilu_symbolic(int lof)
{
	Matrix* levels = this->ilu_levels();
	Matrix* sparsity = this->ilu_sparsity(levels, lof);
	return sparsity;
}

Matrix* CSRMatrix::ilu_sparsity(Matrix* levls, int lof)
{
	//printf("Begin ilu sparsity \n");
	Matrix* sparsity = new CSRMatrix(levls->n_rows,levls->n_cols);

	sparsity->n_rows = levls->n_rows;
	sparsity->n_cols = levls->n_cols;
	sparsity->nnz = 0;
	int sparsity_nnz_dense = sparsity->n_rows*sparsity->n_cols;

	sparsity->idx1.resize(levls->n_rows+1);
	if(sparsity_nnz_dense){
		sparsity->idx2.reserve(sparsity_nnz_dense);
		sparsity->vals.reserve(sparsity_nnz_dense);
	}
	sparsity->idx1[0] = 0;
	
	for(int row_i = 0; row_i<sparsity->n_rows;row_i++){
		sparsity->idx1[row_i] = sparsity->nnz;
		int start_ri = levls->idx1[row_i];
		int end_ri = levls->idx1[row_i+1];
		for(int j = start_ri; j<end_ri;j++){
			int col_j = levls->idx2[j];
			int levl_j = levls->vals[j];
			if(levl_j <= lof){
				sparsity->idx2.push_back(col_j);
				sparsity->vals.push_back(levl_j);
				sparsity->nnz = sparsity->nnz + 1;
			}
		}
	}
	sparsity->idx1[sparsity->n_rows] = sparsity->nnz;
/*	
	printf("sparsity idx1 = ");
    for (auto i: sparsity->idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("sparsity idx2 = ");
    for (auto i: sparsity->idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("sparsity vals = ");
    for (auto i: sparsity->vals)
		printf("%.2f ",i);
	printf("\n");
*/	

	return sparsity; 
}

std::vector<double>& CSRMatrix::ilu_numeric(Matrix* sparsity){

	int m = n_rows;
	int n = n_cols;
	
	std::vector<double> factors(sparsity->nnz,0.0);
/*	
	printf("factors data = ");
	for(auto i: factors)
		printf("%.2f ",i);
	printf("\n");
*/
	//initialize entries in factors to entries in A
	int index_i = 0;
	for(int i = 0; i<sparsity->nnz;i++){
		if(sparsity->vals[i] == 0.0){
			factors[i] = vals[index_i];
			index_i++;
		}
	}
/*
	printf("initialize factors data = ");
	for(auto i: factors)
		printf("%.2f ",i);
	printf("\n");
*/
	//get vector of diagonal elements
	std::vector<double> diag_vec(n_rows,0.0);
	
	for(int row = 0; row < n_rows; row++){
		int start_i = sparsity->idx1[row];
		int end_i = sparsity->idx1[row+1];
		for(int j = start_i; j<end_i; j++){
			int col = sparsity->idx2[j];
			double val = factors[j];
			if(row == col)
				diag_vec[row] = val;
		}
	}
/*
	printf("diag vec = ");
	for(auto i: diag_vec)
		printf("%.2f ",i);
	printf("\n");
*/	
	//ikj Gaussian Elimination
	for(int row_i = 1; row_i<n_rows; row_i++){
		//printf("row i = %d\n", row_i);
		int start_ri = sparsity->idx1[row_i];
		int end_ri = sparsity->idx1[row_i+1];

		for(int j = start_ri; j<end_ri; j++){
			//printf("j = %d\n", j);
			int col_k = sparsity->idx2[j];
			//printf("col k = %d\n",col_k);

			//since k loop goes from 0 to i-1
			if(col_k >= row_i){
				//printf("exit k loop\n");
				break;
			}

			//compute multiplier
			factors[j] /= diag_vec[col_k];
			double mult = factors[j];
			//printf("multiplier = %.2f\n",mult);

			if(row_i == col_k){
				//printf("diag element\n");
				diag_vec[row_i] = mult;
			}
		
			//get row k from original matrix
			int start_rk = idx1[col_k];
			int end_rk = idx1[col_k+1];

			//temp variables
			int ind_i = -1;
			int ind_k = -1;

			int col_ij = -1;
			int col_kj = -1;
			int col_ik = col_k;
			
			//printf("\n");
			//printf("Update rest\n");
			//get starting indices from rows i and k such that col>= k+1
			for(int i=start_ri; i< end_ri;i++){
				int col = sparsity->idx2[i];
				if(col >= col_k+1){
					ind_i = i;
					col_ij = sparsity->idx2[i];
					break;
				}		
			}

			for(int i=start_rk; i<end_rk;i++){
				int col = idx2[i];
				if(col >= col_k+1){
					ind_k = i;
					col_kj = sparsity->idx2[i];
					break;
				}
			}

			//check if reached end of row k
			if(ind_k == -1)
				col_kj = n_rows + 1;

			double current_it = 0.0;
			int current_col = -1;
			int current_row = -1;

			while(1){
				if((col_ij >= n) && (col_kj >= n))
					break;

				if(col_ij == col_kj){
					current_col = col_ij;
					current_row = row_i;
					current_it = factors[ind_i] - mult*factors[ind_k];
					factors[ind_i] = current_it;
				
					//check if it's a diagonal element
					if(current_row == current_col)
						diag_vec[current_row] = current_it;
				
					ind_i++;
					ind_k++;	
				
				}	
				else if(col_ij<col_kj){
					current_col = col_ij;
					current_row = row_i;
					current_it = factors[ind_i];
					factors[ind_i] = current_it;

					//check if it's a diagonal element
					if(current_row == current_col)
						diag_vec[current_row] = current_it;

					ind_i++;
				}
				else if(col_ij>col_kj){
					current_col = col_kj; 
					current_row = col_k;
					current_it = -mult*factors[ind_k];
					factors[ind_k] = current_it;

					//check if it's a diagonal element
					if(current_row == current_col)
						diag_vec[current_row] = current_it;

					ind_k++;
				}

				if((ind_i<end_ri)&&(ind_i !=-1))
					col_ij = sparsity->idx2[ind_i];
				else
					col_ij = n_rows + 1;

				if((ind_k<end_rk)&&(ind_k != -1))
					col_kj = sparsity->idx2[ind_k];
				else
					col_kj = n_rows + 1;
			}
		}
	}
	printf("factors = ");
    for (auto i: factors)
		printf("%.2f ",i);
	printf("\n");
 
 	printf("ilu numeric phase done\n");
	printf("\n");
	return factors;
}

Matrix* CSRMatrix::ilu_levels()
{
	//printf("Begin ilu levels \n");
	Matrix * levls = new CSRMatrix(n_rows,n_cols);

	//initialize vectors for final levels matrix
	levls->n_rows = n_rows;
	levls->n_cols = n_cols;
	levls->nnz = 0;
	int levls_nnz_dense = n_rows*n_cols;

	
	levls->idx1.resize(n_rows+1);
	if(levls_nnz_dense){
		levls->idx2.reserve(levls_nnz_dense);
		levls->vals.reserve(levls_nnz_dense);
	}

	levls->idx1[0] = 0;

	//copy first row indices into final levls matrix cz that doesn't change
	int start_r = idx1[0];
	int end_r = idx1[1];
	for(int i = start_r; i<end_r;i++){
		levls->idx2.push_back(idx2[i]);
		levls->vals.push_back(0);
		levls->nnz = levls->nnz + 1;
		//printf("First row: Added %lf at 0,%d\n",val,col);
	}
	levls->idx1[1] = levls->nnz;
	
	/*	
	printf("Levls idx1 = ");
    for (auto i: levls->idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("Levls idx2 = ");
    for (auto i: levls->idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("Levls vals = ");
    for (auto i: levls->vals)
		printf("%e ",i);
	printf("\n");
	*/

	//temporary vector for level of fills of current row
	std::vector <int>  current_row_levls;
	std::vector <int>  current_row_idx2;
	std::vector <int>  temp_row_levls;
	std::vector <int>  temp_row_idx2;

	//dummy pointers for swapping
	std::vector <int>  dummy_row_levls;
	std::vector <int>  dummy_row_idx2;
/////////////////////////////////////////
//////////// ASK AMANDA HOW TO ACCESS NNZ PER ROW
	current_row_levls.reserve(n_cols);
	current_row_idx2.reserve(n_cols);
	temp_row_levls.reserve(n_cols);
	temp_row_idx2.reserve(n_cols);
	dummy_row_levls.reserve(n_cols);
	dummy_row_idx2.reserve(n_cols);

	int start_ri, end_ri;
		
	//begin ILU process
	for(int row_i = 1; row_i < n_rows;row_i++){
		//get row i
		//printf("row i = %d\n",row_i);

		start_ri = idx1[row_i];
		end_ri = idx1[row_i+1];
		int nnz_i = 0;
		int temp_nnz_i = 0;
		int current_row_ind = 0;

		//initialize temporary levls and idx2 vectors for row i
		for(int j = start_ri; j < end_ri; j++){
			current_row_levls.push_back(0);
			current_row_idx2.push_back(idx2[j]);
			nnz_i++;
		}

		//printf("Initialize temp levls and idx2 for row i\n");
		/*
		printf("current row Levls = ");
    	for (auto i: current_row_levls)
			printf("%d ",i);
		printf("\n");
  	
		printf("current row idx2 = ");
    	for (auto i: current_row_idx2)
			printf("%d ",i);
		printf("\n");
  		
		printf("\n");
		*/

		for(int j = start_ri; j < end_ri; j++){
			//printf("j = %d\n",j);

			int col_k = idx2[j];
			
			//printf("col k = %d\n", col_k);
			//since Gaussian elimination k loop goes from 0 to i-1
			if(col_k >= row_i){
				//printf("exit k loop\n");
				break;
			}
			
			//Add multiplier levels to final matrix 
			//printf("multiplier\n");
			/*
			printf("current row Levls = ");
    		for (auto i: current_row_levls)
				printf("%d ",i);
			printf("\n");
  	
			printf("current row idx2 = ");
    		for (auto i: current_row_idx2)
				printf("%d ",i);
			printf("\n");
  		
			printf("\n");
			*/
			
			levls->vals.push_back(current_row_levls[current_row_ind]);
			levls->idx2.push_back(current_row_idx2[current_row_ind]);
			/*
			printf("Levls idx1 = ");
    		for (auto i: levls->idx1)
				printf("%d ",i);
			printf("\n");
  
			printf("Levls idx2 = ");
    		for (auto i: levls->idx2)
				printf("%d ",i);
			printf("\n");
  	
			printf("Levls vals = ");
    		for (auto i: levls->vals)
				printf("%e ",i);
			printf("\n");
			*/
			current_row_ind ++;
			levls->nnz = levls->nnz + 1;		


			//get row k from updated levls matrix
			int start_rk = levls->idx1[col_k];
			int end_rk = levls->idx1[col_k+1];
			
			//temp variables
			int ind_i = -1;
			int ind_k = -1;
			
			int col_ij = -1;
			int col_kj = -1;
			int col_ik = col_k;

			int lev_ij = 100000;
			int lev_ik = 100000;
			int lev_kj = 100000;
			
			//printf("Update rest\n");
			//get starting indices for rows i and k such that col >=k+1
			for(int i = 0; i < nnz_i; i++){
				int col = current_row_idx2[i];
				if(col == col_k)
					lev_ik = current_row_levls[i];
				if(col >= col_k+1){
					ind_i = i;
					col_ij = current_row_idx2[i];
					break;
				}
			}


			for(int i = start_rk; i < end_rk; i++){
				int col = levls->idx2[i];
				if(col >=col_k+1){
					ind_k = i;
					col_kj = levls->idx2[i];
					break;
				}
			}

			//check if reached end of row k
			if(ind_k == -1)
				col_kj = n_rows + 1;

			int current_col = -1;
			int current_row = -1;
			//printf("\n");
			//printf("indices before while loop");
			//printf("ind i = %d\n",ind_i);
			//printf("ind k = %d\n",ind_k);
			//printf("\n");

			while(1){
				//printf("While iteration \n");
				//printf("col ij = %d\n",col_ij);
				//printf("col ik = %d\n",col_ik);
				//printf("col kj = %d\n",col_kj);
				//printf("ind i = %d\n",ind_i);
				//printf("ind k = %d\n",ind_k);
				//printf("\n");
				/*
				printf("temp row idx2 = ");
    			for (auto i: temp_row_idx2)
					printf("%d ",i);
				printf("\n");
  	
				printf("temp row levls = ");
    			for (auto i: temp_row_levls)
					printf("%d ",i);
				printf("\n");
  		
				printf("\n");


				printf("current row idx2 = ");
    			for (auto i: current_row_idx2)
					printf("%d ",i);
				printf("\n");
  	
				printf("current row levls = ");
    			for (auto i: current_row_levls)
					printf("%d ",i);
				printf("\n");
  		
				printf("\n");
				*/

		
				if((col_ij >= n_rows) and (col_kj >= n_rows)){
					//printf("exit j loop\n");
					break;
				}

				if(col_ij == col_kj){
					//printf("col(i,j) = col(k,j), equal\n");
					current_col = col_ij;
					current_row = row_i;

					lev_ij = current_row_levls[ind_i];
					//printf("lev(i,j) = %d\n",lev_ij);
					lev_kj = levls->vals[ind_k];

					//printf("lev(k,j) = %d\n",lev_kj);
					lev_ij = min(lev_ij, lev_ik+lev_kj+1);

					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_i++;
					ind_k++;
				}

				else if(col_ij<col_kj){
					//printf("col(i,j) < col(k,j)\n");
					current_col=col_ij;
					current_row=row_i;

					lev_ij = current_row_levls[ind_i];

					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_i++;
				}

				else if(col_ij>col_kj){
					//printf("col(i,j) > col(k,j)\n");
					current_col = col_kj;
					current_row = col_k;
					
					lev_kj = levls->vals[ind_k];

					//printf("lev(k,j) = %d\n",lev_kj);
					lev_ij = lev_ik + lev_kj + 1;
					
					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_k++;
				}

				//printf("ind i = %d\n",ind_i);
				//printf("nnz i = %d\n",nnz_i);
				if((ind_i < nnz_i) && (ind_i != -1))
					col_ij=current_row_idx2[ind_i];
				else
					col_ij = n_rows+1;

				//printf("ind k = %d\n",ind_k);
				//printf("end rk = %d\n",end_rk);
				if((ind_k < end_rk) && (ind_k != -1))
					col_kj=levls->idx2[ind_k];
				else
					col_kj = n_rows+1;
				/*
				printf("temp row levls = ");
    			for (auto i: temp_row_levls)
					printf("%d ",i);
				printf("\n");
  	
				printf("temp row idx2 = ");
    			for (auto i: temp_row_idx2)
					printf("%d ",i);
				printf("\n");
				*/
			}
			
			current_row_levls = temp_row_levls;
			current_row_idx2 = temp_row_idx2;
			nnz_i = temp_nnz_i;
			
			temp_row_levls.clear();
			temp_row_idx2.clear();
			temp_nnz_i = 0; 	
		}
		
		//Add current row levels to final levls matrix
		for(int i = 0; i<nnz_i;i++){
			levls->idx2.push_back(current_row_idx2[i]);
			levls->vals.push_back(current_row_levls[i]);
			levls->nnz = levls->nnz + 1;
			//printf("First row: Added %lf at 0,%d\n",val,col);
		}

		current_row_levls.clear();
		current_row_idx2.clear();

		levls->idx1[row_i+1] = levls->nnz;

	}//end i loop
 	
/*	
	printf("Levls rowptr = ");
    for (auto i: levls->idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("Levls cols = ");
    for (auto i: levls->idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("Levls data = ");
    for (auto i: levls->vals)
		printf("%.2f ",i);
	printf("\n");
*/	
	return levls;    
	
}


Matrix* CSCMatrix::ilu_k(int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

Matrix* CSCMatrix::ilu_levels()
{
	printf("Function not implemented \n");
	return NULL;
}


Matrix* CSCMatrix::ilu_sparsity(Matrix* levls, int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

Matrix* CSCMatrix::ilu_symbolic(int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

std::vector<double>& CSCMatrix::ilu_numeric(Matrix* sparsity)
{
	printf("Function not implemented \n");
	std::vector<double> emp;
	return emp;
}




///////////////////////////BSR////////////////////////// 
Matrix* BSRMatrix::ilu_k(int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

Matrix* BSRMatrix::ilu_levels()
{
	//printf("Begin ilu levels \n");
	Matrix * levls = new BSRMatrix(n_rows,n_cols,b_rows,b_cols);

	//initialize vectors for final levels matrix
	//levls->n_rows = n_rows;
	//levls->n_cols = n_cols;
	//levls->b_rows = b_rows;
	//levls->b_cols = b_cols;
	//levls->b_size = b_size;
	//levls->nnz = 0;
	int levls_nnz_dense = n_rows*n_cols;

	
	levls->idx1.resize(n_rows/b_rows+1);
	if(levls_nnz_dense){
		levls->idx2.reserve(levls_nnz_dense);
		levls->vals.reserve(levls_nnz_dense);
	}

	levls->idx1[0] = 0;

	//copy first row indices into final levls matrix cz that doesn't change
	int start_r = idx1[0];
	int end_r = idx1[1];
	for(int i = start_r; i<end_r;i++){
		levls->idx2.push_back(idx2[i]);
		levls->vals.push_back(0);
		levls->nnz = levls->nnz + 1;
		//printf("First row: Added %lf at 0,%d\n",val,col);
	}
	levls->idx1[1] = levls->nnz;
	
	/*	
	printf("Levls idx1 = ");
    for (auto i: levls->idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("Levls idx2 = ");
    for (auto i: levls->idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("Levls vals = ");
    for (auto i: levls->vals)
		printf("%e ",i);
	printf("\n");
	*/

	//temporary vector for level of fills of current row
	std::vector <int>  current_row_levls;
	std::vector <int>  current_row_idx2;
	std::vector <int>  temp_row_levls;
	std::vector <int>  temp_row_idx2;

	//dummy pointers for swapping
	std::vector <int>  dummy_row_levls;
	std::vector <int>  dummy_row_idx2;
/////////////////////////////////////////
//////////// ASK AMANDA HOW TO ACCESS NNZ PER ROW
	current_row_levls.reserve(n_cols);
	current_row_idx2.reserve(n_cols);
	temp_row_levls.reserve(n_cols);
	temp_row_idx2.reserve(n_cols);
	dummy_row_levls.reserve(n_cols);
	dummy_row_idx2.reserve(n_cols);

	int start_ri, end_ri;
		
	//begin ILU process
	for(int row_i = 1; row_i < n_rows;row_i++){
		//get row i
		//printf("row i = %d\n",row_i);

		start_ri = idx1[row_i];
		end_ri = idx1[row_i+1];
		int nnz_i = 0;
		int temp_nnz_i = 0;
		int current_row_ind = 0;

		//initialize temporary levls and idx2 vectors for row i
		for(int j = start_ri; j < end_ri; j++){
			current_row_levls.push_back(0);
			current_row_idx2.push_back(idx2[j]);
			nnz_i++;
		}

		//printf("Initialize temp levls and idx2 for row i\n");
		/*
		printf("current row Levls = ");
    	for (auto i: current_row_levls)
			printf("%d ",i);
		printf("\n");
  	
		printf("current row idx2 = ");
    	for (auto i: current_row_idx2)
			printf("%d ",i);
		printf("\n");
  		
		printf("\n");
		*/

		for(int j = start_ri; j < end_ri; j++){
			//printf("j = %d\n",j);

			int col_k = idx2[j];
			
			//printf("col k = %d\n", col_k);
			//since Gaussian elimination k loop goes from 0 to i-1
			if(col_k >= row_i){
				//printf("exit k loop\n");
				break;
			}
			
			//Add multiplier levels to final matrix 
			//printf("multiplier\n");
			/*
			printf("current row Levls = ");
    		for (auto i: current_row_levls)
				printf("%d ",i);
			printf("\n");
  	
			printf("current row idx2 = ");
    		for (auto i: current_row_idx2)
				printf("%d ",i);
			printf("\n");
  		
			printf("\n");
			*/
			
			levls->vals.push_back(current_row_levls[current_row_ind]);
			levls->idx2.push_back(current_row_idx2[current_row_ind]);
			/*
			printf("Levls idx1 = ");
    		for (auto i: levls->idx1)
				printf("%d ",i);
			printf("\n");
  
			printf("Levls idx2 = ");
    		for (auto i: levls->idx2)
				printf("%d ",i);
			printf("\n");
  	
			printf("Levls vals = ");
    		for (auto i: levls->vals)
				printf("%e ",i);
			printf("\n");
			*/
			current_row_ind ++;
			levls->nnz = levls->nnz + 1;		


			//get row k from updated levls matrix
			int start_rk = levls->idx1[col_k];
			int end_rk = levls->idx1[col_k+1];
			
			//temp variables
			int ind_i = -1;
			int ind_k = -1;
			
			int col_ij = -1;
			int col_kj = -1;
			int col_ik = col_k;

			int lev_ij = 100000;
			int lev_ik = 100000;
			int lev_kj = 100000;
			
			//printf("Update rest\n");
			//get starting indices for rows i and k such that col >=k+1
			for(int i = 0; i < nnz_i; i++){
				int col = current_row_idx2[i];
				if(col == col_k)
					lev_ik = current_row_levls[i];
				if(col >= col_k+1){
					ind_i = i;
					col_ij = current_row_idx2[i];
					break;
				}
			}


			for(int i = start_rk; i < end_rk; i++){
				int col = levls->idx2[i];
				if(col >=col_k+1){
					ind_k = i;
					col_kj = levls->idx2[i];
					break;
				}
			}

			//check if reached end of row k
			if(ind_k == -1)
				col_kj = n_rows + 1;

			int current_col = -1;
			int current_row = -1;
			//printf("\n");
			//printf("indices before while loop");
			//printf("ind i = %d\n",ind_i);
			//printf("ind k = %d\n",ind_k);
			//printf("\n");

			while(1){
				//printf("While iteration \n");
				//printf("col ij = %d\n",col_ij);
				//printf("col ik = %d\n",col_ik);
				//printf("col kj = %d\n",col_kj);
				//printf("ind i = %d\n",ind_i);
				//printf("ind k = %d\n",ind_k);
				//printf("\n");
				/*
				printf("temp row idx2 = ");
    			for (auto i: temp_row_idx2)
					printf("%d ",i);
				printf("\n");
  	
				printf("temp row levls = ");
    			for (auto i: temp_row_levls)
					printf("%d ",i);
				printf("\n");
  		
				printf("\n");


				printf("current row idx2 = ");
    			for (auto i: current_row_idx2)
					printf("%d ",i);
				printf("\n");
  	
				printf("current row levls = ");
    			for (auto i: current_row_levls)
					printf("%d ",i);
				printf("\n");
  		
				printf("\n");
				*/

		
				if((col_ij >= n_rows) and (col_kj >= n_rows)){
					//printf("exit j loop\n");
					break;
				}

				if(col_ij == col_kj){
					//printf("col(i,j) = col(k,j), equal\n");
					current_col = col_ij;
					current_row = row_i;

					lev_ij = current_row_levls[ind_i];
					//printf("lev(i,j) = %d\n",lev_ij);
					lev_kj = levls->vals[ind_k];

					//printf("lev(k,j) = %d\n",lev_kj);
					lev_ij = min(lev_ij, lev_ik+lev_kj+1);

					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_i++;
					ind_k++;
				}

				else if(col_ij<col_kj){
					//printf("col(i,j) < col(k,j)\n");
					current_col=col_ij;
					current_row=row_i;

					lev_ij = current_row_levls[ind_i];

					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_i++;
				}

				else if(col_ij>col_kj){
					//printf("col(i,j) > col(k,j)\n");
					current_col = col_kj;
					current_row = col_k;
					
					lev_kj = levls->vals[ind_k];

					//printf("lev(k,j) = %d\n",lev_kj);
					lev_ij = lev_ik + lev_kj + 1;
					
					//printf("updated lev(i,j) = %d\n",lev_ij);
					if(lev_ij < 10000){
						temp_row_levls.push_back(lev_ij);
						temp_row_idx2.push_back(current_col);
						temp_nnz_i++;
					}

					ind_k++;
				}

				//printf("ind i = %d\n",ind_i);
				//printf("nnz i = %d\n",nnz_i);
				if((ind_i < nnz_i) && (ind_i != -1))
					col_ij=current_row_idx2[ind_i];
				else
					col_ij = n_rows+1;

				//printf("ind k = %d\n",ind_k);
				//printf("end rk = %d\n",end_rk);
				if((ind_k < end_rk) && (ind_k != -1))
					col_kj=levls->idx2[ind_k];
				else
					col_kj = n_rows+1;
				/*
				printf("temp row levls = ");
    			for (auto i: temp_row_levls)
					printf("%d ",i);
				printf("\n");
  	
				printf("temp row idx2 = ");
    			for (auto i: temp_row_idx2)
					printf("%d ",i);
				printf("\n");
				*/
			}
			
			current_row_levls = temp_row_levls;
			current_row_idx2 = temp_row_idx2;
			nnz_i = temp_nnz_i;
			
			temp_row_levls.clear();
			temp_row_idx2.clear();
			temp_nnz_i = 0; 	
		}
		
		//Add current row levels to final levls matrix
		for(int i = 0; i<nnz_i;i++){
			levls->idx2.push_back(current_row_idx2[i]);
			levls->vals.push_back(current_row_levls[i]);
			levls->nnz = levls->nnz + 1;
			//printf("First row: Added %lf at 0,%d\n",val,col);
		}

		current_row_levls.clear();
		current_row_idx2.clear();

		levls->idx1[row_i+1] = levls->nnz;

	}//end i loop
 	
/*	
	printf("Levls rowptr = ");
    for (auto i: levls->idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("Levls cols = ");
    for (auto i: levls->idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("Levls data = ");
    for (auto i: levls->vals)
		printf("%.2f ",i);
	printf("\n");
*/	
	return levls;    
}


Matrix* BSRMatrix::ilu_sparsity(Matrix* levls, int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

Matrix* BSRMatrix::ilu_symbolic(int lof)
{
	printf("Function not implemented \n");
	return NULL;
}

std::vector<double>& BSRMatrix::ilu_numeric(Matrix* sparsity)
{
	printf("Function not implemented \n");
	std::vector<double> emp;
	return emp;
}



/*
void CSRMatrix::ilu_k(int lof,COOMatrix* levls)
{
	COOMatrix* temp_levls = new COOMatrix(n_rows,n_cols); 

   	
	printf("A csr rowptr = ");
    for (auto i: idx1)
		printf("%d ",i);
	printf("\n");
  
	printf("A csr cols = ");
    for (auto i: idx2)
		printf("%d ",i);
	printf("\n");
  	
	printf("A csr data = ");
    for (auto i: vals)
		printf("%e ",i);
	printf("\n");
    
		
	//get vector of diagonal elements
	Vector diag_vec(n_rows);

	diag_vec.set_const_value(0.0);

	for(int row = 0; row < n_rows; row++){
		int start_i = idx1[row];
		int end_i = idx1[row+1];
		for(int j = start_i; j < end_i; j++){
			int col = idx2[j];
			double val = vals[j];
			if(row==col){
				diag_vec.values[row] = val;
			}
		}
	}

	//copy first row into new matrix
	int start_r = idx1[0];
	int end_r = idx1[1];
	for(int i = start_r; i<end_r;i++){
		int col = idx2[i];
		double val = vals[i];
		levls->add_value(0,col,val);
		//printf("First row: Added %lf at 0,%d\n",val,col);
	}

	//printf("Function \n");
	for(int row_i = 1; row_i < n_rows;row_i++){
		//get row i
		int start_ri = idx1[row_i];
		int end_ri = idx1[row_i+1];
		for(int j = start_ri; j < end_ri; j++){
			int col_k = idx2[j];
			
			if(col_k >= row_i)
				break;
			//a(i,k)=a(i,k)/a(k,k)
			//printf("\n");
			//printf("a(i,k)=a(i,k)/a(k,k)\n");
			//printf("a(i,k)=%e\n",vals[j]);
			//printf("a(k,k)=%e\n",diag_vec.values[col_k]);

			vals[j] /= diag_vec.values[col_k];
			double temp = vals[j];

			levls->add_value(row_i,col_k,temp);
			
			//printf("multiplier: Added %e at %d,%d\n",temp,row_i,col_k);
			
			//check if it's a diagonal element, then add it to diag vector
			if(row_i==col_k)
				diag_vec.values[row_i] = temp;
			
			//get row k
			int start_rk = idx1[col_k];
			int end_rk = idx1[col_k+1];
			
			int ind_i = -1;
			int ind_k = -1;
			
			int col_ij = -1;
			int col_kj = -1;
			
			//get starting indices for rows i and k such that col >=k+1
			for(int i = start_ri; i < end_ri; i++){
				int col = idx2[i];
				if(col >=col_k+1){
					ind_i = i;
					col_ij = idx2[i];
					break;
				}
			}


			for(int i = start_rk; i < end_rk; i++){
				int col = idx2[i];
				if(col >=col_k+1){
					ind_k = i;
					col_kj = idx2[i];
					break;
				}
			}

			double current_it =0 ;
			int current_col = -1;
			int current_row = -1;

			while(1){
				if((col_ij>=n_rows) and (col_kj>=n_rows))
					break;

				if(col_ij==col_kj){
					current_col=col_ij;
					current_row=row_i;
					current_it = vals[ind_i]-temp*vals[ind_k];		
					
					vals[ind_i]=current_it;
					
					levls->add_value(current_row,current_col,current_it);
					
					//printf("Equal: Added %e at %d,%d\n",current_it,current_row,current_col);
					
					//check if it's a diagonal element, then add it to diag vector
					if(current_row==current_col)
						diag_vec.values[current_row] = current_it;
		
					ind_i++;
					ind_k++;
				}

				else if(col_ij<col_kj){
					current_col=col_ij;
					current_row=row_i;
					current_it = vals[ind_i];
					
					vals[ind_i]=current_it;

					levls->add_value(current_row,current_col,current_it);

					//printf("col i < k: Added %e at %d,%d\n",current_it,current_row,current_col);
					
					//check if it's a diagonal element, then add it to diag vector
					if(current_row==current_col)
						diag_vec.values[current_row] = current_it;
		
					ind_i++;
				}

				else if(col_ij>col_kj){
					current_col = col_kj;
					current_row = col_k;
					current_it = -temp*vals[ind_k];
					levls->add_value(current_row, current_col,current_it);
					
					//printf("col j > k: Added %e at %d,%d\n",current_it,current_row,current_col);

					//check if it's a diagonal element, then add it to diag vector
					if(current_row==current_col)
						diag_vec.values[current_row] = current_it;
	
					ind_k++;
				}
				if(ind_i<end_ri)
					col_ij=idx2[ind_i];
				else
					col_ij = n_rows+1;

				if(ind_k<end_rk)
					col_kj=idx2[ind_k];
				else
					col_kj = n_rows+1;

			}
	
		}
	}

}
*/



