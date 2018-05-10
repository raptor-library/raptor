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

std::vector<double>& COOMatrix::ilu_numeric(Matrix* levls)
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
	int adj_lof = lof + 1;

	Matrix* sparsity = new CSRMatrix(levls->n_rows,levls->n_cols);

	sparsity->n_rows = levls->n_rows;
	sparsity->n_cols = levls->n_cols;
	int sparsity_nnz = 0;
	sparsity->nnz = 0;
	int sparsity_nnz_dense = sparsity->n_rows*sparsity->n_cols;

	sparsity->idx1.resize(levls->n_rows+1);
	if(sparsity_nnz_dense){
		sparsity->idx2.reserve(sparsity_nnz_dense);
		sparsity->vals.reserve(sparsity_nnz_dense);
	}
	
	for(int row_i = 0; row_i<sparsity->n_rows;row_i++){
		sparsity->idx1[row_i] = sparsity->nnz;
		int start_ri = levls->idx1[row_i];
		int end_ri = levls->idx1[row_i+1];
		for(int j = start_ri; j<end_ri;j++){
			int col_j = levls->idx2[j];
			int levl_j = levls->vals[j];
			if(levl_j <= lof){
				sparsity->idx2[sparsity_nnz] = (col_j);
				sparsity->vals[sparsity_nnz] = (levl_j);
				sparsity_nnz ++;
				sparsity->nnz = sparsity_nnz;
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

std::vector<double>& CSRMatrix::ilu_numeric(Matrix* levls){

	int m = n_rows;
	int n = n_cols;

	std::vector<double> factors_data(levls->nnz);
	std::fill(factors_data.begin(), factors_data.end(), 0.0);
/*	
	printf("factors data = ");
	for(auto i: factors)
		printf("%.2f ",i);
	printf("\n");
*/
	//initialize entries in factors to entries in A
	int index_i = 0;
	for(int i = 0; i<levls->nnz;i++){
		if(levls->vals[i] - 1< 1){
			factors_data[i] = vals[index_i];
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
	std::vector<double> diag_vec(n_rows);
	std::fill(diag_vec.begin(),diag_vec.end(),0.0);
	
	for(int row = 0; row < n_rows; row++){
		int start_i = levls->idx1[row];
		int end_i = levls->idx1[row+1];
		for(int j = start_i; j<end_i; j++){
			int col = levls->idx2[j];
			double val = factors_data[j];
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
		std::vector<int> current_row_levls(n_rows);
		std::fill(current_row_levls.begin(),current_row_levls.end(),0);

		std::vector<double> current_row_factors(n_rows);
		std::fill(current_row_factors.begin(),current_row_factors.end(),0.0);
		
		int start_ri = levls->idx1[row_i];
		int end_ri = levls->idx1[row_i+1];

		for(int jj=start_ri;jj<end_ri;jj++){
			int col = levls->idx2[jj];
			current_row_levls[col] = levls->vals[jj];
			current_row_factors[col] = factors_data[jj];
		}

		for(int k = 0; k<row_i;k++){
			//compute multiplier
			if(current_row_levls[k]!=0){
				current_row_factors[k]=current_row_factors[k]/diag_vec[k];
				if(k==row_i)
					diag_vec[k] = current_row_factors[k];
			}

			//get row k
			int start_rk = levls->idx1[k];
			int end_rk = levls->idx1[k+1];
			std::vector<double> row_k_factors(n_rows);
			std::fill(row_k_factors.begin(),row_k_factors.end(),0.0);

			for(int jj=start_rk;jj<end_rk;jj++){
				int col = levls->idx2[jj];
				row_k_factors[col]=factors_data[jj];
			}

			for(int j=k+1;j<n_rows;j++){
				if(current_row_levls[j] != 0){
					current_row_factors[j] = current_row_factors[j]-current_row_factors[k]*row_k_factors[j];
					if(j==row_i)
						diag_vec[j] = current_row_factors[j];
				}
			}
		}

		for(int jj=start_ri;jj<end_ri;jj++){
			int col = levls->idx2[jj];
			factors_data[jj] = current_row_factors[col];
		}
	}
	
	/*
	printf("factors = ");
    for (auto i: factors)
		printf("%.2f ",i);
	printf("\n");
 
 	printf("ilu numeric phase done\n");
	printf("\n");
	*/
	return factors_data;
}

Matrix* CSRMatrix::ilu_levels()
{
	//printf("Begin ilu levels \n");
	Matrix * levls = new CSRMatrix(n_rows,n_cols);

	//initialize vectors for final levels matrix
	levls->n_rows = n_rows;
	levls->n_cols = n_cols;
	int levls_nnz = 0;
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
		levls->idx2[levls_nnz] = idx2[i];
		levls->vals[levls_nnz] = 1;
		levls_nnz ++;
		levls->nnz = levls_nnz;
		//printf("First row: Added %lf at 0,%d\n",val,col);
	}
	levls->idx1[1] = levls_nnz;
	
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
	
	//begin ILU process
	for(int row_i = 1; row_i < n_rows;row_i++){
		std::vector<int> current_row_levls(n_rows);
		std::fill(current_row_levls.begin(), current_row_levls.end(), 100);
		//get row i
		//printf("row i = %d\n",row_i);

		int start_ri = idx1[row_i];
		int end_ri = idx1[row_i+1];

		//initialize temporary levls and idx2 vectors for row i
		for(int jj = start_ri; jj < end_ri; jj++){
			current_row_levls[idx2[jj]] = 1;
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
		
		for(int k = 0; k <row_i; k++){
			//get row k
			int start_rk = levls->idx1[k];
			int end_rk = levls->idx1[k+1]; 
			std::vector<int> row_k_levls(n_rows);
			std::fill(row_k_levls.begin(), current_row_levls.end(), 100);
	
			for(int t =start_rk;t<end_rk;t++)
				row_k_levls[levls->idx2[t]] = levls->vals[t];

			for(int j = k+1; k<n_rows; k++)
				current_row_levls[j] = min(current_row_levls[j], current_row_levls[k]+row_k_levls[j]);
		}

		for(int jj=0; jj<n_rows;jj++){
			if(current_row_levls[jj]<100){
				levls->idx2[levls_nnz] = jj;
				levls->vals[levls_nnz] = current_row_levls[jj];
				levls_nnz++;
				levls->nnz = levls->nnz + 1;
			}
		}

	}
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

std::vector<double>& CSCMatrix::ilu_numeric(Matrix* levls)
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
	//std::vector <int>  dummy_row_levls;
	//std::vector <int>  dummy_row_idx2;
/////////////////////////////////////////
//////////// ASK AMANDA HOW TO ACCESS NNZ PER ROW
	current_row_levls.reserve(n_cols);
	current_row_idx2.reserve(n_cols);
	temp_row_levls.reserve(n_cols);
	temp_row_idx2.reserve(n_cols);
	//dummy_row_levls.reserve(n_cols);
	//dummy_row_idx2.reserve(n_cols);

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

std::vector<double>& BSRMatrix::ilu_numeric(Matrix* levls)
{
	printf("Function not implemented \n");
	std::vector<double> emp;
	return emp;
}



