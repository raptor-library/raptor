// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "raptor/core/par_matrix.hpp"

using namespace raptor;

// Declare Private Methods
ParCSRMatrix* init_mat(ParCSCMatrix* A);
ParCSRMatrix* init_mat(ParCSRMatrix* A);
ParBSRMatrix* init_mat(ParBSRMatrix* A);
ParBSRMatrix* init_mat(ParBSCMatrix* A);

ParCSRMatrix* init_mat(ParCSCMatrix* A)
{
    return new ParCSRMatrix(A->partition);
}
ParCSRMatrix* init_mat(ParCSRMatrix* A)
{
    return new ParCSRMatrix(A->partition);
}
template <typename T>
ParCSRMatrix* init_mat(ParCSRMatrix* A, T* B)
{
    Partition* part = new Partition(A->partition, B->partition);
    ParCSRMatrix* C = new ParCSRMatrix(part);
    part->num_shared = 0;
    return C;
}
ParBSRMatrix* init_mat(ParBSRMatrix* A)
{
    return new ParBSRMatrix(A->partition, A->on_proc->b_rows, A->on_proc->b_cols);
}
ParBSRMatrix* init_mat(ParBSCMatrix* A)
{
    return new ParBSRMatrix(A->partition, A->on_proc->b_rows, A->on_proc->b_cols);
}
template <typename T>
ParBSRMatrix* init_mat(ParBSRMatrix* A, T* B)
{
    Partition* part = new Partition(A->partition, B->partition);
    ParBSRMatrix* C = new ParBSRMatrix(part, A->on_proc->b_rows, A->on_proc->b_cols);
    part->num_shared = 0;
    return C;
}
template <typename T, typename U>
ParCSRMatrix* init_matrix(T* A, U* B)
{
    ParCSRMatrix* C;

    if (A->partition == B->partition)
    {
        C = init_mat(A); 
    }
    else
    {
        if (A->partition->global_num_rows == B->partition->global_num_rows &&
            A->partition->local_num_rows == B->partition->local_num_rows &&
            A->partition->first_local_row == B->partition->first_local_row &&
            A->partition->last_local_row == B->partition->last_local_row)
        {
            C = init_mat(B);
        }
        else if (A->partition->global_num_cols == B->partition->global_num_cols &&
            A->partition->local_num_cols == B->partition->local_num_cols &&
            A->partition->first_local_col == B->partition->first_local_col &&
            A->partition->last_local_col == B->partition->last_local_col)
        {
            C = init_mat(A);
        }
        else
        {
            C = init_mat(A, B);
        }
    }

    return C;
}

ParCSRMatrix* ParCSRMatrix::mult(ParCSRMatrix* B, bool tap)
{
    if (tap)
    {
        return this->tap_mult(B);
    }

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = init_matrix(this, B);
    std::vector<char> send_buffer;

    // Communicate data and multiply
    comm->init_par_mat_comm(B, send_buffer);

    // Fully Local Computation
    CSRMatrix* C_on_on = on_proc->mult((CSRMatrix*) B->on_proc);
    CSRMatrix* C_on_off = on_proc->mult((CSRMatrix*) B->off_proc);

    CSRMatrix* recv_mat = comm->complete_mat_comm();

    mult_helper(B, C, recv_mat, C_on_on, C_on_off);

    delete C_on_on;
    delete C_on_off;
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::tap_mult(ParCSRMatrix* B)
{
    // Check that communication package has been initialized
    if (tap_mat_comm == NULL)
    {
        // Always 2-step
        tap_mat_comm = new TAPComm(partition, off_proc_column_map, 
                on_proc_column_map, false);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = init_matrix(this, B);;
    std::vector<char> send_buffer;

    // Communicate data and multiply
    tap_mat_comm->init_par_mat_comm(B, send_buffer);

    // Fully Local Computation
    CSRMatrix* C_on_on = on_proc->mult((CSRMatrix*) B->on_proc);
    CSRMatrix* C_on_off = on_proc->mult((CSRMatrix*) B->off_proc);

    CSRMatrix* recv_mat = tap_mat_comm->complete_mat_comm();

    mult_helper(B, C, recv_mat, C_on_on, C_on_off);
    delete C_on_on;
    delete C_on_off;
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult_T(ParCSRMatrix* A, bool tap)
{
    ParCSCMatrix* Acsc = A->to_ParCSC();
    ParCSRMatrix* C = this->mult_T(Acsc, tap);
    delete Acsc;
    return C;
}

ParCSRMatrix* ParCSRMatrix::tap_mult_T(ParCSRMatrix* A)
{
    ParCSCMatrix* Acsc = A->to_ParCSC();
    ParCSRMatrix* C = this->tap_mult_T(Acsc);
    delete Acsc;
    return C;
}

ParCSRMatrix* ParCSRMatrix::mult_T(ParCSCMatrix* A, bool tap)
{
    if (tap)
    {
        return this->tap_mult_T(A);
    }

    if (A->comm == NULL)
    {
        A->comm = new ParComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = init_matrix(this, A);;

    CSRMatrix* Ctmp = mult_T_partial(A);
    std::vector<char> send_buffer;

    A->comm->init_mat_comm_T(send_buffer, Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals);

    CSRMatrix* C_on_on = on_proc->mult_T((CSCMatrix*) A->on_proc);
    CSRMatrix* C_off_on = off_proc->mult_T((CSCMatrix*) A->on_proc);

    CSRMatrix* recv_mat = A->comm->complete_mat_comm_T(A->on_proc_num_cols);

    mult_T_combine(A, C, recv_mat, C_on_on, C_off_on);

    // Clean up
    delete Ctmp;
    delete C_on_on;
    delete C_off_on;
    delete recv_mat;

    // Return matrix containing product
    return C;
}

ParCSRMatrix* ParCSRMatrix::tap_mult_T(ParCSCMatrix* A)
{
    if (A->tap_mat_comm == NULL)
    {
        A->tap_mat_comm = new TAPComm(A->partition, A->off_proc_column_map, 
                A->on_proc_column_map, false);
    }

    // Initialize C (matrix to be returned)
    ParCSRMatrix* C = init_matrix(this, A);

    CSRMatrix* Ctmp = mult_T_partial(A);
    std::vector<char> send_buffer;

    A->tap_mat_comm->init_mat_comm_T(send_buffer, Ctmp->idx1, Ctmp->idx2, 
            Ctmp->vals);

    CSRMatrix* C_on_on = on_proc->mult_T((CSCMatrix*) A->on_proc);
    CSRMatrix* C_off_on = off_proc->mult_T((CSCMatrix*) A->on_proc);

    CSRMatrix* recv_mat = A->tap_mat_comm->complete_mat_comm_T(A->on_proc_num_cols);

    mult_T_combine(A, C, recv_mat, C_on_on, C_off_on);

    // Clean up
    delete Ctmp;
    delete recv_mat;
    delete C_on_on;
    delete C_off_on;

    // Return matrix containing product
    return C;
}

ParMatrix* ParMatrix::mult(ParCSRMatrix* B, bool tap)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    if (rank == 0) 
        printf("Multiplication is not implemented for these ParMatrix types.\n");
    return NULL;
}

void ParCSRMatrix::mult_helper(ParCSRMatrix* B, ParCSRMatrix* C, 
        CSRMatrix* recv_mat, CSRMatrix* C_on_on, CSRMatrix* C_on_off)
{
    // Set dimensions of C
    C->global_num_rows = global_num_rows;
    C->global_num_cols = B->global_num_cols;
    C->local_num_rows = local_num_rows;

    C->on_proc_column_map = B->get_on_proc_column_map();
    C->local_row_map = get_local_row_map();
    C->on_proc_num_cols = C->on_proc_column_map.size();
    
    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    // Declare Variables
    int row_start, row_end;
    int global_col;
            
    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(recv_mat->n_rows, -1);
    CSRMatrix* recv_off = new CSRMatrix(recv_mat->n_rows, -1);

	auto part_to_col = B->map_partition_to_local();
    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        row_start = recv_mat->idx1[i];
        row_end = recv_mat->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = recv_mat->idx2[j];
            if (global_col < B->partition->first_local_col ||
                    global_col > B->partition->last_local_col)
            {
                recv_off->idx2.emplace_back(global_col);
                recv_off->vals.emplace_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on->idx2.emplace_back(part_to_col[global_col - 
                        B->partition->first_local_col]);
                recv_on->vals.emplace_back(recv_mat->vals[j]);
            }
        }
        recv_on->idx1[i+1] = recv_on->idx2.size();
        recv_off->idx1[i+1] = recv_off->idx2.size();
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();

    // Calculate global_to_C and B_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> B_to_C(B->off_proc_num_cols);

    std::copy(recv_off->idx2.begin(), recv_off->idx2.end(),
            std::back_inserter(C->off_proc_column_map));
    for (std::vector<int>::iterator it = B->off_proc_column_map.begin();
            it != B->off_proc_column_map.end(); ++it)
    {
        C->off_proc_column_map.emplace_back(*it);
    }
    std::sort(C->off_proc_column_map.begin(), C->off_proc_column_map.end());

    int prev_col = -1;
    C->off_proc_num_cols = 0;
    for (std::vector<int>::iterator it = C->off_proc_column_map.begin();
            it != C->off_proc_column_map.end(); ++it)
    {
        if (*it != prev_col)
        {
            global_to_C[*it] = C->off_proc_num_cols;
            C->off_proc_column_map[C->off_proc_num_cols++] = *it;
            prev_col = *it;
        }
    }
    C->off_proc_column_map.resize(C->off_proc_num_cols);

    for (int i = 0; i < B->off_proc_num_cols; i++)
    {
        global_col = B->off_proc_column_map[i];
        B_to_C[i] = global_to_C[global_col];
    }
    for (std::vector<int>::iterator it = recv_off->idx2.begin(); 
            it != recv_off->idx2.end(); ++it)
    {
        *it = global_to_C[*it];
    }

    for (std::vector<int>::iterator it = C_on_off->idx2.begin();
            it != C_on_off->idx2.end(); ++it)
    {
        *it = B_to_C[*it];
    }
    C->off_proc_num_cols = C->off_proc_column_map.size();
    recv_on->n_cols = B->on_proc->n_cols;
    recv_off->n_cols = C->off_proc_num_cols;
    C_on_off->n_cols = C->off_proc_num_cols;

    // Multiply A->off_proc * B->recv_on -> C_off_on
    CSRMatrix* C_off_on = off_proc->mult(recv_on);
    delete recv_on;

    // Multiply A->off_proc * B->recv_off -> C_off_off
    CSRMatrix* C_off_off = off_proc->mult(recv_off);
    delete recv_off;

    // Create C->on_proc by adding C_on_on + C_off_on
    C_on_on->add_append(C_off_on, (CSRMatrix*) C->on_proc);
    delete C_off_on;

    // Create C->off_proc by adding C_off_on + C_off_off
    C_on_off->add_append(C_off_off, (CSRMatrix*) C->off_proc);
    delete C_off_off;

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;
}

CSRMatrix* ParCSRMatrix::mult_T_partial(CSCMatrix* A_off)
{
    CSRMatrix* C_off_on = on_proc->mult_T(A_off, on_proc_column_map.data());
    CSRMatrix* C_off_off = off_proc->mult_T(A_off, off_proc_column_map.data());
    CSRMatrix* Ctmp = C_off_on->add(C_off_off, false);

    delete C_off_on;
    delete C_off_off;

    return Ctmp;
}

// A_T * self
CSRMatrix* ParCSRMatrix::mult_T_partial(ParCSCMatrix* A)
{
    // Declare Variables
    return mult_T_partial((CSCMatrix*) A->off_proc); 
}

void ParCSRMatrix::mult_T_combine(ParCSCMatrix* P, ParCSRMatrix* C, CSRMatrix* recv_mat,
        CSRMatrix* C_on_on, CSRMatrix* C_off_on)
{ 
    int start, end, ctr;
    int col, col_C;

    std::vector<double> sums;
    std::vector<int> next;

    // Split recv_mat into recv_on and recv_off
    // Split recv_mat into on and off proc portions
    CSRMatrix* recv_on = new CSRMatrix(recv_mat->n_rows, -1);
    CSRMatrix* recv_off = new CSRMatrix(recv_mat->n_rows, -1);
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < partition->first_local_col
                    || col > partition->last_local_col)
            {
                recv_off->idx2.emplace_back(col);
                recv_off->vals.emplace_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on->idx2.emplace_back(col);
                recv_on->vals.emplace_back(recv_mat->vals[j]);
            }
        }
        recv_on->idx1[i+1] = recv_on->idx2.size();
        recv_off->idx1[i+1] = recv_off->idx2.size();
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();


    // Set dimensions of C
    C->global_num_rows = P->global_num_cols; // AT global rows
    C->global_num_cols = global_num_cols;
    C->local_num_rows = P->on_proc_num_cols; // AT local rows

    // Initialize nnz as 0 (will increment this as nonzeros are added)
    C->local_nnz = 0;

    /******************************
     * Form on_proc
     ******************************/
    // Resize variables in on_proc
    C->on_proc_column_map = get_on_proc_column_map();
    C->local_row_map = P->get_on_proc_column_map();
    C->on_proc_num_cols = C->on_proc_column_map.size();

    // Update recv_on columns (to match local cols)
    auto part_to_col = map_partition_to_local();
    for (std::vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        *it = part_to_col[(*it - partition->first_local_col)];
    }

    // Multiply on_proc    
    recv_on->n_cols = C->on_proc_num_cols;
    C_on_on->add_append(recv_on, (CSRMatrix*) C->on_proc);

    /******************************
     * Form off_proc
     ******************************/
    // Calculate global_to_C and map_to_C column maps
    std::map<int, int> global_to_C;
    std::vector<int> map_to_C;
    if (off_proc_num_cols)
    {
        map_to_C.reserve(off_proc_num_cols);
    }

    // Create set of global columns in B_off_proc and recv_mat
    std::set<int> C_col_set;
    for (std::vector<int>::iterator it = recv_off->idx2.begin(); 
            it != recv_off->idx2.end(); ++it)
    {
        C_col_set.insert(*it);
    }
    for (std::vector<int>::iterator it = off_proc_column_map.begin(); 
            it != off_proc_column_map.end(); ++it)
    {
        C_col_set.insert(*it);
    }

    C->off_proc_num_cols = C_col_set.size();
    if (C->off_proc_num_cols)
    {
        C->off_proc_column_map.reserve(C->off_proc_num_cols);
    }
    for (std::set<int>::iterator it = C_col_set.begin(); 
            it != C_col_set.end(); ++it)
    {
        global_to_C[*it] = C->off_proc_column_map.size();
        C->off_proc_column_map.emplace_back(*it);
    }

    // Map local off_proc_cols to C->off_proc_column_map
    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        col_C = global_to_C[*it];
        map_to_C.emplace_back(col_C);
    }

    // Update recvd cols from global_col to local col in C
    for (std::vector<int>::iterator it = recv_off->idx2.begin();
            it != recv_off->idx2.end(); ++it)
    {
        *it = global_to_C[*it];
    }

    recv_off->n_cols = C->off_proc_num_cols;
    for (std::vector<int>::iterator it = C_off_on->idx2.begin();
            it != C_off_on->idx2.end(); ++it)
    {
        *it = map_to_C[*it];
    }
    C_off_on->add_append(recv_off, (CSRMatrix*) C->off_proc);

    C->local_nnz = C->on_proc->nnz + C->off_proc->nnz;

    // Condense columns!  A lot of them are zero columns...
    // Could instead add global column indices, and then map to local
    std::vector<int> off_col_sizes;
    std::vector<int> col_orig_to_new;
    if (C->off_proc_num_cols)
    {
        off_col_sizes.resize(C->off_proc_num_cols, 0);
        col_orig_to_new.resize(C->off_proc_num_cols);
    }
    for (int i = 0; i < C->local_num_rows; i++)
    {
        start = C->off_proc->idx1[i];
        end = C->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            off_col_sizes[C->off_proc->idx2[j]]++;
        }
    }
    ctr = 0;
    for (int i = 0; i < C->off_proc_num_cols; i++)
    {
        if (off_col_sizes[i])
        {
            col_orig_to_new[i] = ctr;
            C->off_proc_column_map[ctr++] = C->off_proc_column_map[i];
        }
    }
    C->off_proc_num_cols = ctr;
    C->off_proc->n_cols = ctr;
    if (ctr)
    {
        C->off_proc_column_map.resize(ctr);
    }
    else
    {
        C->off_proc_column_map.clear();
    }
    for (int i = 0; i < C->local_num_rows; i++)
    {
        start = C->off_proc->idx1[i];
        end = C->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = C->off_proc->idx2[j];
            C->off_proc->idx2[j] = col_orig_to_new[col];
        }
    }

    delete recv_on;
    delete recv_off;
}

