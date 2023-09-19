// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"

void compare_mats(CSRMatrix* A, CSRMatrix* B)
{
    if (A->n_rows != B->n_rows)
    {
        printf("NRows %d vs %d\n", A->n_rows, B->n_rows);
        return;
    }

    for (int i = 0; i < A->n_rows; i++)
    {
        if (A->idx1[i+1] != B->idx1[i+1])
        {
            printf("Idx1[%d] : %d vs %d\n", i+1, 
                    A->idx1[i+1], B->idx1[i+1]);
            return;
        }
        for (int j = A->idx1[i]; j < A->idx1[i+1]; j++)
        {
            if (A->idx2[j] != B->idx2[j])
                printf("Idx2[%d] : %d vs %d\n", j,
                        A->idx2[j], B->idx2[j]);
            if (fabs(A->vals[j] - B->vals[j]) > 1e-10)
                printf("Vals[%d] : %e vs %e\n", j,
                        A->vals[j], B->vals[j]);
        }
    }
}

void compare_mats(ParCSRMatrix* A, ParCSRMatrix* B)
{
    if (A->local_num_rows != B->local_num_rows)
    {
        printf("Local num rows %d vs %d\n", 
                A->local_num_rows, B->local_num_rows);
        return;
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (A->on_proc->idx1[i+1] != B->on_proc->idx1[i+1])
        {
            printf("OnProc idx1[%d] : %d vs %d\n", i+1,
                    A->on_proc->idx1[i+1], B->on_proc->idx1[i+1]);
            return;
        }
        for (int j = A->on_proc->idx1[i]; j < A->on_proc->idx1[i+1]; j++)
        {
            if (A->on_proc->idx2[j] != B->on_proc->idx2[j])
                printf("OnProc idx2[%d] : %d vs %d\n", j, 
                        A->on_proc->idx2[j], B->on_proc->idx2[j]);
            if (fabs(A->on_proc->vals[j] - B->on_proc->vals[j]) > 1e-10)
                printf("OnProc vals[%d] : %e vs %e\n", j,
                        A->on_proc->vals[j], B->on_proc->vals[j]);
        }

        if (A->off_proc->idx1[i+1] != B->off_proc->idx1[i+1])
        {
            printf("OffProc idx1[%d] : %d vs %d\n", i+1,
                    A->off_proc->idx1[i+1], B->off_proc->idx1[i+1]);
            return;
        }
        for (int j = A->off_proc->idx1[i]; j < A->off_proc->idx1[i+1]; j++)
        {
            if (A->off_proc->idx2[j] != B->off_proc->idx2[j])
                printf("OffProc idx2[%d] : %d vs %d\n", j, 
                        A->off_proc->idx2[j], B->off_proc->idx2[j]);
            if (fabs(A->off_proc->vals[j] - B->off_proc->vals[j]) > 1e-10)
                printf("OffProc vals[%d] : %e vs %e\n", j,
                        A->off_proc->vals[j], B->off_proc->vals[j]);
        }

    }
}

int pack_mat(ParCSRMatrix* A, std::vector<char>& packed_array, MPI_Comm comm)
{
    int rank; MPI_Comm_rank(comm, &rank);
    int nnz = A->on_proc->nnz + A->off_proc->nnz;

    int int_bytes, double_bytes;
    MPI_Pack_size(1, MPI_INT, comm, &int_bytes);
    MPI_Pack_size(1, MPI_DOUBLE, comm, &double_bytes);

    packed_array.resize((2*A->local_num_rows + nnz) * int_bytes + 
            nnz * double_bytes);

    // Pack matrix (combining on_proc and off_proc)
    int start_on, end_on, start_off, end_off;
    int row_size, col, global_col, global_row;
    double val;
    int ctr = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        // Calculate on_proc and off_proc sizes
        // Need to pack global rows indices, because coarse level idx aren't contiguous
        global_row = A->local_row_map[i];
        MPI_Pack(&global_row, 1, MPI_INT, packed_array.data(), int_bytes, &ctr, comm);

        start_on = A->on_proc->idx1[i];
        end_on = A->on_proc->idx1[i+1];
        start_off = A->off_proc->idx1[i];
        end_off = A->off_proc->idx1[i+1];
        row_size = end_on - start_on + end_off - start_off;
        MPI_Pack(&row_size, 1, MPI_INT, packed_array.data(), int_bytes, &ctr,
                comm);

        // Pack on_proc portion of row
        for (int j = start_on; j < end_on; j++)
        {
            col = A->on_proc->idx2[j];
            global_col = A->on_proc_column_map[col];
            MPI_Pack(&global_col, 1, MPI_INT, packed_array.data(), int_bytes, &ctr, 
                    comm);
            val = A->on_proc->vals[j];
            MPI_Pack(&val, 1, MPI_DOUBLE, packed_array.data(), double_bytes, &ctr, 
                    comm);
        }

        // Pack off_proc portion of row
        for (int j = start_off; j < end_off; j++)
        {
            col = A->off_proc->idx2[j];
            global_col = A->off_proc_column_map[col];
            MPI_Pack(&global_col, 1, MPI_INT, packed_array.data(), int_bytes, &ctr, comm);
            val = A->off_proc->vals[j];
            MPI_Pack(&val, 1, MPI_DOUBLE, packed_array.data(), double_bytes, &ctr, comm);
        }
    }

    return ctr;
}

void unpack_mat(ParCSRMatrix* A, CSRMatrix* recv_mat, char* recv_data, 
        std::vector<int>& off_proc_col_to_proc, int count, int mat_proc,
        int* map_proc_ptr, int* tmp_row_ptr, MPI_Comm comm)
{
    int rank; MPI_Comm_rank(comm, &rank);
    int row_size, global_col, global_row;
    double val;
    int idx = *map_proc_ptr;

    int ctr = 0;
    int tmp_rows = *tmp_row_ptr;
    int int_bytes, double_bytes;

    // How many rows (and which) should I recv from process i?
    while (ctr < count)
    {
        MPI_Unpack(recv_data, count, &ctr, &global_row, 1, MPI_INT, comm);
        MPI_Unpack(recv_data, count, &ctr, &row_size, 1, MPI_INT, comm);
        if (A->off_proc_column_map[idx] == global_row)
        {
            // I hold this global column, need to store this row
            recv_mat->idx1[tmp_rows + 1] = recv_mat->idx1[tmp_rows] + row_size;
            tmp_rows++;
            for (int j = 0; j < row_size; j++)
            {
                MPI_Unpack(recv_data, count, &ctr, &global_col, 1, MPI_INT,
                        comm);
                MPI_Unpack(recv_data, count, &ctr, &val, 1, MPI_DOUBLE,
                        comm);
                recv_mat->idx2.push_back(global_col);
                recv_mat->vals.push_back(val);
            }
            
            idx++;
            if (idx == A->off_proc_num_cols)
                idx = 0;

            // Check if I am done recving from this proc
            if (off_proc_col_to_proc[idx] != mat_proc)
                break;
        }
        else // Throw away this row
        {
            MPI_Pack_size(row_size, MPI_INT, comm, &int_bytes);
            ctr += int_bytes;
            MPI_Pack_size(row_size, MPI_DOUBLE, comm, &double_bytes);
            ctr += double_bytes;
        }
    }
    
    *map_proc_ptr = idx;
    *tmp_row_ptr = tmp_rows;
}

CSRMatrix* mat_comm(ParCSRMatrix* A, ParCSRMatrix* B, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    CSRMatrix* recv_mat = new CSRMatrix(A->comm->recv_data->size_msgs, -1);
    CSRMatrix* recv_mat_tmp = new CSRMatrix(A->comm->recv_data->size_msgs, -1);
    recv_mat->idx1[0] = 0;
    recv_mat_tmp->idx1[0] = 0;

    int send_proc = rank - 1;
    if (send_proc < 0) send_proc += num_procs;
    int recv_proc = rank + 1;
    if (recv_proc >= num_procs) recv_proc -= num_procs;

    std::vector<int> off_proc_col_to_proc(A->off_proc_column_map.size());
    A->partition->form_col_to_proc(A->off_proc_column_map, off_proc_col_to_proc);
    int map_proc_ptr = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (off_proc_col_to_proc[i] >= recv_proc)
        {
            map_proc_ptr = i;
            break;
        }
    }

    // Communicate one message at a time, forming recv_mat
    // 1. Pack all data from Pl->on_proc and Pl->off_proc into a single buffer
    std::vector<char> packed_array;
    int count = pack_mat(B, packed_array, comm);
    
    std::vector<char> recv_array;
    char* send_data = packed_array.data();
    char* recv_data;

    MPI_Request send_req;
    MPI_Status status;
    int key = 84725;

    // Receive msgs from all processes p > rank
    // Adding needed rows (in order) to recv_mat_tmp
    int tmp_rows = 0;
    for (int i = rank + 1; i < num_procs; i++)
    {
        MPI_Isend(send_data, count, MPI_PACKED, send_proc, key, 
                comm, &send_req);

        // Probe for message of next portion of P
        MPI_Probe(recv_proc, key, comm, &status);
        
        // Get size of message
        MPI_Get_count(&status, MPI_PACKED, &count);

        if (send_data == packed_array.data())
        {
            if (count > recv_array.size()) recv_array.resize(count);
            recv_data = recv_array.data();
        }
        else
        {
            if (count > packed_array.size()) packed_array.resize(count);
            recv_data = packed_array.data();
        }

        // Recv data into recv_array
        MPI_Recv(recv_data, count, MPI_PACKED, recv_proc, key,
                comm, &status);
        
        // Wait for send to complete
        MPI_Wait(&send_req, &status);

        send_data = recv_data;

        // If I don't need any rows of this process, move on
        if (off_proc_col_to_proc[map_proc_ptr] != i)
            continue;

        // 2. Add recvd data into recv_mat_tmp
        // Get first_local_row of process i
        unpack_mat(A, recv_mat_tmp, recv_data, off_proc_col_to_proc, count, i,
                &map_proc_ptr, &tmp_rows, comm);
    }

    // Recv messages from all processes p < rank
    // Adding needed rows (in order) directly to recv_mat
    int recv_mat_rows = 0;
    for (int i = 0; i < rank; i++)
    {
        // Initialize send of current portion of P
        MPI_Isend(send_data, count, MPI_PACKED, send_proc, key, 
                comm, &send_req);

        // Probe for message of next portion of P
        MPI_Probe(recv_proc, key, comm, &status);
        
        // Get size of message
        MPI_Get_count(&status, MPI_PACKED, &count);

        if (send_data == packed_array.data())
        {
            if (count > recv_array.size()) recv_array.resize(count);
            recv_data = recv_array.data();
        }
        else
        {
            if (count > packed_array.size()) packed_array.resize(count);
            recv_data = packed_array.data();
        }

        // Recv data into recv_array
        MPI_Recv(recv_data, count, MPI_PACKED, recv_proc, key,
                comm, &status);
        
        // Wait for send to complete
        MPI_Wait(&send_req, &status);

        send_data = recv_data;

        // If I don't need any rows of this process, move on
        if (off_proc_col_to_proc[map_proc_ptr] != i)
            continue;

        // 2. Add recvd data into recv_mat_tmp
        // Get first_local_row of process i
        unpack_mat(A, recv_mat, recv_data, off_proc_col_to_proc, count, i,
                &map_proc_ptr, &recv_mat_rows, comm);
    }

    // Append recv_mat_tmp to end of recv_mat
    for (int i = 0; i < tmp_rows; i++)
    {
        recv_mat->idx1[recv_mat_rows+1] = recv_mat->idx1[recv_mat_rows] + 
            (recv_mat_tmp->idx1[i+1] - recv_mat_tmp->idx1[i]);
        recv_mat_rows++;
    }
    recv_mat->idx2.insert(recv_mat->idx2.end(), recv_mat_tmp->idx2.begin(), recv_mat_tmp->idx2.end());
    recv_mat->vals.insert(recv_mat->vals.end(), recv_mat_tmp->vals.begin(), recv_mat_tmp->vals.end());

    recv_mat->nnz = recv_mat->idx2.size();

    // Free recv_mat_tmp
    delete recv_mat_tmp;

    return recv_mat;
}

ParCSRMatrix* matmat(ParCSRMatrix* A, ParCSRMatrix* B, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Currently only using for A*P, so can use P's partition
    ParCSRMatrix* C = new ParCSRMatrix(B->partition);
        
    CSRMatrix* C_on_on = A->on_proc->mult((CSRMatrix*) B->on_proc);
    CSRMatrix* C_on_off = A->on_proc->mult((CSRMatrix*) B->off_proc);

    CSRMatrix* recv_mat = mat_comm(A, B, comm);
        
    // Remainder can stay the same as original send/recv (isends/irecvs/waitall version)
    A->mult_helper(B, C, recv_mat, C_on_on, C_on_off);
    
    delete recv_mat;

    delete C_on_on;
    delete C_on_off;

    return C;
}

/*
double* to_dense(ParCSRMatrix* A, int* first_ptr)
{
    int proc_first = -1;
    int n_cols = A->local_num_cols + A->off_proc_num_cols;
    double* A_d = new double[(A->local_num_rows + 1) * n_cols];

    // Sorting columns globally, combining on_proc and off_proc matrices
    for (int col = 0; col < A->off_proc_num_cols; col++)
    {
        global_col = A->off_proc_column_map[col];
        if (global_col > A->first_local_col)
        {
            if (proc_first < 0)
                proc_first = col;
            col += A->local_num_cols;
        }
        A_d[A->local_num_rows*n_cols + col] = A->off_proc_column_map[col];
    }
    for (int col = 0; col < A->local_num_cols; col++)
        A_d[A->local_num_rows*n_cols + col + proc_first] = col + A->first_local_col;


    for (int row = 0; row < A->local_num_rows; row++)
    {
        start = A->on_proc->idx1[row];
        end = A->on_proc->idx1[row+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            A_d[row*n_cols + col + proc_first] = val;
        }
        start = A->off_proc->idx1[row];
        end = A->off_proc->idx1[row+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            val = A->off_proc->vals[j];

            if (col < proc_first)
                A_d[row*n_cols + col] = val;
            else
                A_d[row*n_cols + A->local_num_cols + col] = val;                
        }
    }

    *first_ptr = proc_first;

    return A_d;
}

double* matmat_onproc(double* A, double* P, int n, int m)
{
    double* AP = new double[n*m]();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < n; k++)
            {
                AP[i*m+j] = A[i*n+k] * P[k*m+j];
            }
        }
    }

    return AP;
}
*/

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 5;
    int system = 0;
    double strong_threshold = 0.25;
    int num_variables = 1;

    coarsen_t coarsen_type = HMIS;
    interp_t interp_type = Extended;

    ParMultilevel* ml;
    ParCSRMatrix* A = NULL;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }
    if (system < 2)
    {
        int dim = 3;
        double* stencil = NULL;
        std::vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            coarsen_type = Falgout;
            interp_type = ModClassical;

            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/4.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
#ifdef USING_MFEM
    else if (system == 2)
    {
        const char* mesh_file = argv[2];
        int mfem_system = 0;
        int order = 2;
        int seq_refines = 1;
        int par_refines = 1;
        int max_dofs = 1000000;
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    max_dofs = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }

        coarsen_type = HMIS;
        interp_type = Extended;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
                strong_threshold = 0.5;
                A = mfem_linear_elasticity(x, b, &num_variables, mesh_file, order, 
                        seq_refines, par_refines);
                break;
            case 3:
                A = mfem_adaptive_laplacian(x, b, mesh_file, order);
                x.set_const_value(1.0);
                A->mult(x, b);
                x.set_const_value(0.0);
                break;
            case 4:
                A = mfem_dg_diffusion(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 5:
                A = mfem_dg_elasticity(x, b, &num_variables, mesh_file, order, seq_refines, par_refines);
                break;
        }
    }
#endif
    else if (system == 3)
    {
        const char* file = "../../examples/LFAT5.pm";
        A = readParMatrix(file);
    }

    if (system != 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols);
        b = ParVector(A->global_num_rows, A->local_num_rows);
        x.set_rand_values();
        A->mult(x, b);
        x.set_const_value(0.0);
    }


    // Ruge-Stuben AMG - Create Hierarchy
    if (rank == 0) printf("Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-05;
    ml->num_variables = num_variables;
    ml->track_times = true;
    ml->setup(A);

    int n_iter = 1000;

    // Go Through Each level in hierarchy
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);

        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSRMatrix *AP, *AP_new;
        
        MPI_Comm level_comm;
        int active = 0;
        if (Pl->local_num_rows)
            active = 1;
        MPI_Comm_split(MPI_COMM_WORLD, active, rank, &level_comm);

        // Compare matmat strategies, make sure equivalent
        /*AP = Al->mult(Pl);
        if (active)
            AP_new = matmat(Al, Pl, level_comm);
        else AP_new = new ParCSRMatrix();
    
        compare_mats(AP, AP_new);

        delete AP_new;
        delete AP;*/

        CSRMatrix* recv_mat = Al->comm->communicate(Pl);
        CSRMatrix* recv_mat_new;
        if (active)
            recv_mat_new = mat_comm(Al, Pl, level_comm);
        else recv_mat_new = new CSRMatrix();

        compare_mats(recv_mat, recv_mat_new);

        delete recv_mat;
        delete recv_mat_new;

        // Time Standard Communication
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_iter; j++)
        {
            AP = Al->mult(Pl);
            delete AP;
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Sparse A*P : %e\n", t0);

        // Time New Communication
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_iter; j++)
        {
            if (active)
            {
                AP_new = matmat(Al, Pl, level_comm);
                delete AP_new;
            }
        }
        tfinal = (MPI_Wtime() - t0) / n_iter;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Sparse Fox A*P : %e\n", t0);


        MPI_Comm_free(&level_comm);

        // Test 2 : Convert P to condensed dense matrix (in method at top of file... may help w/ GPUs)
        // Test 3 : Convert both A and P to condensed dense matrices (may be helpful for GPUs?)
        //

    }

    delete ml;

    delete A;

    MPI_Finalize();
    return 0;
}

