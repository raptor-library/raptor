// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
  #include "external/mfem_wrapper.hpp"
#endif

using namespace raptor;

#define eager_cutoff 1000
#define short_cutoff 62

void fast_waitall(int n_msgs, std::vector<MPI_Request>& requests)
{
    if (n_msgs == 0) return;

    int recv_n, idx;
    std::vector<int> recv_indices(n_msgs);
    while (n_msgs)
    {
        MPI_Waitsome(n_msgs, requests.data(), &recv_n, recv_indices.data(), MPI_STATUSES_IGNORE);
        for (int i = 0; i < recv_n; i++)
        {
            idx = recv_indices[i];
            requests[idx] = MPI_REQUEST_NULL;
        }
        idx = 0;
        for (int i = 0; i < n_msgs; i++)
        {
            if (requests[i] != MPI_REQUEST_NULL)
                requests[idx++] = requests[i];
        }
        n_msgs -= recv_n;
    }
 
    return;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    // Cube topology info
    int num_nodes = num_procs / 16;
    int node = rank / 16;
    int num_dir = cbrt(num_nodes);
    if (num_dir * num_dir * num_dir < num_nodes)
        num_dir++;

    int x_dim = num_dir;
    int y_dim = x_dim * num_dir;
    int z_dim = y_dim * num_dir;

    int x_pos = node % num_dir;
    int y_pos = (node / x_dim) % num_dir;
    int z_pos = (node / y_dim) % num_dir;

    std::vector<int> my_cube_pos(3);
    my_cube_pos[0] = x_pos;
    my_cube_pos[1] = y_pos;
    my_cube_pos[2] = z_pos;
    std::vector<int> cube_pos(3*num_procs);
    MPI_Allgather(my_cube_pos.data(), 3, MPI_INT, cube_pos.data(), 3, MPI_INT, MPI_COMM_WORLD);
 
    std::vector<int> proc_distances(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        proc_distances[i] = (fabs(cube_pos[i*3] - x_pos) + fabs(cube_pos[i*3+1] - y_pos)
                + fabs(cube_pos[i*3+2] - z_pos));
    }

    // Mesh topology info
    x_dim = 24;
    y_dim = num_nodes;
    x_pos = node % 24;
    y_pos = (node / 24) % 24;
    my_cube_pos[0] = x_pos;
    my_cube_pos[1] = y_pos;
    MPI_Allgather(my_cube_pos.data(), 2, MPI_INT, cube_pos.data(), 2, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> worst_proc_distances(num_procs);
    for (int i = 0; i < num_procs; i++)
    {
        worst_proc_distances[i] = (fabs(cube_pos[i*2] - x_pos) + fabs(cube_pos[i*2+1] - y_pos));
    }
 

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double t0_comm, tfinal_comm;
    int n0, s0;
    int nfinal, sfinal;
    double raptor_setup, raptor_solve;
    int num_variables = 1;
    relax_t relax_type = SOR;
    coarsen_t coarsen_type = CLJP;
    interp_t interp_type = ModClassical;
    double strong_threshold = 0.25;

    std::vector<double> residuals;

    if (system < 2)
    {
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
        strong_threshold = 0.0;
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
    }

    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, relax_type);
    ml->num_variables = num_variables;
    ml->setup(A);

    int n_tests = 100;
    int n_spmv_tests = 1000;
    CSRMatrix* C;
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;

        if (rank == 0) printf("Level %d\n", i);
        std::vector<int> rowptr(Pl->local_num_rows + 1);
        int nnz = Pl->on_proc->nnz + Pl->off_proc->nnz;
        std::vector<int> col_indices;
        std::vector<double> values;
        if (nnz)
        {
            col_indices.resize(nnz);
            values.resize(nnz);
        }
        rowptr[0] = 0;
        for (int j = 0; j < Pl->local_num_rows; j++)
        {
            int start = Pl->on_proc->idx1[j];
            int end = Pl->on_proc->idx1[j+1];
            for (int k = start; k < end; k++)
            {
                int col = Pl->on_proc->idx2[k];
                double val = Pl->on_proc->vals[k];
                col_indices.push_back(Pl->on_proc_column_map[col]);
                values.push_back(val);
            }
            start = Pl->off_proc->idx1[j];
            end = Pl->off_proc->idx1[j+1];
            for (int k = start; k < end; k++)
            {
                int col = Pl->off_proc->idx2[k];
                double val = Pl->off_proc->vals[k];
                col_indices.push_back(Pl->off_proc_column_map[col]);
                values.push_back(val);
            }
            rowptr[j+1] = col_indices.size();
        }

        // Count Queue
        int queue_search = 0;
        MPI_Status status;
        int msg_s;
        std::vector<int> recv_idx(Al->comm->recv_data->num_msgs, 1);
        for (int send = 0; send < Al->comm->send_data->num_msgs; send++)
        {   
            int proc = Al->comm->send_data->procs[send];
            int start = Al->comm->send_data->indptr[send];
            int end = Al->comm->send_data->indptr[send+1]; 
            MPI_Isend(&Al->comm->send_data->buffer[start], end - start, MPI_DOUBLE, proc,
                    4938, MPI_COMM_WORLD, &Al->comm->send_data->requests[send]);
        }
        for (int recv = 0; recv < Al->comm->recv_data->num_msgs; recv++)
        {   
            MPI_Probe(MPI_ANY_SOURCE, 4938, MPI_COMM_WORLD, &status);
            int proc = status.MPI_SOURCE;
            MPI_Get_count(&status, MPI_DOUBLE, &msg_s);
            MPI_Recv(&Al->comm->recv_data->buffer[0], msg_s, MPI_DOUBLE, proc,
                    4938, MPI_COMM_WORLD, &status);
            for (int idx = 0; idx < Al->comm->recv_data->num_msgs; idx++)
            {   
                int proc_idx = Al->comm->recv_data->procs[idx];
                if (proc == proc_idx)
                {   
                    queue_search++; 
                    recv_idx[idx] = 0;
                    break;
                }
                queue_search += recv_idx[idx];
            }
        }
        if (Al->comm->send_data->num_msgs) MPI_Waitall(Al->comm->send_data->num_msgs,
                Al->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);

        int max_queue = 0;
        MPI_Allreduce(&queue_search, &max_queue, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int msg_n = 0;
        if (queue_search == max_queue) msg_n = Al->comm->recv_data->num_msgs;
        MPI_Allreduce(MPI_IN_PLACE, &msg_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) printf("MaxQueueN %d\t MsgN %d\n", max_queue, msg_n);


        if (rank == 0) printf("A*P:\n");
        //Al->print_mult(Pl, proc_distances, worst_proc_distances);
        int active = 1;
        int sum_active;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = Al->comm->communicate(rowptr, col_indices, values);
        delete C;
        tfinal = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            C = Al->comm->communicate(rowptr, col_indices, values);
            tfinal += (MPI_Wtime() - t0);
            delete C;
        }
        tfinal /= n_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);

        /*
        if (rank == 0) printf("\nP.T*AP:\n");
        AP->print_mult_T(Pl_csc, proc_distances, worst_proc_distances);
        active = 1;
        if (AP->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active); 
        C = AP->mult_T(Pl_csc);
        delete C;
        AP->spgemm_T_data.time = 0;
        AP->spgemm_T_data.comm_time = 0;
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            C = AP->mult_T(Pl_csc);
            delete C;
        }
        AP->spgemm_T_data.time /= n_tests;
        AP->spgemm_T_data.comm_time /= n_tests;
        MPI_Reduce(&(AP->spgemm_T_data.time), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Time: %e\n", t0);
        MPI_Reduce(&(AP->spgemm_T_data.comm_time), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);
        */

        //delete Pl_csc;
        //delete AP;



        MPI_Barrier(MPI_COMM_WORLD);
        ml->levels[i]->x.set_const_value(0.0);
        if (rank == 0) printf("A*x\n");
	//Al->print_mult(proc_distances, worst_proc_distances);
        active = 1;
        if (Al->local_num_rows == 0) active = 0;
        MPI_Reduce(&active, &sum_active, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Active Processes: %d\n", sum_active);
//        Al->comm->communicate(ml->levels[i]->x);
    
        if (rank == 0) printf("Using Send + Irecv...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
//            Al->comm->communicate(ml->levels[i]->x);

            for (int recv = 0; recv < Al->comm->recv_data->num_msgs; recv++)
            {
                int proc = Al->comm->recv_data->procs[recv];
                int start = Al->comm->recv_data->indptr[recv];
                int end = Al->comm->recv_data->indptr[recv+1];
                MPI_Irecv(&Al->comm->recv_data->buffer[start], end - start, MPI_DOUBLE, proc,
                        93002, MPI_COMM_WORLD, &Al->comm->recv_data->requests[recv]);
            }

            for (int send = 0; send < Al->comm->send_data->num_msgs; send++)
            {
                int proc = Al->comm->send_data->procs[send];
                int start = Al->comm->send_data->indptr[send];
                int end = Al->comm->send_data->indptr[send+1];
                for (int k = start; k < end; k++)
                {
                    int index = Al->comm->send_data->indices[k];
                    Al->comm->send_data->buffer[k] = ml->levels[i]->x[index];
                }
                MPI_Send(&Al->comm->send_data->buffer[start], end - start, MPI_DOUBLE, proc,
                        93002, MPI_COMM_WORLD);
            }

            if (Al->comm->recv_data->num_msgs)
                MPI_Waitall(Al->comm->recv_data->num_msgs, Al->comm->recv_data->requests.data(),
                        MPI_STATUSES_IGNORE);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);


        if (rank == 0) printf("Using Isend/Irecv + MPI_Waitsomes\n");
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int test = 0; test < n_spmv_tests; test++)
        {
            // b <- A*x
//            Al->comm->communicate(ml->levels[i]->x);

            for (int send = 0; send < Al->comm->send_data->num_msgs; send++)
            {
                int proc = Al->comm->send_data->procs[send];
                int start = Al->comm->send_data->indptr[send];
                int end = Al->comm->send_data->indptr[send+1];
                for (int k = start; k < end; k++)
                {
                    int index = Al->comm->send_data->indices[k];
                    Al->comm->send_data->buffer[k] = ml->levels[i]->x[index];
                }
                MPI_Isend(&Al->comm->send_data->buffer[start], end - start, MPI_DOUBLE, proc,
                        93002, MPI_COMM_WORLD, &Al->comm->send_data->requests[send]);
            }

            for (int recv = 0; recv < Al->comm->recv_data->num_msgs; recv++)
            {
                int proc = Al->comm->recv_data->procs[recv];
                int start = Al->comm->recv_data->indptr[recv];
                int end = Al->comm->recv_data->indptr[recv+1];
                MPI_Irecv(&Al->comm->recv_data->buffer[start], end - start, MPI_DOUBLE, proc,
                        93002, MPI_COMM_WORLD, &Al->comm->recv_data->requests[recv]);
            }

            fast_waitall(Al->comm->send_data->num_msgs, Al->comm->send_data->requests);
            fast_waitall(Al->comm->recv_data->num_msgs, Al->comm->recv_data->requests);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
        MPI_Reduce(&(tfinal), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Comm Time: %e\n", t0);
    }

    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


