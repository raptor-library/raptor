// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/par_mis.hpp"

void comm_states(const ParCSRMatrix* A, CommPkg* comm, 
        const aligned_vector<int>& states, aligned_vector<int>& recv_indices, 
        aligned_vector<int>& off_proc_states, bool first_pass = false)
{
    if (first_pass)
    {
        aligned_vector<int>& recvbuf = comm->communicate(states);
        std::copy(recvbuf.begin(), recvbuf.end(), off_proc_states.begin());
    }
    else
    {
        ParComm* pc = (ParComm*)(comm);
        std::function<bool(int)> compare_func = [](const int a)
        {
            return a == Unassigned || a > Selected;
        };
        aligned_vector<int>& recvbuf = pc->conditional_comm(states, states, 
                off_proc_states, compare_func);
        int off_proc_num_cols = off_proc_states.size();
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            if (off_proc_states[i] == Unassigned || off_proc_states[i] > Selected)
                off_proc_states[i] = recvbuf[i];
        }
    }
}

void comm_off_proc_states(const ParCSRMatrix* A, CommPkg* comm,
        const aligned_vector<int>& off_proc_states, aligned_vector<int>& recv_indices, 
        aligned_vector<int>& states, bool first_pass = false)
{
    std::function<int(int,int)> result_func = [](const int a, const int b)
    {
        if (b == TmpSelection) return TmpSelection;
        else return a;
    };

    if (first_pass)
    {
        comm->communicate_T(off_proc_states, states, A->off_proc->b_cols,
                result_func, result_func);
    }
    else
    {
        ParComm* pc = (ParComm*)(comm);
        std::function<bool(int)> compare_func = [](const int a)
        {
            return a == Unassigned || a > Selected;
        };
        pc->conditional_comm_T(off_proc_states, states, off_proc_states, compare_func,
                states, result_func);
    }
}

void comm_finished(const ParCSRMatrix* A, 
        aligned_vector<int>& active_sends,
        aligned_vector<int>& active_recvs, 
        int remaining)
{
    int ctr, prev_ctr;
    int n_sends, n_recvs;
    int start, end;
    int proc, idx, size;
    int finish_tag = 19432;
    int finish_tag_T = 23491;

    n_sends = 0;
    for (int i = 0; i < A->comm->send_data->num_msgs; i++)
    {
        proc = A->comm->send_data->procs[i];
        start = A->comm->send_data->indptr[i];
        if (active_sends[i])
        {
            A->comm->send_data->int_buffer[start] = remaining;
            RAPtor_MPI_Isend(&(A->comm->send_data->int_buffer[start]), 1, RAPtor_MPI_INT, proc,
                    finish_tag, RAPtor_MPI_COMM_WORLD, 
                    &(A->comm->send_data->requests[n_sends++]));
        }
    }
    n_recvs = 0;
    for (int i = 0; i < A->comm->recv_data->num_msgs; i++)
    {
        proc = A->comm->recv_data->procs[i];
        start = A->comm->recv_data->indptr[i];
        if (active_recvs[i])
        {
            RAPtor_MPI_Irecv(&(A->comm->recv_data->int_buffer[start]), 1, RAPtor_MPI_INT, proc,
                    finish_tag, RAPtor_MPI_COMM_WORLD, 
                    &(A->comm->recv_data->requests[n_recvs++]));
        }
    }

    if (n_sends)
    {
        RAPtor_MPI_Waitall(n_sends, A->comm->send_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }
    if (n_recvs)
    {
        RAPtor_MPI_Waitall(n_recvs, A->comm->recv_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }

    n_sends = 0;
    for (int i = 0; i < A->comm->send_data->num_msgs; i++)
    {
        proc = A->comm->send_data->procs[i];
        if (active_sends[i])
        {
            RAPtor_MPI_Irecv(&(active_sends[i]), 1, RAPtor_MPI_INT, proc,
                    finish_tag_T, RAPtor_MPI_COMM_WORLD, 
                    &(A->comm->send_data->requests[n_sends++]));
        }
    }

    n_recvs = 0;
    for (int i = 0; i < A->comm->recv_data->num_msgs; i++)
    {
        proc = A->comm->recv_data->procs[i];
        start = A->comm->recv_data->indptr[i];
        if (active_recvs[i])
        {
            active_recvs[i] = A->comm->recv_data->int_buffer[start];
            A->comm->recv_data->int_buffer[start] = remaining;
            RAPtor_MPI_Isend(&(A->comm->recv_data->int_buffer[start]), 1, RAPtor_MPI_INT, proc,
                    finish_tag_T, RAPtor_MPI_COMM_WORLD, 
                    &(A->comm->recv_data->requests[n_recvs++]));
        }
    }

    if (n_sends)
    {
        RAPtor_MPI_Waitall(n_sends, A->comm->send_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }
    if (n_recvs)
    {
        RAPtor_MPI_Waitall(n_recvs, A->comm->recv_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }
}

void comm_coarse_dist1(const ParCSRMatrix* A, 
        CommPkg* comm,
        aligned_vector<int>& active_sends,
        aligned_vector<int>& active_recvs,
        aligned_vector<int>& C,
        bool first_pass = false)
{
    int n_sends, n_recvs;
    int start, end;
    int proc, idx, size;
    int tag = 935921;

    if (first_pass)
    {
        comm->communicate(C);
    }
    else
    {
        n_sends = 0;
        for (int i = 0; i < A->comm->send_data->num_msgs; i++)
        {
            proc = A->comm->send_data->procs[i];
            start = A->comm->send_data->indptr[i];
            end = A->comm->send_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = A->comm->send_data->indices[j];
                A->comm->send_data->int_buffer[j] = C[idx];
            }
            if (active_sends[i])
            {
                RAPtor_MPI_Isend(&(A->comm->send_data->int_buffer[start]), end - start, RAPtor_MPI_INT, proc,
                        tag, RAPtor_MPI_COMM_WORLD, &(A->comm->send_data->requests[n_sends++]));
            }
        }

        n_recvs = 0;
        for (int i = 0; i < A->comm->recv_data->num_msgs; i++)
        {
            proc = A->comm->recv_data->procs[i];
            start = A->comm->recv_data->indptr[i];
            end = A->comm->recv_data->indptr[i+1];
            if (active_recvs[i])
            {
                RAPtor_MPI_Irecv(&(A->comm->recv_data->int_buffer[start]), end - start, RAPtor_MPI_INT,
                        proc, tag, RAPtor_MPI_COMM_WORLD, &(A->comm->recv_data->requests[n_recvs++]));
            }
        }

        if (n_sends)
        {
            RAPtor_MPI_Waitall(n_sends, A->comm->send_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
        }

        if (n_recvs)
        {
            RAPtor_MPI_Waitall(n_recvs, A->comm->recv_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);
        }
    }
}

int mis2(const ParCSRMatrix* A, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, 
        bool tap_comm, double* rand_vals)
{
    // Get MPI Information
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    // Declare Variables
    int start, end, col;
    int start_k, end_k;
    int remaining, iterate;
    int off_remaining;
    int ctr, v, w, u;
    int head, length;
    bool found;

    CommPkg* comm = A->comm;
    if (tap_comm) 
    {
        comm = A->tap_comm;
    }

    aligned_vector<int> recv_indices;
    if (A->comm->recv_data->size_msgs || A->comm->send_data->size_msgs)
    {
        ctr = A->comm->recv_data->size_msgs; 
        if (A->comm->send_data->size_msgs > ctr)
            ctr = A->comm->send_data->size_msgs;
        recv_indices.resize(ctr);
    }

    // Initialize vector of random values associated with each row
    aligned_vector<double> r;
    aligned_vector<double> off_proc_r;
    aligned_vector<int> V;
    aligned_vector<int> off_V;
    aligned_vector<int> C;
    aligned_vector<int> next;
    aligned_vector<int> active_sends;
    aligned_vector<int> active_recvs;
    aligned_vector<double> row_max;
    int tmp;
    if (A->local_num_rows)
    {
        V.resize(A->local_num_rows);
        std::iota(V.begin(), V.end(), 0);
        states.resize(A->local_num_rows);
        std::fill(states.begin(), states.end(), Unassigned);
        r.resize(A->local_num_rows);
        C.resize(A->local_num_rows, Unselected);
        next.resize(A->local_num_rows, Unassigned);
        row_max.resize(A->local_num_rows);
        if (rand_vals)
        {
            for (int i = 0; i < A->local_num_rows; i++)
            {
                r[i] = rand_vals[i];
            }
        }
        else
        {
            for (int i = 0; i < A->local_num_rows; i++)
            {
                r[i] = ((double)(rand()) / RAND_MAX);
            }
        }
    }
    if (A->off_proc_num_cols)
    {
        off_V.resize(A->off_proc_num_cols);
        std::iota(off_V.begin(), off_V.end(), 0);
        off_proc_states.resize(A->off_proc_num_cols);
        std::fill(off_proc_states.begin(), off_proc_states.end(), Unassigned);
        off_proc_r.resize(A->off_proc_num_cols);
    }
    aligned_vector<double>& recvbuf = comm->communicate(r);

    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        off_proc_r[i] = recvbuf[i];
    }
    if (A->comm->send_data->num_msgs)
    {
        active_sends.resize(A->comm->send_data->num_msgs, 1);
    }
    if (A->comm->recv_data->num_msgs)
    {
        active_recvs.resize(A->comm->recv_data->num_msgs, 1);
    }

    // Create D matrices (directed graph of A)
    CSRMatrix* D_on = new CSRMatrix(A->local_num_rows, A->on_proc_num_cols);
    CSRMatrix* D_off = new CSRMatrix(A->local_num_rows, A->off_proc_num_cols);
    D_on->idx2.reserve(0.5*A->on_proc->nnz);
    D_on->vals.clear();
    D_off->idx2.reserve(0.5*A->off_proc->nnz);
    D_off->vals.clear();
    D_on->idx1[0] = 0;
    D_off->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            if (r[i] > r[col])
            {
                D_on->idx2.emplace_back(col);
            }
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            if (r[i] > off_proc_r[col])
            {
                D_off->idx2.emplace_back(col);
            }
        }

        D_on->idx1[i+1] = D_on->idx2.size();
        D_off->idx1[i+1] = D_off->idx2.size();
    }

    // Form column-wise local matrices of A
    CSCMatrix* A_on_csc = A->on_proc->to_CSC();
    CSCMatrix* A_off_csc = A->off_proc->to_CSC();
    
    // Find DistTmpSelection Maximal Independent Set -- Main Loop
    remaining = A->local_num_rows;
    off_remaining = A->off_proc_num_cols;
    iterate = 0;
    int total_remaining;
    int set_size, total_set_size;
    bool first_pass = true;
    while (remaining || off_remaining || first_pass)
    {
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            found = false;

            // Check if unassigned neighbor
            start = D_on->idx1[v];
            end = D_on->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = D_on->idx2[j];
                if (states[w] == Unassigned || states[w] > Selected)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                start = D_off->idx1[v];
                end = D_off->idx1[v+1];
                for (int j = start; j < end; j++)
                {
                    w = D_off->idx2[j];
                    if (off_proc_states[w] == Unassigned || off_proc_states[w] > Selected)
                    {
                        found = true;
                        break;
                    }
                }
            }

            // If no unassigned neighbor, select v
            if (!found)
            {
                states[v] = TmpSelection;
            }
        }

        // Communicate new (temporary) states
        comm_states(A, comm, states, recv_indices, off_proc_states, first_pass);

        // Find max temp state random in each row
        for (int i = 0; i < A->local_num_rows; i++)
        {
            double max_val = -1.0;

            start = A->on_proc->idx1[i];
            end = A->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                u = A->on_proc->idx2[j];
                if (states[u] > Selected && r[u] > max_val)
                {
                    max_val = r[u];
                }
            }

            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                u = A->off_proc->idx2[j];
                if (off_proc_states[u] > Selected && off_proc_r[u] > max_val)
                {
                    max_val = off_proc_r[u];
                }
            }

            row_max[i] = max_val;
        }

        // Find any conflicts in A->on_proc
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] != TmpSelection)
            {
                continue;
            }
            found = false;

            start = A->on_proc->idx1[v];
            end = A->on_proc->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = A->on_proc->idx2[j];
                if (row_max[w] > r[v])
                {
                    found = true;
                    break;
                }
            }
 
            if (!found)
            {
                states[v] = NewSelection;
            }
        }

        // Find any conflicts in A->off_proc
        for (int i = 0; i < off_remaining; i++)
        {
            v = off_V[i];
            if (off_proc_states[v] != TmpSelection)
            {
                continue;
            }

            found = false;
            start = A_off_csc->idx1[v];
            end = A_off_csc->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = A_off_csc->idx2[j];
                if (row_max[w] > off_proc_r[v])
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                off_proc_states[v] = NewSelection;
            }
        }
            
        // Communicate new states (containing new coarse)
        // Finding max (if any proc has state[v] == TmpSelection, state[v] should
        // be TmpSelection. Else (all procs have state[v] == NewSelection), state[v] is NewSelection 
        // aka new coarse point)
        comm_off_proc_states(A, comm, off_proc_states, recv_indices, states, first_pass);
        comm_states(A, comm, states, recv_indices, off_proc_states, first_pass);

        // Update states connecting to (dist1 or dist2) any
        // new coarse points (with state == NewSelection)
        head = TmpSelection;
        length = 0;
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == NewSelection)
            {
                start = A_on_csc->idx1[v];
                end = A_on_csc->idx1[v+1];
                for (int j = start; j < end; j++)
                {
                    w = A_on_csc->idx2[j];
                    C[w] = Selected;
                    if (next[w] == Unassigned)
                    {
                        next[w] = head;
                        head = w;
                        length++;
                    }
                }
            }
        }
        for (int i = 0; i < off_remaining; i++)
        {
            v = off_V[i];
            if (off_proc_states[v] == NewSelection)
            {
                start = A_off_csc->idx1[v];
                end = A_off_csc->idx1[v+1];
                for (int j = start; j < end; j++)
                {
                    w = A_off_csc->idx2[j];
                    C[w] = Selected;
                    if (next[w] == Unassigned)
                    {
                        next[w] = head;
                        head = w;
                        length++;
                    }
                }
            }
        }

        // Communicate updated states (if states[v] == NewUnselection, there exists 
        // and idx in row v such that states[idx] is new coarse)
        comm_coarse_dist1(A, comm, active_sends, active_recvs, C, first_pass);
        aligned_vector<int>& recv_C = comm->get_int_buffer();


        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == NewSelection)
            {
                continue;
            }
            found = false;

            start = A->on_proc->idx1[v];
            end = A->on_proc->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = A->on_proc->idx2[j];
                if (states[w] == NewSelection)
                {
                    found = true;
                    break;
                }
                if (C[w] == Selected)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                start = A->off_proc->idx1[v];
                end = A->off_proc->idx1[v+1];
                for (int j = start; j < end; j++)
                {
                    w = A->off_proc->idx2[j];
                    if (off_proc_states[w] == NewSelection)
                    {
                        found = true;
                        break;
                    }
                    if (recv_C[w] == Selected)
                    {
                        found = true;
                        break;
                    }
                }
            }
            if (found)
            {
                states[v] = NewUnselection;
            }
        }
        for (int i = 0; i < length; i++)
        {
            tmp = head;
            head = next[head];
            C[tmp] = Unselected;
            next[tmp] = 0;
        }

        // Communicate final updated states of iteration
        comm_states(A, comm, states, recv_indices, off_proc_states, first_pass);

        // Update states
        ctr = 0;
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == NewSelection)
            {
                states[v] = Selected;
            }
            else if (states[v] == NewUnselection)
            {
                states[v] = Unselected;
            }
            else
            {
                V[ctr++] = v;
            }
        }
        remaining = ctr;

        ctr = 0;
        for (int i = 0; i < off_remaining; i++)
        {
            v = off_V[i];
            if (off_proc_states[v] == NewSelection)
            {
                off_proc_states[v] = Selected;
            }
            else if (off_proc_states[v] == NewUnselection)
            {
                off_proc_states[v] = Unselected;
            }
            else
            {
                off_V[ctr++] = v;
            }
        }
        off_remaining = ctr;

        first_pass = false;
        comm = A->comm;

        comm_finished(A, active_sends, active_recvs, remaining + off_remaining);

        iterate++;
    }

    delete D_on;
    delete D_off;
    delete A_on_csc;
    delete A_off_csc;

    return iterate;
}

