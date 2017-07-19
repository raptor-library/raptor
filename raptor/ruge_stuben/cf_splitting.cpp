// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

// Active == -1
// F == 0
// C == 1
//
// TODO -- currently communicating all of local_states at each iteration... this
// is not necessary (only need to communicate updates)
//

void update_weights(ParCSRMatrix* S, std::vector<int>& local_new_coarse,
        std::vector<int>& off_proc_new_coarse, std::vector<double>& weights)
{
    // Heuristic 1
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (local_new_coarse[i] == 1)
        {
            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                if (S->on_proc->vals[j] == 1)
                {
                    weights[i] -= 1;
                    S->on_proc->vals[j] = -1;
                }
            }
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                if (S->off_proc->vals[j] == 1)
                {
                    weights -= 1;
                    S->off_proc->vals[j] = -1;
                }
            }
        }
    }


    // Weight update is MUCH simpler if S is CSC... could transpose local S
    // matrices to CSC here...
    //
    // Heuristic 2
    //
    // Want to find rows i, j such that they both share the same coarse column k
    // Sik != 0, Sjk != 0
    // Also, i and j need to have a strong connection, Sij = 1
    //
    // For all i such that Sik != 0
    //     Sik = -1
    //     for all j such that Sji == 1
    //         if Sjk != 0
    //             wi -= 1
    //             Sji = -1
    //
    // Could create lists for each coarse column... which rows have strong
    // connections with said coarse column
    for (int i = 0; i < S->local_num_rows; i++)
    {
        // Look through on proc cols for row i
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (int jj = start; jj < end; jj++)
        {
            // If col is newly coarse, set S_i,col to -1
            k = S->on_proc->idx2[jj]; 
            if (local_new_coarse[k] == 1) // Sik != 0 (coarse k) 
            {
                S->on_proc->vals[jj] = -1;  // Sik = -1


                // Want to find all rows j such that Sji == 1 and Sjk != 0
                //
                //
                //
                //
                //
                //
                //
                //
                //
                // Go through row of newly coarse col
                col_start = S->on_proc->idx1[k];
                col_end = S->on_proc->idx1[k+1];
                for (int k = col_start; k < col_end; k++)
                {
                    // If S_col,col_k is 1, alter weight
                    if (S->on_proc->vals[k] == 1)
                    {
                        col_k = S->on_proc->idx2[k];
                        weights[i] -= 1;
                        S->on_proc->vals[k] = -1;
                    }
                }
                col_start = S->off_proc->idx1[col];
                col_end = S->off_proc->idx1[col+1];
                for (int k = col_start; k < col_end; k++)
                {
                    // If S_col,col_k is 1, alter weight
                    if (S->off_proc->vals[k] == 1)
                    {
                        col_k = S->on_proc->idx2[k];
                        weights[col_k] -= 1;
                        S->on_proc->vals[k] = -1;
                    }
                }

            }
        }

        // Look through off_proc cols for row i
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            // If off_proc col is newly coarse, set S_i,col to -1
            col = S->off_proc->idx2[j];
            if (off_proc_new_coarse[col] == 1)
            {
                S->off_proc->vals[j] = -1;

                // Go through row of newly coarse col
                col_start = C->off_proc->idx1[col];
                col_end = C->off_proc->idx1[col+1];
                for (int k = col_start; k < col_end; k++)
                {
                    if (S->off_proc->vals[k] == 1)
                    {
                        col_k = S->off_proc->idx2[k];
                        weights[col_k] -= 1;
                        S->off_proc->vals[k] = -1;
                    }
                }
            }
        }
    }
}

void ruge_stuben(ParCSRMatrix* S, std::vector<int>& coarse, std::vector<int>& fine)
{

    // Alter weights based on off_process C/F splittings
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = 
        if (




    

}



// Falgout Coarse-Grid Selection
// Falgout(S):
//     (C, F) = RugeStuben(S) -- runs ruge stuben using only local data
//     Mark processor boundaries as unassigned
//     (C, F) = CLJP(S, C, F)
//     return (C, F)
//
// ... says to coarsen interior points with ruge-stuben, and then use CLJP to
// coarsen boundary with interior C/F splitting as input
int ParCSRMatrix::cf_splitting()
{
    int tag = 8204;
    int max_weight, max_vertex;
    int start, end;
    int col_start, col_end;
    int col, col_k, idx;
    int head, length, tmp;
    int proc;

    std::vector<int> states;
    std::vector<double> weights;
    std::vector<int> remaining;
    std::vector<int> send_buffer;
    std::vector<int> neighbor_states;
    std::vector<int> row_coarse;

    CSCMatrix* on_proc_csc = new CSCMatrix(on_proc);
    CSCMatrix* off_proc_csc = new CSCMatrix(off_proc);

    // Initialize lists to store state of each local vertex
    // 1: coarse
    // 0: fine
    // -1: unassigned
    // Initialze lists to store weights of each vertex, equal to the number of
    // nonzeros in row, and remaining vertices (list of unassigned)
    if (local_num_rows)
    {
        states.resize(local_num_rows, -1);
        weights.resize(local_num_rows);
        remaining.resize(local_num_rows);
        std::iota(remaining.begin(), remaining.end(), 0);
        row_coarse.resize(local_num_rows, -1); // wants cols, but square matrix
    }
    if (comm->send_data->num_msgs)
    {
        send_buffer.resize(comm->send_data->num_msgs);
    }
    if (off_proc_num_cols)
    {
        neighbor_states.resize(off_proc_num_cols);
    }

    /****************************************
     * Ruge-Stuben Coarsening, Pass 1
     ****************************************/
    // Calculate weight of each row (equal to nnz in row) and find which row has
    // maximum weight
    max_weight = -1;
    max_vertex = -1;
    for (int i = 0; i < local_num_rows; i++)
    {
        weights[i] = on_proc->idx1[i+1] - on_proc->idx1[i];
        if (weights[i] > max_weight)
        {
            max_weight = weights[i];
            max_vertex = i;
        }
    }

    // While any unassigned vertices, add vertex with largest weight to coarse
    while (remaining.size())
    {
        // Set local state of row to 1 (coarse)
        states[max_vertex] = 1;

        // Set all unassigned neighbors to 0 (fine)
        start = on_proc->idx1[max_vertex];
        end = on_proc->idx1[max_vertex+1];
        for (int j = start; j < end; j++)
        {
            col = on_proc->idx2[j];
            if (local_states[col] == -1)
            {
                states[col] = 0;

                // If unassigned distance-2 neighbor, increase weight
                col_start = on_proc->idx1[col];
                col_end = on_proc->idx1[col+1];
                for (int k = col_start; k < col_end; k++)
                {
                    col_k = on_proc->idx2[j];
                    if (states[col_k] == -1)
                    {
                        weights[col_k] += 1;
                    }
                }
            }
        }

        // Remove any now initailized vertices from remaining
        max_weight = -1;
        idx = 0;
        while (idx < remaining.size())
        {
            vertex = remaining[i];
            if (states[vertex] != -1)
            {
                remaining.erase(i);
            }
            else
            {            
                // If still in remaining, find vertex corresponding to next
                // max_weight
                if (weights[vertex] > max_weight)
                {
                    max_weight = weights[vertex];
                    max_vertex = vertex;
                }
                idx++;
            }
        }
    }

    /****************************************
     * Ruge-Stuben Coarsening, Pass 2
     ****************************************/
    // For each vertex in fine:
    //    Make list of all coarse points in row S_vertex, coarse_vertex
    //    For each col in S_vertex:
    //        Max list of all coarse points in row S_col, coarse_col
    //        If no vertices in coarse_vertex are also in coarse_col, 
    //        remove vertex from fine, and add vertex to coarse
    for (int i = 0; i < local_num_rows; i++)
    {
        if (state[i] == 1) continue;

        head = -2;
        length = 0;

        // Add to coarse_cols all cols in row i such that Si,col is coarse
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = on_proc->idx2[j];
            if (state[col] == 1)
            {
                row_coarse[col] = head;
                head = col;
                length++;
            }
        }
        for (int j = start; j < end; j++)
        {
            col = on_proc->idx2[j];

            col_start = on_proc->idx1[col];
            col_end = on_proc->idx1[col+1];
            if (col_end - col_start)
            {
                for (idx = col_start; idx < col_end; idx++)
                {
                    col_k = on_proc->idx2[idx];
                    if (state[col_k] == 1 && row_coarse[col_k] != -1)
                    {
                        break;
                    }
                }
                if (idx == col_end)
                {
                    state[col] = 1;
                    continue;
                }
            }
        }
        for (int i = 0; i < length; i++)
        {
            tmp = head;
            head = row_coarse[tmp];
            row_coarse[tmp] = -1;
        }
    }

    /****************************************
     * Reset boundary states to unassigned
     ****************************************/
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (S->off_proc->idx1[i+1] - S->off_proc->idx1[i] == 0)
        {
            state[i] = -1;
        }
    }

    /****************************************
     * CLJP - Initialize weights and find 
     * initial neighbor states
     ****************************************/
    // Initialize weights to standard
    for (int i = 0; i < S->local_num_rows; i++)
    {
        weights[i] = (on_proc->idx1[i+1] - on_proc->idx1[i]) + 
                (off_proc->idx1[i+1] - off_proc->idx1[i]) +
                (((double)rand()) / RAND_MAX);
    }

    // Send Coarse States to neighbors
    // TODO -- currently sending all state info... is this necessary?
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = comm->send_data->indices[j];
            send_buffer[j] = states[idx];
        }
        MPI_Isend(&(send_buffer[start]), end - start, MPI_INT, proc, tag, 
                MPI_COMM_WORLD, &(comm->send_data->requests[i]));
    }
    // Recv coarse/fine states from neighbors
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Irecv(&(neighbor_states[start]), end - start, MPI_INT, proc, tag,
                MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
    }
    // Wait for communication to complete
    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, comm->send_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }
    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs, comm->recv_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }

    /****************************************
     * CLJP - Update initial weights based
     * on previously calculate C/F
     ****************************************/
    update_weights(on_proc_csc, off_proc_csc, states, neighbor_states, weights);
    


}


