// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "raptor.hpp"
#include <string.h>
#include <sstream>

using namespace raptor;

// Cut-off sizes for short and eager messages
#define short_cutoff 500
#define eager_cutoff 8000

// Functions used to help with message profiling
void print_internode_comm(Topology* topology, aligned_vector<int> num_msgs, aligned_vector<int> size_msgs,
        aligned_vector<int> node_size_msgs, FILE* filename)
{
    int n, max_n;
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = topology->get_node(rank);
    ranks_per_socket = topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / topology->PPN;

    // Number of processes each process talks to 
    n = num_msgs[6] + num_msgs[7] + num_msgs[8];
    fprintf(filename, "Num processes with which proc communicates: %d\n", n);
    //printf("Rank %d Rank Node %d Num processes with which proc communicates: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max num processes with which each proc communicates: %d\n", max_n);

    // Number of nodes each process talks to
    node_size_msgs[rank_node] = 0;
    n = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_size_msgs[i]) n++;
    }
    fprintf(filename, "Num nodes with which proc communicates: %d\n", n);
    //printf("Rank %d Rank Node %d Num nodes with which proc communicates: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max num nodes with with each process communicates: %d\n", max_n);

    // Number of nodes each node talks to
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, node_size_msgs.data(), num_nodes, RAPtor_MPI_INT,
            RAPtor_MPI_SUM, topology->local_comm);
    n = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_size_msgs[i]) n++;
    }
    fprintf(filename, "Num nodes with which node communicates: %d\n", n);
    //printf("Rank %d Rank Node %d Num nodes with which node communicates: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max num nodes with with each node communicates: %d\n", max_n);

    // Inter-node bytes sent by process
    // Procwise bytes (total sent by process)
    n = size_msgs[6] + size_msgs[7] + size_msgs[8];
    fprintf(filename, "Inter-node bytes sent by proc: %d\n", n);
    //printf("Rank %d Rank Node %d Inter-node bytes sent by proc: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max bytes sent by any process: %d\n", max_n);


    // Inter-node bytes sent by node
    n = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        // Max Node-Node Bytes (max between any set of nodes)
        if (node_size_msgs[i] > n)
            n = node_size_msgs[i];
    }
    fprintf(filename, "Max inter-node bytes sent beween this node and another node: %d\n", n);
    //printf("Rank %d Rank Node %d Max inter-node bytes sent beween this node and another node: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max bytes sent between any pair of nodes: %d\n", max_n);


    // Inter-node bytes sent by node
    n = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        // Nodewise Bytes (total sent by node)
        n += node_size_msgs[i];
    }
    fprintf(filename, "Max bytes sent by node: %d\n", n);
    //printf("Rank %d Rank Node %d Max bytes sent by node: %d\n", rank, rank_node, n);
    //RAPtor_MPI_Reduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max bytes sent by any node: %d\n", max_n);
     
}

void print_comm(aligned_vector<int>& num_msgs, aligned_vector<int>& size_msgs, 
        aligned_vector<int>& node_size_msgs, FILE* filename)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    int n_arch_types = 3;
    int n_protocols = 3;
    const char* arch_labels[3] = {" Socket", " Node", ""};
    const char* protocol_labels[3] = {"Short", "Eager", "Rend"};

    int n, max_n;
    long nl, sum_nl;

    // Total Number of Messages
    nl = 0;
    n = 0;
    for (aligned_vector<int>::iterator it = num_msgs.begin(); it != num_msgs.end(); ++it)
    {
        nl += *it;
        n += *it;
    }
    fprintf(filename, "Total Num Msgs Sent by Proc: %ld\n", nl);
    //printf("Rank %d Rank Node -- Total Num Msgs Sent by Proc: %ld\n", rank, nl);

    // Sum Messages Among Processes
    //RAPtor_MPI_Reduce(&nl, &sum_nl, 1, RAPtor_MPI_LONG, RAPtor_MPI_SUM, 0, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Total Num Msgs: %ld\n", sum_nl);

    // Max Number Among Processes
    //RAPtor_MPI_Allreduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max Num Msgs: %d\n", max_n);
    /*if (n < max_n)
    {
        for (aligned_vector<int>::iterator it = num_msgs.begin(); it != num_msgs.end(); ++it)
        {
            *it = 0;
        }
    }*/

    // Total Size of Messages
    nl = 0;
    n = 0;
    for (aligned_vector<int>::iterator it = size_msgs.begin(); it != size_msgs.end(); ++it)
    {
        nl += *it;
        n += *it;
    }

    fprintf(filename, "Total Size Msgs for Proc: %ld\n", nl);
    //printf("Rank %d Rank Node -- Total Size Msgs for Proc: %ld\n", rank, nl);
    // Sum Message Sizes Among Processes
    //RAPtor_MPI_Allreduce(&nl, &sum_nl, 1, RAPtor_MPI_LONG, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Total Size Msgs: %ld\n", sum_nl);

    // Max Size Among Processes
    //RAPtor_MPI_Allreduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, RAPtor_MPI_COMM_WORLD);
    //if (rank == 0) printf("Max Size Msgs: %d\n", max_n);
    if (n < max_n)
    {
        for (aligned_vector<int>::iterator it = size_msgs.begin(); it != size_msgs.end(); ++it)
        {
            *it = 0;
        }
    }

    aligned_vector<int> max_num_msgs(num_msgs.size());
    aligned_vector<int> max_size_msgs(size_msgs.size());
    RAPtor_MPI_Reduce(num_msgs.data(), max_num_msgs.data(), num_msgs.size(), RAPtor_MPI_INT,
            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);
    RAPtor_MPI_Reduce(size_msgs.data(), max_size_msgs.data(), size_msgs.size(), RAPtor_MPI_INT,
            RAPtor_MPI_MAX, 0, RAPtor_MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < n_arch_types; i++)
        {
            for (int j = 0; j < n_protocols; j++)
            {
                fprintf(filename, "Num %s%s: %d\n", arch_labels[i], protocol_labels[j], 
                        max_num_msgs[i*n_protocols + j]);
                fprintf(filename, "Size %s%s: %d\n", arch_labels[i], protocol_labels[j], 
                        max_size_msgs[i*n_protocols + j]);
            }
        }
    }
}

void calc_and_print(Topology* topology, CommData* comm_data, FILE* filename)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = topology->get_node(rank);
    ranks_per_socket = topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / topology->PPN;

    int n_arch_types = 3;
    int n_protocols = 3;
    aligned_vector<bool> arch_types(n_arch_types, true);
    aligned_vector<bool>protocol(n_protocols, true);

    int start, end;
    int proc, node, socket;
    int size;

    aligned_vector<int> num_msgs(n_arch_types * n_protocols, 0);
    aligned_vector<int> size_msgs(n_arch_types * n_protocols, 0); 
    // Contains the total bytes of all messages sent to node i from this process
    aligned_vector<int> node_size_msgs(num_nodes, 0);

    for (int i = 0; i < comm_data->num_msgs; i++)
    {
        start = comm_data->indptr[i];
        end = comm_data->indptr[i+1];
        proc = comm_data->procs[i];
        node = topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start) * sizeof(double);

        arch_types[0] = (socket == rank_socket);
        arch_types[1] = (node == rank_node);
        protocol[0] = size < short_cutoff;
        protocol[1] = size < eager_cutoff;

        // Add to node size
        node_size_msgs[node] += size;

        for (int j = 0; j < n_arch_types; j++)
        {
            if (arch_types[j])
            {
                for (int k = 0; k < n_protocols; k++)
                {
                    if (protocol[k])
                    {
                        size_msgs[j*n_arch_types + k] += size;
                        num_msgs[j*n_arch_types + k]++;
                        break;
                    }
                } 
                break;
            }
        }
    }

    print_internode_comm(topology, num_msgs, size_msgs, node_size_msgs, filename);
    print_comm(num_msgs, size_msgs, node_size_msgs, filename);
}

void calc_and_print_all_messages(Topology* topology, CommData* comm_data, FILE* filename)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = topology->get_node(rank);
    ranks_per_socket = topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / topology->PPN;

    int n_arch_types = 3;
    int n_protocols = 3;
    aligned_vector<bool> arch_types(n_arch_types, true);
    aligned_vector<bool>protocol(n_protocols, true);

    int start, end;
    int proc, node, socket;
    int size;

    aligned_vector<int> num_msgs(n_arch_types * n_protocols, 0);
    aligned_vector<int> size_msgs(n_arch_types * n_protocols, 0); 
    // Contains the total bytes of all messages sent to node i from this process
    aligned_vector<int> node_size_msgs(num_nodes, 0);

    for (int i = 0; i < comm_data->num_msgs; i++)
    {
        start = comm_data->indptr[i];
        end = comm_data->indptr[i+1];
        proc = comm_data->procs[i];
        node = topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start) * sizeof(double);

        arch_types[0] = (socket == rank_socket);
        arch_types[1] = (node == rank_node);
        protocol[0] = size < short_cutoff;
        protocol[1] = size < eager_cutoff;

        // Add to node size
        node_size_msgs[node] += size;

        for (int j = 0; j < n_arch_types; j++)
        {
            if (arch_types[j])
            {
                for (int k = 0; k < n_protocols; k++)
                {
                    if (protocol[k])
                    {
                        size_msgs[j*n_arch_types + k] += size;
                        num_msgs[j*n_arch_types + k]++;
                        break;
                    }
                } 
                break;
            }
        }
    }

    print_internode_comm(topology, num_msgs, size_msgs, node_size_msgs, filename);
    print_comm(num_msgs, size_msgs, node_size_msgs, filename);
}

void print_all_messages(Topology* topology, CommData* comm_data, FILE* filename)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = topology->get_node(rank);
    ranks_per_socket = topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / topology->PPN;

    int start, end;
    int proc, node, socket;
    int size;

    // Contains the total bytes of all messages sent to node i from this process
    aligned_vector<int> node_size_msgs(num_nodes, 0);

    for (int i = 0; i < comm_data->num_msgs; i++)
    {
        start = comm_data->indptr[i];
        end = comm_data->indptr[i+1];
        proc = comm_data->procs[i];
        node = topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start) * sizeof(double);

        fprintf(filename, "Size %ld Proc %d Node %d Socket %d\n", size, proc, node, socket);
    }
}

int get_idx(NonContigData* comm_data, int j)
{
    return comm_data->indices[j];
}
int get_idx(ContigData* comm_data, int j)
{
    return j;
}

// Main function to run all message profilings
int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);

    //setenv("PPN", "16", 1);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (_argc < 3)
    {
        printf("Usage: <mat#> <nrhs>\n");
        exit(-1);
    }

    // Grab command line arguments
    int mat = atoi(_argv[1]);
    int nrhs = atoi(_argv[2]);

    // Matrix market filenames
    const char* mat1 = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    const char* mat2 = "../../../../../mtx_market_matrices/G3_circuit.pm";
    const char* mat3 = "../../../../../mtx_market_matrices/Hook_1498.pm";

    // Convert rank to string
    std::stringstream ss;
    ss << rank;
    string p_string = ss.str();

    // Process filename
    char filename[150];
    if (mat == 1)
    {
        //strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs256/p");
        //strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs512/p");
        //strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs1024/p");
        //strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs2048/p");
        //strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs4096/p");
        strcpy(filename, "../../../../../modelling_output/mfem_dg_diffusion/2step/procs8192/p");
    }
    else if (mat == 2)
    {
        //strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs256/p");
        //strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs512/p");
        //strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs1024/p");
        //strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs2048/p");
        //strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs4096/p");
        strcpy(filename, "../../../../../modelling_output/g3circuit/2step/procs8192/p");
    }
    else if (mat == 3)
    {
        //strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs256/p");
        //strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs512/p");
        //strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs1024/p");
        //strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs2048/p");
        //strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs4096/p");
        strcpy(filename, "../../../../../modelling_output/hook1498/2step/procs8192/p");
    }
    strcat(filename, p_string.c_str());
    strcat(filename, ".out");

    // Read in matrix
    ParCSRMatrix* A; 
    if (mat == 1)
    {
        A = readParMatrix(mat1);
    }
    else if (mat == 2)
    {
        A = readParMatrix(mat2);
    }
    else if (mat == 3)
    {
        A = readParMatrix(mat3);
    }

    // Declare vectors and residual variables
    ParBVector x(A->global_num_rows, A->local_num_rows, nrhs);
    ParBVector b(A->global_num_rows, A->local_num_rows, nrhs);

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);
    
    // Initial call before init_profile called
    x.set_const_value(1.0);
    // Perform bvspmv to make sure communicator is setup
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    A->tap_mult(x, b);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //calc_and_print(A->comm->topology, A->comm->send_data);

    // Print message profiling
    // Print out residuals to file
    FILE *f;
    f = fopen(filename, "w");
    // Comm for redistributing inter-node messages received
    fprintf(f, "local_R_par_comm\n"); 
    print_all_messages(A->tap_comm->topology, A->tap_comm->local_R_par_comm->send_data, f);
    // Comm for inter-node communication
    fprintf(f, "global_par_comm\n"); 
    print_all_messages(A->tap_comm->topology, A->tap_comm->global_par_comm->send_data, f);
    // Comm for on-node communication 
    fprintf(f, "local_L_par_comm\n"); 
    print_all_messages(A->tap_comm->topology, A->tap_comm->local_L_par_comm->send_data, f);
    fclose(f);
    
    //setenv("PPN", "16", 1);
    
    delete A;
    MPI_Finalize();
    return 0;

} // end of main() //
