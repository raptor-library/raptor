void calc_mult(Topology* topology, CommData* comm_data, aligned_vector<int>& num_msgs, 
        aligned_vector<int>& size_msgs, aligned_vector<int>& proc_node_info)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = comm->topology->get_node(rank);
    ranks_per_socket = comm->topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / comm->topology->PPN;

    int n_arch_types = 3;
    int n_protocols = 3;
    aligned_vector<bool> arch_types(n_arch_types, true);
    aligned_vector<bool>protocol(n_protocols, true);

    aligned_vector<int> node_size_msgs(num_nodes, 0);

    int start, end;
    int proc, node, socket;
    int size;

    num_msgs.resize(n_arch_types * n_protocols, 0);
    size_msgs.resize(n_arch_types * n_protocols, 0); 
    proc_node_info.resize(6, 0);

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
            if (arch_type[j])
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

    // Number of processes each process talks to 
    proc_node_info[0] = num_msgs[6] + num_msgs[7] + num_msgs[8];

    // Number of nodes each process talks to
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;
        if (node_size_msgs[i]) proc_node_info[1]++;
    }

    // Number of nodes each node talks to
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, node_size_msgs.data(), num_nodes, MPI_INT,
            MPI_SUM, comm_data->topology->local_comm);
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;
        if (node_size_msgs[i]) proc_node_info[2]++;
    }

    // Inter-node bytes sent by process
    // Procwise bytes (total sent by process)
    proc_node_info[3] = size_msgs[6] + size_msgs[7] + size_msgs[8];

    // Inter-node bytes sent by node
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;

        // Max Node-Node Bytes (max between any set of nodes)
        if (node_size_msgs[i] > proc_node_info[4])
            proc_node_info[4] = node_size_msgs[i];

        // Nodewise Bytes (total sent by node)
        proc_node_info[5] += node_size_msgs[i];
    }
}

void calc_mult(CSRMatrix* A, CSRMatrix* B, Topology* topology, CommData* comm_data, aligned_vector<int>& num_msgs, 
        aligned_vector<int>& size_msgs, aligned_vector<int>& proc_node_info)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    int num_procs, num_nodes;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = comm->topology->get_node(rank);
    ranks_per_socket = comm->topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;
    num_nodes = num_procs / comm->topology->PPN;

    int n_arch_types = 3;
    int n_protocols = 3;
    aligned_vector<bool> arch_types(n_arch_types, true);
    aligned_vector<bool>protocol(n_protocols, true);

    aligned_vector<int> node_size_msgs(num_nodes, 0);

    int start, end, proc, node, socket;
    int size, idx;

    num_msgs.resize(n_arch_types * n_protocols, 0);
    size_msgs.resize(n_arch_types * n_protocols, 0); 
    proc_node_info.resize(6, 0);

    for (int i = 0; i < comm_data->num_msgs; i++)
    {
        start = comm_data->indptr[i];
        end = comm_data->indptr[i+1];
        proc = comm_data->procs[i];
        node = topology->get_node(proc);
        socket = proc / ranks_per_socket;

        size = 0;
        for (int j = start; j < end; j++)
        {
            idx = comm_data->indices[j];
            size += (A->idx1[idx+1] - A->idx1[idx]);
            if (B)
                size += (B->idx1[idx+1] - B->idx1[idx]);
        }
        size = size * (2*sizeof(int) + sizeof(double));

        arch_types[0] = (socket == rank_socket);
        arch_types[1] = (node == rank_node);
        protocol[0] = size < short_cutoff;
        protocol[1] = size < eager_cutoff;

        // Add to node size
        node_size_msgs[node] += size;

        for (int j = 0; j < n_arch_types; j++)
        {
            if (arch_type[j])
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

    // Number of processes each process talks to 
    proc_node_info[0] = num_msgs[6] + num_msgs[7] + num_msgs[8];

    // Number of nodes each process talks to
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;
        if (node_size_msgs[i]) proc_node_info[1]++;
    }

    // Number of nodes each node talks to
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, node_size_msgs.data(), num_nodes, MPI_INT,
            MPI_SUM, comm_data->topology->local_comm);
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;
        if (node_size_msgs[i]) proc_node_info[2]++;
    }

    // Inter-node bytes sent by process
    // Procwise bytes (total sent by process)
    proc_node_info[3] = size_msgs[6] + size_msgs[7] + size_msgs[8];

    // Inter-node bytes sent by node
    for (int i = 0; i < num_nodes; i++)
    {
        if (i == rank_node) continue;

        // Max Node-Node Bytes (max between any set of nodes)
        if (node_size_msgs[i] > proc_node_info[4])
            proc_node_info[4] = node_size_msgs[i];

        // Nodewise Bytes (total sent by node)
        proc_node_info[5] += node_size_msgs[i];
    }
}

void print_comm(aligned_vector<int>& num_msgs, aligned_vector<int>& size_msgs, 
        aligned_vector<int>& proc_node_info)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    int n_arch_types = 3;
    int n_protocols = 3;
    int n_proc_node_info = 6;
    char* arch_labels[3] = {" Socket", " Node", ""};
    char* protocol_labels[3] = {"Short", "Eager", "Rend"};
    char* proc_node_labels[6] = {"Max Proc-Proc Msgs", "Max Proc-Node Msgs",
        "Max Node-Node Msgs", "Max Procwise Bytes", "Max Node-Node Bytes", 
        "Max Nodewise Bytes"};

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

    // Sum Messages Among Processes
    RAPtor_MPI_Reduce(&nl, &sum_nl, 1, RAPtor_MPI_LONG, RAPtor_MPI_SUM, 0, RAPtor_MPI_COMM_WORLD);
    if (rank == 0) printf("Total Num Msgs: %ld\n", sum_n);

    // Max Number Among Processes
    RAPtor_MPI_Allreduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, RAPtor_MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Msgs: %d\n", max_n);
    if (n < max_n)
    {
        for (aligned_vector<int>::iterator it = num_msgs.begin(); it != num_msgs.end(); ++it)
        {
            *it = 0;
        }
    }

    // Total Size of Messages
    nl = 0;
    n = 0;
    for (aligned_vector<int>::iterator it = size_msgs.begin(); it != size_msgs.end(); ++it)
    {
        nl += *it;
        n += *it;
    }

    // Sum Message Sizes Among Processes
    RAPtor_MPI_Allreduce(&nl, &sum_nl, 1, RAPtor_MPI_LONG, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    if (rank == 0) printf("Total Size Msgs: %ld\n", sum_n);

    // Max Size Among Processes
    RAPtor_MPI_Allreduce(&n, &max_n, 1, RAPtor_MPI_INT, RAPtor_MPI_MAX, RAPtor_MPI_COMM_WORLD);
    if (rank == 0) printf("Max Size Msgs: %d\n", max_n);
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
                printf("Num %s%s: %d\n", arch_labels[i], protocol_labels[j], 
                        max_num_msgs[i*n_protocols + j]);
                printf("Size %s%s: %d\n", arch_labels[i], protocol_labels[j], 
                        max_size_msgs[i*n_protocols + j]);
            }
        }
    }

    if (rank == 0)
    {
        for (int i = 0; i < n_proc_node_info; i++)
        {
            printf("%s: %d\n", proc_node_labels[i], proc_node_info[i]);
        }
    }
}


void ParCSRMatrix::print_mult(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<int> num_msgs;
    aligned_vector<int> size_msgs;

    calc_mult(comm->topology, comm->send_data, num_msgs, size_msgs);
    print_comm(num_msgs, size_msgs); 
}

void ParCSRMatrix::print_mult_T(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<int> num_msgs;
    aligned_vector<int> size_msgs;

    calc_mult(comm->topology, comm->recv_data, num_msgs, size_msgs);
    print_comm(num_msgs, size_msgs); 
}

void ParCSRMatrix::print_mult(ParCSRMatrix* B, const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }

    aligned_vector<int> num_msgs;
    aligned_vector<int> size_msgs;

    calc_mult(B->on_proc, B->off_proc, comm->topology, comm->send_data, num_msgs, size_msgs);
    print_comm(num_msgs, size_msgs); 

}


void ParCSRMatrix::print_mult_T(ParCSCMatrix* A, const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    // Check that communication package has been initialized
    if (A->comm == NULL)
    {
        A->comm = new ParComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }

    CSRMatrix* Ctmp = mult_T_partial(A);

    aligned_vector<int> num_msgs;
    aligned_vector<int> size_msgs;

    calc_mult(Ctmp, NULL, A->comm->topology, A->comm->recv_data, num_msgs, size_msgs);
    print_comm(num_msgs, size_msgs); 

    delete Ctmp;
}


