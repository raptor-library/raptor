void ParCSRMatrix::print_mult(const aligned_vector<int>& proc_distances,
                const aligned_vector<int>& worst_proc_distances)
{
    int rank, rank_node, rank_socket;
    int ranks_per_socket;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_short = 0;
    int num_eager = 0;
    int num_rend = 0;
    int num_short_node = 0;
    int num_eager_node = 0;
    int num_rend_node = 0;
    int num_short_socket = 0;
    int num_eager_socket = 0;
    int num_rend_socket = 0;

    int size_short = 0;
    int size_eager = 0;
    int size_rend = 0;
    int size_short_node = 0;
    int size_eager_node = 0;
    int size_rend_node = 0;
    int size_short_socket = 0;
    int size_eager_socket = 0;
    int size_rend_socket = 0;

    long byte_hops = 0;
    long worst_byte_hops = 0;

    int short_cutoff = 500;
    int eager_cutoff = 8000;

    int start, end, size, idx;
    int proc, node, socket, n;

    // Check that communication package has been initialized
    if (comm == NULL)
    {
        comm = new ParComm(partition, off_proc_column_map, on_proc_column_map);
    }
    rank_node = comm->topology->get_node(rank);
    ranks_per_socket = comm->topology->PPN / 2;
    rank_socket = rank / ranks_per_socket;

    // Communicate data and multiply
    // Will communicate the rows of B based on comm
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        proc = comm->send_data->procs[i];
        node = comm->topology->get_node(proc);
        socket = proc / ranks_per_socket;
        size = (end - start)*sizeof(double);
        byte_hops += (size * proc_distances[proc]);
        worst_byte_hops += (size * worst_proc_distances[proc]);

        if (size < short_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_short_socket += size;
                    num_short_socket++;
                }
                else
                {
                    size_short_node += size;
                    num_short_node++;
                }
            }
            else
            {
                size_short += size;
                num_short++;
            }
        }
        else if (size < eager_cutoff)
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_eager_socket += size;
                    num_eager_socket++;
                }
                else
                {
                    size_eager_node += size;
                    num_eager_node++;
                }
            }
            else
            {
                size_eager += size;
                num_eager++;
            }
        }
        else
        {
            if (node == rank_node)
            {
                if (socket == rank_socket)
                {
                    size_rend_socket += size;
                    num_rend_socket++;
                }
                else
                {
                    size_rend_node += size;
                    num_rend_node++;
                }
            }
            else
            {
                size_rend += size;
                num_rend++;
            }
        }
    }

    int max_n;
    int max_s;
    long nl;
    n = num_short + num_eager + num_rend + num_short_node + num_eager_node + num_rend_node
            + num_short_socket + num_eager_socket + num_rend_socket;
    MPI_Allreduce(&n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Msgs: %d\n", max_n);
    if (n < max_n)
    {
        num_short = 0;
        num_eager = 0;
        num_rend = 0;
        num_short_node = 0;
        num_eager_node = 0;
        num_rend_node = 0;
        num_short_socket = 0;
        num_eager_socket = 0;
        num_rend_socket = 0;
    }

    long bytes;
    bytes = size_short + size_eager + size_rend;
    MPI_Reduce(&bytes, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Bytes = %ld\n", nl);

    n = size_short + size_eager + size_rend + size_short_node + size_eager_node
            + size_rend_node + size_short_socket + size_eager_socket + size_rend_socket;
    MPI_Allreduce(&n, &max_s, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (n < max_s)
    {
        size_short = 0;
        size_eager = 0;
        size_rend = 0;
        size_short_node = 0;
        size_eager_node = 0;
        size_rend_node = 0;
        size_short_socket = 0;
        size_eager_socket = 0;
        size_rend_socket = 0;
    }


    MPI_Reduce(&num_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short: %d\n", n);
    MPI_Reduce(&num_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager: %d\n", n);
    MPI_Reduce(&num_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend: %d\n", n);
    MPI_Reduce(&size_short, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short: %d\n", n);
    MPI_Reduce(&size_eager, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager: %d\n", n);
    MPI_Reduce(&size_rend, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend: %d\n", n);

    MPI_Reduce(&num_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Node: %d\n", n);
    MPI_Reduce(&num_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Node: %d\n", n);
    MPI_Reduce(&num_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Node: %d\n", n);
    MPI_Reduce(&size_short_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Node: %d\n", n);
    MPI_Reduce(&size_eager_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Node: %d\n", n);
    MPI_Reduce(&size_rend_node, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Node: %d\n", n);

    MPI_Reduce(&num_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Short Socket: %d\n", n);
    MPI_Reduce(&num_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Eager Socket: %d\n", n);
    MPI_Reduce(&num_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Rend Socket: %d\n", n);
    MPI_Reduce(&size_short_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Short Socket: %d\n", n);
    MPI_Reduce(&size_eager_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Eager Socket: %d\n", n);
    MPI_Reduce(&size_rend_socket, &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Size Rend Socket: %d\n", n);
    
    MPI_Reduce(&byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Byte Hops = %ld\n", nl);
    MPI_Reduce(&worst_byte_hops, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Worst Byte Hops = %ld\n", nl);
}
