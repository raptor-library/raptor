// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "NAP_partition.hpp"
#include "util/linalg/repartition.hpp"

void partition(CSRMatrix* A, int n_parts, 
        aligned_vector<int>& parts)
{
    int nvtxs = A->n_rows; // # vertices in graph
    int ncon = 1; // Number of balancing constraints
    int* xadj = A->idx1.data(); // Indptr
    int* adjncy = A->idx2.data(); // Indices

    int adjweights[A->nnz];
    for (int i = 0; i < A->nnz; i++)
        adjweights[i] = A->vals[i];

    int options[METIS_NOPTIONS];
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG] = 0;
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_MINCONN] = 0;
    //int* options = NULL;

    int objval;
    if (nvtxs) parts.resize(nvtxs);
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjweights, 
            &n_parts, NULL, NULL, options, &objval, parts.data());
}


// Three-steps: partition on-process, then on-node, and then among group of PPN nodes
ParCSRMatrix* NAP_partition(ParCSRMatrix* A_tmp, aligned_vector<int>& new_rows)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end, col, row, val;
    int ctr = 0;
    int tag = 3492;
    int count, size, idx, pos;
    int part, first_part;
    int first_nr_part, last_nr_part;
    int col_start, col_end;
    int row_start, row_end;
    int col_part, local_part;
    int part_idx, row_part;

    aligned_vector<int> nr_list;
    aligned_vector<int> master_list;
    aligned_vector<int> neighbor_master_list;
    int nr = A_tmp->partition->topology->PPN;
    int master = (rank / nr) * nr;
    int neighbor_master = ((rank / nr) + 1) * nr;
    nr_list.push_back(nr);
    master_list.push_back(master);
    neighbor_master_list.push_back(neighbor_master);
    int mat_size = A_tmp->local_num_rows;

    aligned_vector<int> sizes;

    MPI_Status recv_status;

    CSCMatrix* A_off_csc = A_tmp->off_proc->to_CSC();

    CSRMatrix* A_combined = NULL;
    aligned_vector<int> A_parts;
    aligned_vector<int> A2_parts;

    // Make serial matrix (data = sum of edges / off-proc neighbors)
    CSRMatrix* A_serial = new CSRMatrix(A_tmp->local_num_rows, A_tmp->on_proc_num_cols);
    aligned_vector<int> row_vals(A_tmp->local_num_rows, 0);
    A_serial->idx1[0] = 0;
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        start = A_tmp->on_proc->idx1[i];
        end = A_tmp->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->on_proc->idx2[j];
            if (row_vals[col] == 0)
            {   
                A_serial->idx2.push_back(col);
            }
            row_vals[col]++;
        }
        start = A_tmp->off_proc->idx1[i];
        end = A_tmp->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->off_proc->idx2[j];
            col_start = A_off_csc->idx1[col];
            col_end = A_off_csc->idx1[col+1];
            for (int k = col_start; k < col_end; k++)
            {
                row = A_off_csc->idx2[k];
                if (row_vals[row] == 0) 
                {
                    A_serial->idx2.push_back(row);
                }
                row_vals[row]++;
            }
        }
        A_serial->idx1[i+1] = A_serial->idx2.size();

        start = A_serial->idx1[i];
        end = A_serial->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_serial->idx2[j];
            A_serial->vals.push_back(row_vals[col]);
            row_vals[col] = 0;
        }
    }
    A_serial->nnz = A_serial->idx2.size();

    // Partition serial matrix
    int n_parts = (A_tmp->global_num_rows / num_procs) / nr;
    partition(A_serial, n_parts, A_parts);
    delete A_serial;

    first_part = rank * n_parts;
    for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
        *it += first_part;

    // Send off_proc parts
    aligned_vector<int>& off_parts = A_tmp->comm->communicate(A_parts);

    // Form matrix 'Part' that points to partitions
    CSRMatrix* Part = new CSRMatrix(n_parts, n_parts);
    sizes.resize(n_parts);
    std::fill(sizes.begin(), sizes.end(), 0);
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        sizes[local_part]++;
    }
    Part->idx1[0] = 0;
    for (int i = 0; i < n_parts; i++)
    {
        Part->idx1[i+1] = Part->idx1[i] + sizes[i];
        sizes[i] = 0;
    }
    Part->nnz = Part->idx1[n_parts];
    Part->idx2.resize(Part->nnz);
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        idx = Part->idx1[local_part] + sizes[local_part]++;
        Part->idx2[idx] = i;
    }

    // If part is in 'nr', make value local (subtract first_nr)
    // Otherwise, make part negative
    mat_size = n_parts * nr; // update mat_size (only necessary it n_rows does 
                             // not evenly divide the number of processes)
    first_nr_part = ((rank / nr)*nr)*n_parts;
    last_nr_part = (((rank / nr) + 1)*nr)*n_parts - 1;
    for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
        *it -= first_nr_part;
    for (aligned_vector<int>::iterator it = off_parts.begin(); it != off_parts.end(); ++it)
    {
        if (*it >= first_nr_part && *it <= last_nr_part)
            *it -= first_nr_part;
        else *it *= -1;
    }

    // For all parts not in 'nr', add position to map
    // Create off_nr_idx, containing local parts in each 
    int n_off_nr = off_parts.size();
    std::map<int, int> off_nr_map;
    sizes.resize(n_off_nr);
    std::fill(sizes.begin(), sizes.end(), 0);
    CSRMatrix* OffPart = new CSRMatrix(n_off_nr, n_off_nr);
    ctr = 0;
    for (int i = 0; i < A_tmp->off_proc_num_cols; i++)
    {
        part = off_parts[i];
        if (part < 0)
        {
            if (off_nr_map.find(part) == off_nr_map.end())
            {
                off_nr_map[part] = ctr++;
            }
            idx = off_nr_map[part];
            start = A_off_csc->idx1[i];
            end = A_off_csc->idx1[i+1];
            sizes[idx] += (end - start);
        }
    }
    OffPart->n_rows = ctr;
    OffPart->n_cols = ctr;
    OffPart->idx1.resize(ctr+1);
    OffPart->idx1[0] = 0;
    for (int i = 0; i < ctr; i++)
    {
        OffPart->idx1[i+1] = OffPart->idx1[i] + sizes[i];
        sizes[i] = 0;
    }
    OffPart->nnz = OffPart->idx1[ctr];
    OffPart->idx2.resize(OffPart->nnz);
    for (int i = 0; i < A_tmp->off_proc_num_cols; i++)
    {
        part = off_parts[i];
        if (part < 0)
        {
            idx = off_nr_map[part];
            start = A_off_csc->idx1[i];
            end = A_off_csc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = A_off_csc->idx2[j];
                part = A_parts[row];
                if (sizes[idx] == 0)
                {
                    pos = OffPart->idx1[idx] + sizes[idx]++;
                    OffPart->idx2[idx] = part;
                }
            }
        }
    }
    ctr = 0;
    start = OffPart->idx1[0];
    for (int i = 0; i < OffPart->n_rows; i++)
    {
        part = off_parts[i];
        idx = off_nr_map[part];
        end = start + sizes[idx];
        for (int j = start; j < end; j++)
        {
            OffPart->idx2[ctr++] = OffPart->idx2[j];
        }
        start = OffPart->idx1[i+1];
        OffPart->idx1[i+1] = ctr;
    }
    OffPart->nnz = ctr;


    row_vals.resize(mat_size);
    std::fill(row_vals.begin(), row_vals.end(), 0);
    aligned_vector<int> mat;
    aligned_vector<int> row_off_parts(OffPart->n_rows, 0);
    aligned_vector<int> row_off_ptr(OffPart->n_rows);
    ctr = 0;
    if (rank == master_list[0])
    {
        A_combined = new CSRMatrix(mat_size, mat_size);
        A_combined->idx1[0] = 0;
        for (int i = 0; i < n_parts; i++)
        {
            ctr = 0;
            start = Part->idx1[i];
            end = Part->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = Part->idx2[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        A_combined->idx2.push_back(col_part);
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (col_part > 0)
                    {
                        if (row_vals[col_part] == 0)
                        {
                            A_combined->idx2.push_back(col_part);
                        }
                        row_vals[col_part]++;
                    }
                    else  // Add shared neighbor partitions to list
                    {
                        part_idx = off_nr_map[col_part];
                        if (row_off_parts[part_idx] == 0)
                        {
                            row_off_ptr[ctr++] = part_idx;
                            row_off_parts[part_idx] = 1;
                        }
                    }
                }
            }

            // Add weights for shared neighbor partitions
            for (int j = 0; j < ctr; j++)
            {
                part = row_off_ptr[j];
                start = OffPart->idx1[part];
                end = OffPart->idx1[part+1];
                for (int k = start; k < end; k++)
                {
                    row_part = OffPart->idx2[k];
                    if (row_vals[row_part] == 0)
                    {
                        A_combined->idx2.push_back(row_part);
                    }
                    row_vals[row_part]++;
                }
                row_off_parts[part] = 0;
            }


            A_combined->idx1[i+1] = A_combined->idx2.size();

            start = A_combined->idx1[i];
            end = A_combined->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_combined->idx2[j];
                A_combined->vals.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
        }

        int idx_ctr = n_parts+1;
        for (int i = master+1; i < neighbor_master; i++)
        {
            MPI_Probe(i, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > mat.size()) mat.resize(count);
            MPI_Recv(mat.data(), count, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_status);
            ctr = 0;
            while (ctr < count)
            {
                size = mat[ctr++];
                for (int j = 0; j < size; j++)
                {
                    A_combined->idx2.push_back(mat[ctr++]);
                    A_combined->vals.push_back(mat[ctr++]);
                }
                A_combined->idx1[idx_ctr++] = A_combined->idx2.size();
            }
        }
        A_combined->nnz = A_combined->idx1[A_combined->n_rows];
        A_combined->n_rows = mat_size;

        A2_parts.resize(mat_size);
        partition(A_combined, A_tmp->partition->topology->PPN, A2_parts);
        delete A_combined;

        start = n_parts;
        for (int i = master + 1; i < neighbor_master; i++)
        {
            MPI_Send(&(A2_parts[start]), n_parts, MPI_INT, i, tag, MPI_COMM_WORLD);
            start += n_parts;
        }
        A2_parts.resize(n_parts);
    }
    else
    {
        aligned_vector<int> mat_parts(mat_size, 0);
        int mat_ctr = 0;
        for (int i = 0; i < n_parts; i++)
        {
            ctr = 0;
            start = Part->idx1[i];
            end = Part->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = Part->idx2[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        mat_parts[mat_ctr++] = col_part;
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (col_part >= 0)
                    {
                        if (row_vals[col_part] == 0)
                        {
                            mat_parts[mat_ctr++] = col_part;
                        }
                        row_vals[col_part]++;
                    }
                    else
                    {
                        part_idx = off_nr_map[col_part];
                        if (row_off_parts[part_idx] == 0)
                        {
                            row_off_ptr[ctr++] = part_idx;
                            row_off_parts[part_idx] = 1;
                        }
                    }
                }
            }

            for (int j = 0; j < ctr; j++)
            {
                part = row_off_ptr[j];
                start = OffPart->idx1[part];
                end = OffPart->idx1[part+1];
                for (int k = start; k < end; k++)
                {
                    row_part = OffPart->idx2[k];
                    if (row_vals[row_part] == 0)
                    {
                        mat_parts[mat_ctr++] = row_part;
                    }
                    row_vals[row_part]++;
                }
                row_off_parts[part] = 0;
            }

            mat.push_back(mat_ctr);
            for (int j = 0; j < mat_ctr; j++)
            {
                col = mat_parts[j];
                mat.push_back(col);
                mat.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
            mat_ctr = 0;
        }

        MPI_Send(mat.data(), mat.size(), MPI_INT, master, tag, MPI_COMM_WORLD);

        A2_parts.resize(n_parts);
        MPI_Recv(A2_parts.data(), n_parts, MPI_INT, master, tag, 
                MPI_COMM_WORLD, &recv_status);
    }

    delete OffPart;
    delete Part;

    // Update original A_parts with new partition info
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part + first_nr_part - first_part;
        A_parts[i] = A2_parts[local_part];
    }

    // Update nr, and update parts to make unique in nr
    first_part = (rank / nr) * n_parts;
    nr *= A_tmp->partition->topology->PPN;
    master_list.push_back((rank / nr)*nr);
    neighbor_master_list.push_back(((rank / nr) + 1) * nr);
    for (aligned_vector<int>::iterator it = A_parts.begin();
            it != A_parts.end(); ++it)
    {
        *it += first_part;
    }

    // Send off_proc parts
    off_parts = A_tmp->comm->communicate(A_parts);

    // Form matrix 'Part' that points to partitions
    Part = new CSRMatrix(n_parts, n_parts);
    sizes.resize(n_parts);
    std::fill(sizes.begin(), sizes.end(), 0);
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        sizes[local_part]++;
    }
    Part->idx1[0] = 0;
    for (int i = 0; i < n_parts; i++)
    {
        Part->idx1[i+1] = Part->idx1[i] + sizes[i];
        sizes[i] = 0;
    }
    Part->nnz = Part->idx1[n_parts];
    Part->idx2.resize(Part->nnz);
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        idx = Part->idx1[local_part] + sizes[local_part]++;
        Part->idx2[idx] = i;
    }

    first_nr_part = (rank / nr)*n_parts;
    last_nr_part = ((rank / nr) + 1)*n_parts - 1;
    for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
        *it -= first_nr_part;
    for (aligned_vector<int>::iterator it = off_parts.begin(); it != off_parts.end(); ++it)
        *it -= first_nr_part;


    row_vals.resize(mat_size, 0);
    mat.clear();
    ctr = 0;
    if (rank == master_list[0])
    {
        A_combined = new CSRMatrix(mat_size, mat_size);
        A_combined->idx1[0] = 0;
        for (int i = 0; i < n_parts; i++)
        {
            start = Part->idx1[i];
            end = Part->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = Part->idx2[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        A_combined->idx2.push_back(col_part);
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        A_combined->idx2.push_back(col_part);
                    }
                }
            }

            A_combined->idx1[i+1] = A_combined->idx2.size();

            start = A_combined->idx1[i];
            end = A_combined->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_combined->idx2[j];
                A_combined->vals.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
        }

        int idx_ctr = n_parts+1;
        for (int i = master_list[0]+1; i < neighbor_master_list[0]; i++)
        {
            MPI_Probe(i, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > mat.size()) mat.resize(count);
            MPI_Recv(mat.data(), count, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_status);
            ctr = 0;
            while (ctr < count)
            {
                size = mat[ctr++];
                for (int j = 0; j < size; j++)
                {
                    A_combined->idx2.push_back(mat[ctr++]);
                    A_combined->vals.push_back(mat[ctr++]);
                }
                A_combined->idx1[idx_ctr++] = A_combined->idx2.size();
            }
        }

        // Combine rows (all mats have duplicate rows)
        row_vals.resize(mat_size);
        std::fill(row_vals.begin(), row_vals.end(), 0);
        sizes.resize(n_parts);
        ctr = 0;
        for (int i = 0; i < n_parts; i++)
        {
            for (int j = 0; j < nr; j++)
            {
                start = A_combined->idx1[j*n_parts + i];
                end = A_combined->idx1[j*n_parts + i + 1];
                for (int k = start; k < end; k++)
                {
                    col = A_combined->idx2[k];
                    if (row_vals[col] == 0)
                    {
                        A_combined->idx2[ctr++] = col;
                    }
                    row_vals[col] += A_combined->vals[k];
                }
            }
            sizes[i] = ctr;
        }
        for (int i = 0; i < n_parts; i++)
        {
            A_combined->idx1[i+1] = sizes[i];
        }

        if (rank == master_list[1])
        {
            printf("Rank %d, n_parts %d\n", rank, n_parts);
            idx_ctr = n_parts+1;
            for (int i = master_list[1]+nr_list[0]; i < neighbor_master_list[1]; i += nr_list[0])
            {
                printf("i %d\n", i);
                MPI_Probe(i, tag, MPI_COMM_WORLD, &recv_status);
                MPI_Get_count(&recv_status, MPI_INT, &count);
                if (count > mat.size()) mat.resize(count);
                MPI_Recv(mat.data(), count, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_status);
                ctr = 0;
                while (ctr < count)
                {
                    size = mat[ctr++];
                    for (int j = 0; j < size; j++)
                    {
                        A_combined->idx2.push_back(mat[ctr++]);
                        A_combined->vals.push_back(mat[ctr++]);
                    }
                    A_combined->idx1[idx_ctr++] = A_combined->idx2.size();
                }
            }
            A_combined->n_rows = idx_ctr;
            A_combined->nnz = A_combined->idx1[idx_ctr];
            printf("N rows %d, mat size %d\n", idx_ctr, mat_size);
            A2_parts.resize(A_combined->n_rows);
            partition(A_combined, A_tmp->partition->topology->PPN, A2_parts);
            start = n_parts;
            for (int i = master_list[1]+nr_list[0]; i < neighbor_master_list[1]; i += nr_list[0])
            {
                MPI_Send(&(A2_parts[start]), n_parts, MPI_INT, i, tag, MPI_COMM_WORLD);
                start += n_parts;
            }
            A2_parts.resize(n_parts);
        }
        else
        {
            mat.clear();
            for (int i = 0; i < n_parts; i++)
            {
                start = A_combined->idx1[i];
                end = A_combined->idx1[i+1];
                mat.push_back(end - start);
                for (int j = start; j < end; j++)
                {
                    mat.push_back(A_combined->idx2[j]);
                    mat.push_back(A_combined->vals[j]);
                }
            }
            MPI_Send(mat.data(), mat.size(), MPI_INT, master_list[1], tag, MPI_COMM_WORLD);

            A2_parts.resize(n_parts);
          //  MPI_Recv(A2_parts.data(), n_parts, MPI_INT, master_list[1], tag, 
          //          MPI_COMM_WORLD, &recv_status);
        }

        for (int i = master_list[0]+1; i < neighbor_master_list[0]; i++)
        {
            MPI_Send(A2_parts.data(), n_parts, MPI_INT, i, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        mat.clear();
        aligned_vector<int> mat_parts(mat_size, 0);
        int mat_ctr = 0;
        for (int i = 0; i < n_parts; i++)
        {
            ctr = 0;
            start = Part->idx1[i];
            end = Part->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = Part->idx2[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        mat_parts[mat_ctr++] = col_part;
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        mat_parts[mat_ctr++] = col_part;
                    }
                    row_vals[col_part]++;
                }
            }

            mat.push_back(mat_ctr);
            for (int j = 0; j < mat_ctr; j++)
            {
                col = mat_parts[j];
                mat.push_back(col);
                mat.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
            mat_ctr = 0;
        }

        MPI_Send(mat.data(), mat.size(), MPI_INT, master_list[0], tag, MPI_COMM_WORLD);

        A2_parts.resize(n_parts);
        MPI_Recv(A2_parts.data(), n_parts, MPI_INT, master_list[0], tag, 
                MPI_COMM_WORLD, &recv_status);
    }
    delete OffPart;
    delete Part;

    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        A_parts[i] = A2_parts[local_part];
    }
*/

for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
    *it = rank;

    delete A_off_csc;
    return repartition_matrix(A_tmp, A_parts.data(), new_rows);

    
}




// Two-steps... partition on process and then on-node (on local process 0)
/*
ParCSRMatrix* NAP_partition(ParCSRMatrix* A_tmp, aligned_vector<int>& new_rows)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end, col, row, val;
    int ctr = 0;
    int tag = 3492;
    int count, size, idx;
    int part, first_part;
    int col_start, col_end;
    int row_start, row_end;
    int col_part, local_part;
    MPI_Status recv_status;

    CSCMatrix* A_off_csc = A_tmp->off_proc->to_CSC();

    CSRMatrix* A_combined = NULL;
    aligned_vector<int> A_parts;
    aligned_vector<int> P_parts;

    // Make serial matrix (data = sum of edges / off-proc neighbors)
    CSRMatrix* A_serial = new CSRMatrix(A_tmp->local_num_rows, A_tmp->on_proc_num_cols);
    aligned_vector<int> row_vals(A_tmp->local_num_rows, 0);
    A_serial->idx1[0] = 0;
    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        start = A_tmp->on_proc->idx1[i];
        end = A_tmp->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->on_proc->idx2[j];
            if (row_vals[col] == 0)
            {   
                A_serial->idx2.push_back(col);
            }
            row_vals[col]++;
        }
        start = A_tmp->off_proc->idx1[i];
        end = A_tmp->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_tmp->off_proc->idx2[j];
            col_start = A_off_csc->idx1[col];
            col_end = A_off_csc->idx1[col+1];
            for (int k = col_start; k < col_end; k++)
            {
                row = A_off_csc->idx2[k];
                if (row_vals[row] == 0) 
                {
                    A_serial->idx2.push_back(row);
                }
                row_vals[row]++;
            }
        }
        A_serial->idx1[i+1] = A_serial->idx2.size();

        start = A_serial->idx1[i];
        end = A_serial->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A_serial->idx2[j];
            A_serial->vals.push_back(row_vals[col]);
            row_vals[col] = 0;
        }
    }
    A_serial->nnz = A_serial->idx2.size();

    // Partition serial matrix
    int n_parts = (A_tmp->global_num_rows / num_procs) / 16;
    partition(A_serial, n_parts, A_parts);

    first_part = rank * n_parts;
    for (aligned_vector<int>::iterator it = A_parts.begin(); it != A_parts.end(); ++it)
        *it += first_part;

    // Send off_proc parts
    aligned_vector<int>& off_parts = A_tmp->comm->communicate(A_parts);

    // Form matrix of partitions
    aligned_vector<int> part_ptr(n_parts+1);
    aligned_vector<int> part_sizes(n_parts, 0);
    aligned_vector<int> part_idx;
    for (int i = 0; i < A_serial->n_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        part_sizes[local_part]++;
    }
    part_ptr[0] = 0;
    for (int i = 0; i < n_parts; i++)
    {
        part_ptr[i+1] = part_ptr[i] + part_sizes[i];
        part_sizes[i] = 0;
    }
    part_idx.resize(part_ptr[n_parts]);
    for (int i = 0; i < A_serial->n_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        idx = part_ptr[local_part] + part_sizes[local_part]++;
        part_idx[idx] = i;
    }

    // Gather matrix P on rank 0
    row_vals.resize(n_parts*16, 0);
    aligned_vector<int> mat;
    if (rank == 0)
    {
        A_combined = new CSRMatrix(n_parts*16, n_parts*16);
        A_combined->idx1[0] = 0;
        for (int i = 0; i < n_parts; i++)
        {
            start = part_ptr[i];
            end = part_ptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = part_idx[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        A_combined->idx2.push_back(col_part);
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        A_combined->idx2.push_back(col_part);
                    }
                    row_vals[col_part]++;
                }
            }
            A_combined->idx1[i+1] = A_combined->idx2.size();

            start = A_combined->idx1[i];
            end = A_combined->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_combined->idx2[j];
                A_combined->vals.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
        }

        int idx_ctr = n_parts+1;
        for (int i = 1; i < A_tmp->partition->topology->PPN; i++)
        {
            MPI_Probe(i, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > mat.size()) mat.resize(count);
            MPI_Recv(mat.data(), count, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_status);
            ctr = 0;
            while (ctr < count)
            {
                size = mat[ctr++];
                for (int j = 0; j < size; j++)
                {
                    A_combined->idx2.push_back(mat[ctr++]);
                    A_combined->vals.push_back(mat[ctr++]);
                }
                A_combined->idx1[idx_ctr++] = A_combined->idx2.size();
            }
        }
        A_combined->nnz = A_combined->idx1[A_combined->n_rows];
        A_combined->n_rows = n_parts*16;

        P_parts.resize(n_parts*16);
        partition(A_combined, A_tmp->partition->topology->PPN, P_parts);
        delete A_combined;

        start = n_parts;
        for (int i = 1; i < A_tmp->partition->topology->PPN; i++)
        {
            MPI_Send(&(P_parts[start]), n_parts, MPI_INT, i, tag, MPI_COMM_WORLD);
            start += n_parts;
        }

        P_parts.resize(n_parts);
    }
    else
    {
        aligned_vector<int> mat_parts(n_parts*16, 0);
        int mat_ctr = 0;
        ctr = 0;
        for (int i = 0; i < n_parts; i++)
        {
            start = part_ptr[i];
            end = part_ptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = part_idx[j];
                row_start = A_tmp->on_proc->idx1[row];
                row_end = A_tmp->on_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->on_proc->idx2[k];
                    col_part = A_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        mat_parts[mat_ctr++] = col_part;
                    }
                    row_vals[col_part]++;
                }
                row_start = A_tmp->off_proc->idx1[row];
                row_end = A_tmp->off_proc->idx1[row+1];
                for (int k = row_start; k < row_end; k++)
                {
                    col = A_tmp->off_proc->idx2[k];
                    col_part = off_parts[col];
                    if (row_vals[col_part] == 0)
                    {
                        mat_parts[mat_ctr++] = col_part;
                    }
                    row_vals[col_part]++;
                }
            }

            mat.push_back(mat_ctr);
            for (int j = 0; j < mat_ctr; j++)
            {
                col = mat_parts[j];
                mat.push_back(col);
                mat.push_back(row_vals[col]);
                row_vals[col] = 0;
            }
            mat_ctr = 0;
        }

        MPI_Send(mat.data(), mat.size(), MPI_INT, 0, tag, MPI_COMM_WORLD);

        P_parts.resize(n_parts);
        MPI_Recv(P_parts.data(), n_parts, MPI_INT, 0, tag, 
                MPI_COMM_WORLD, &recv_status);
    }

    for (int i = 0; i < A_tmp->local_num_rows; i++)
    {
        part = A_parts[i];
        local_part = part - first_part;
        A_parts[i] = P_parts[local_part];
    }

    delete A_serial;
    delete A_off_csc;
    return repartition_matrix(A_tmp, A_parts.data(), new_rows);
}
*/



// One-step... gather on-node and partition serially
/*
ParCSRMatrix* NAP_partition(ParCSRMatrix* A_tmp, aligned_vector<int>& new_rows)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end, col, val;
    int ctr = 0;
    int tag = 3492;
    int count, size;
    MPI_Status recv_status;

    CSRMatrix* A_combined = NULL;
    aligned_vector<int> parts;

    // Gather matrix A on rank 0
    aligned_vector<int> mat(A_tmp->local_num_rows + A_tmp->on_proc->nnz*2 + A_tmp->off_proc->nnz*2);
    if (rank == 0)
    {
        aligned_vector<int> proc_sizes(A_tmp->partition->topology->PPN);
        proc_sizes[0] = A_tmp->local_num_rows;

        A_combined = new CSRMatrix(A_tmp->global_num_rows, A_tmp->global_num_cols);
        A_combined->idx1[0] = 0;
        for (int i = 0; i < A_tmp->local_num_rows; i++)
        {
            start = A_tmp->on_proc->idx1[i];
            end = A_tmp->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_tmp->on_proc->idx2[j];
                A_combined->idx2.push_back(A_tmp->on_proc_column_map[col]);
                A_combined->vals.push_back(1);
            }
            start = A_tmp->off_proc->idx1[i];
            end = A_tmp->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_tmp->off_proc->idx2[j];
                A_combined->idx2.push_back(A_tmp->off_proc_column_map[col]);
                A_combined->vals.push_back(1);
            }
            A_combined->idx1[i+1] = A_combined->idx2.size();
        }
        int idx_ctr = A_tmp->local_num_rows + 1;
        for (int i = 1; i < A_tmp->partition->topology->PPN; i++)
        {
            MPI_Probe(i, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > mat.size()) mat.resize(count);
            MPI_Recv(mat.data(), count, MPI_INT, i, tag, MPI_COMM_WORLD, &recv_status);
            ctr = 0;
            proc_sizes[i] = 0;
            while (ctr < count)
            {
                size = mat[ctr++];
                for (int j = 0; j < size; j++)
                {
                    A_combined->idx2.push_back(mat[ctr++]);
                    A_combined->vals.push_back(mat[ctr++]);
                }
               // printf("Acomb idx1[%d] = %d\n", idx_ctr, A_combined->idx2.size());
                A_combined->idx1[idx_ctr++] = A_combined->idx2.size();
                proc_sizes[i]++;
            }
        }
        A_combined->nnz = A_combined->idx1[A_combined->n_rows];
        A_combined->n_rows = A_tmp->global_num_rows;
        printf("%d, %d, %d, %d\n", A_combined->n_rows, A_combined->idx1.size(),
                A_combined->nnz, A_combined->idx2.size());

        parts.resize(A_tmp->global_num_rows);
        partition(A_combined, A_tmp->partition->topology->PPN, parts);
        delete A_combined;

        start = proc_sizes[0];
        for (int i = 1; i < A_tmp->partition->topology->PPN; i++)
        {
            size = proc_sizes[i];
            MPI_Send(&(parts[start]), size, MPI_INT, i, tag, MPI_COMM_WORLD);
            start += size;
        }

        parts.resize(A_tmp->local_num_rows);
    }
    else
    {
        for (int i = 0; i < A_tmp->local_num_rows; i++)
        {
            mat[ctr++] = (A_tmp->on_proc->idx1[i+1] - A_tmp->on_proc->idx1[i])
                + (A_tmp->off_proc->idx1[i+1] - A_tmp->off_proc->idx1[i]);

            start = A_tmp->on_proc->idx1[i];
            end = A_tmp->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A_tmp->on_proc->idx2[j];
                val = 1;
                mat[ctr++] = A_tmp->on_proc_column_map[col];
                mat[ctr++] = val;
            }

            start = A_tmp->off_proc->idx1[i];
            end = A_tmp->off_proc->idx1[i+1];
            for (int j =start; j < end; j++)
            {
                col = A_tmp->off_proc->idx2[j];
                val = 1;
                mat[ctr++] = A_tmp->off_proc_column_map[col];
                mat[ctr++] = val;
            }
        }
        printf("Ctr %d, Matsize %d\n", ctr, mat.size());
        MPI_Send(mat.data(), ctr, MPI_INT, 0, tag, MPI_COMM_WORLD);

        parts.resize(A_tmp->local_num_rows);
        MPI_Recv(parts.data(), A_tmp->local_num_rows, MPI_INT, 0, tag, 
                MPI_COMM_WORLD, &recv_status);
    }


    return repartition_matrix(A_tmp, parts.data(), new_rows);

}
*/










