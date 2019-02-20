// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARTITION_HPP 
#define PARTITION_HPP

#include <mpi.h>
#include <math.h>
#include <set>

#include "types.hpp"
#include "topology.hpp"

#define STANDARD_PPN 4
#define STANDARD_PROC_LAYOUT 1

/**************************************************************
 *****   Partition Class
 **************************************************************
 ***** This class holds the partition of a number of vertices 
 ***** across a number of processes
 *****
 ***** Attributes
 ***** -------------
 ***** global_num_indices : index_t
 *****    Number of rows to be partitioned
 ***** first_local_idx : index_t
 *****    First global index of a row in partition local to rank
 ***** local_num_indices : index_t
 *****    Number of rows local to rank's partition
 *****
 ***** Methods
 ***** ---------
 **************************************************************/
namespace raptor
{
  class Partition
  {
  public:
    Partition(index_t _global_num_rows, index_t _global_num_cols,
            Topology* _topology = NULL)
    {
        int rank, num_procs;
        int avg_num;
        int extra;

        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;

        // Partition rows across processes
        avg_num = global_num_rows / num_procs;
        extra = global_num_rows % num_procs;
        first_local_row = avg_num * rank;
        local_num_rows = avg_num;
        if (extra > rank)
        {
            first_local_row += rank;
            local_num_rows++;
        }
        else
        {
            first_local_row += extra;
        }

        // Partition cols across processes
        if (global_num_rows < num_procs)
        {
            num_procs = global_num_rows;
        }
        avg_num = global_num_cols / num_procs;
        extra = global_num_cols % num_procs;
        if (local_num_rows)
        {
            first_local_col = avg_num * rank;
            local_num_cols = avg_num;
            if (extra > rank)
            {
                first_local_col += rank;
                local_num_cols++;
            }
            else
            {
                first_local_col += extra;
            }
        }
        else
        {
            local_num_cols = 0;
        }

        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(index_t _global_num_rows, index_t _global_num_cols,
            index_t _brows, index_t _bcols, Topology* _topology = NULL)
    {
        int rank, num_procs;
        int avg_num_blocks, global_num_row_blocks, global_num_col_blocks;
        int extra;

        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;

        // Partition rows across processes
        global_num_row_blocks = global_num_rows / _brows;
        avg_num_blocks = global_num_row_blocks / num_procs;
        extra = global_num_row_blocks % num_procs;
        first_local_row = avg_num_blocks * rank * _brows;
        local_num_rows = avg_num_blocks * _brows;
        if (extra > rank)
        {
            first_local_row += rank * _brows;
            local_num_rows += _brows;
        }
        else
        {
            first_local_row += extra * _brows;
        }

        // Partition cols across processes
            // local_num_cols = number of cols in on_proc matrix
        if (global_num_row_blocks < num_procs)
        {
            num_procs = global_num_row_blocks;
        }

            global_num_col_blocks = global_num_cols / _bcols;
        avg_num_blocks = global_num_col_blocks / num_procs;
        extra = global_num_col_blocks % num_procs;
        if (local_num_rows)
        {
            first_local_col = avg_num_blocks * rank * _bcols;
            local_num_cols = avg_num_blocks * _bcols;
            if (extra > rank)
            {
                first_local_col += rank * _bcols;
                local_num_cols += _bcols;
            }
            else
            {
                first_local_col += extra * _bcols;
            }
        }
        else
        {
            local_num_cols = 0;
        }

        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(index_t _global_num_rows, index_t _global_num_cols,
            int _local_num_rows, int _local_num_cols,
            index_t _first_local_row, index_t _first_local_col,
            Topology* _topology = NULL)
    {
        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;
        local_num_rows = _local_num_rows;
        local_num_cols = _local_num_cols;
        first_local_row = _first_local_row;
        first_local_col = _first_local_col;
        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(Topology* _topology = NULL)
    {
        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }

        num_shared = 0;
        global_num_rows = 0;
        global_num_cols = 0;
        local_num_rows = 0;
        local_num_cols = 0;
        first_local_row = 0;
        first_local_col = 0;
        last_local_row = 0;
        last_local_col = 0;
        assumed_num_cols = 0;
    }

    Partition(Partition* A, Partition* B)
    {
        global_num_rows = A->global_num_rows;
        global_num_cols = B->global_num_cols;
        local_num_rows = A->local_num_rows;
        local_num_cols = B->local_num_cols;
        first_local_row = A->first_local_row;
        first_local_col = B->first_local_col;
        last_local_row = A->last_local_row;
        last_local_col = B->last_local_col;

        num_shared = 0;

        assumed_num_cols = B->assumed_num_cols;
        first_cols.resize(B->first_cols.size());
        std::copy(B->first_cols.begin(), B->first_cols.end(),
                first_cols.begin());

        create_assumed_partition();

        topology = A->topology;
        topology->num_shared++;
    }

    Partition* transpose()
    {
        return new Partition(global_num_cols, global_num_rows,
                local_num_cols, local_num_rows, first_local_col,
                first_local_row, topology);
    }

    ~Partition()
    {
        if (topology->num_shared)
        {
            topology->num_shared--;
        }
        else
        {
            delete topology;
        }
    }

    void create_assumed_partition()
    {
        // Get RAPtor_MPI Information
        int rank, num_procs;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
        
        assumed_num_cols = global_num_cols / num_procs;
        if (global_num_cols % num_procs) assumed_num_cols++;

        first_cols.resize(num_procs+1);
        RAPtor_MPI_Allgather(&(first_local_col), 1, RAPtor_MPI_INT, first_cols.data(), 1, RAPtor_MPI_INT,
                        RAPtor_MPI_COMM_WORLD);
        first_cols[num_procs] = global_num_cols;
    }

    void form_col_to_proc (const aligned_vector<int>& off_proc_column_map,
            aligned_vector<int>& off_proc_col_to_proc) 
    {
        int rank, num_procs;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

        int global_col, assumed_proc;
        int ctr = 0;
        off_proc_col_to_proc.resize(off_proc_column_map.size());
        for (aligned_vector<int>::const_iterator it = off_proc_column_map.begin();
                        it != off_proc_column_map.end(); ++it)
        {
            global_col = *it;
            assumed_proc = global_col / assumed_num_cols;
            while (global_col < first_cols[assumed_proc])
            {
                assumed_proc--;
            }
            while (assumed_proc < num_procs - 1 && global_col >= first_cols[assumed_proc+1])
            {
                assumed_proc++;
            }
            off_proc_col_to_proc[ctr++] = assumed_proc;
        }
    }


    /**************************************************************
    *****   Split Off Proc Cols
    **************************************************************
    ***** Splits off_proc_column_map into on_node_column_map and 
    ***** off_node_column map.  Also maps each of these columns to 
    ***** their corresponding process, and maps each local index
    ***** in on_node and off_node to off_proc
    *****
    ***** Parameters
    ***** -------------
    ***** off_proc_column_map : aligned_vector<int>&
    *****    Vector holding rank's off_proc_columns
    ***** off_proc_col_to_proc : aligned_vector<int>&
    *****    Vector mapping rank's off_proc_columns to distant procs
    ***** on_node_column_map : aligned_vector<int>&
    *****    Will be returned holding on_node columns
    ***** on_node_col_to_proc : aligned_vector<int>&
    *****    Will be returned holding procs corresponding to on_node cols
    ***** on_node_to_off_proc : aligned_vector<int>&
    *****    Will be returned holding map from on_node to off_proc
    ***** off_node_column_map : aligned_vector<int>&
    *****    Will be returned holding off_node columns
    ***** off_node_col_to_node : aligned_vector<int>&
    *****    Will be returned holding procs corresponding to off_node cols
    ***** off_node_to_off_proc : aligned_vector<int>&
    *****    Will be returned holding map from off_node to off_proc
    **************************************************************/
    void split_off_proc_cols(const aligned_vector<int>& off_proc_column_map,
            const aligned_vector<int>& off_proc_col_to_proc,
            aligned_vector<int>& on_node_column_map,
            aligned_vector<int>& on_node_col_to_proc,
            aligned_vector<int>& on_node_to_off_proc,
            aligned_vector<int>& off_node_column_map,
            aligned_vector<int>& off_node_col_to_proc,
            aligned_vector<int>& off_node_to_off_proc)
    {
        int rank, rank_node, num_procs;
        int proc;
        int node;
        int global_col;
        int off_proc_num_cols = off_proc_column_map.size();

        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
        rank_node = topology->get_node(rank);

        // Reserve size in vectors

        on_node_column_map.reserve(off_proc_num_cols);
        on_node_col_to_proc.reserve(off_proc_num_cols);
        off_node_column_map.reserve(off_proc_num_cols);
        off_node_col_to_proc.reserve(off_proc_num_cols);
        
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            proc = off_proc_col_to_proc[i];
            node = topology->get_node(proc);
            global_col = off_proc_column_map[i];
            if (node == rank_node)
            {
                on_node_column_map.emplace_back(global_col);
                on_node_col_to_proc.emplace_back(topology->get_local_proc(proc));
                on_node_to_off_proc.emplace_back(i);
            }
            else
            {
                off_node_column_map.emplace_back(global_col);
                off_node_col_to_proc.emplace_back(proc);
                off_node_to_off_proc.emplace_back(i);
            }
        }
    }

    index_t global_num_rows;
    index_t global_num_cols;
    int local_num_rows;
    int local_num_cols;
    index_t first_local_row;
    index_t first_local_col;
    index_t last_local_row;
    index_t last_local_col;

    int assumed_num_cols;
    aligned_vector<int> first_cols;

    Topology* topology;

    int num_shared;  // Number of ParMatrix classes using partition

  };
}
#endif



