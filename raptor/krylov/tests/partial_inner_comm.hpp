#include "mpi.h"
    
using namespace raptor;

int create_partial_inner_comm(MPI_Comm &inner_comm, int &my_color, int &first_root, int &second_root, int contig)
{
    /*     inner_comm : MPI communicator containing the processes performing the partial inner product
     *       my_color : 0 : process is part of first half of vector
     *                  1 : process is part of second half of vector 
     *         contig : 1 : use contiguous processes in inner product
     *                  0 : use striped processes in inner product
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs <= 1) return x.local_n;

    int inner_comm_size, recv_comm_size, color, part_global;

    // Calculate number of processes in half inner product
    int half_procs = num_procs / 2;
 
    // If the partial inner product uses contiguous processes 
    if (contig) {
        if (num_procs % 2 != 0) half_procs++;

        if (rank < half_procs) color = 0;
        else color = 1;
        first_root = 0;
        second_root = half_procs;    
    
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);

        // First half
        /*if (half) {
            // If number of processes is odd, always add the spare process to the inner product
            if (num_procs % 2 != 0) half_procs++;

            if (rank < half_procs) color = 0;
            else color = 1;
            inner_root = 0;
            recv_root = half_procs;    
        
            if (!(color)){
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
                MPI_Comm_size(inner_comm, &inner_comm_size);
            }
            else{
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);
                MPI_Comm_size(recv_comm, &recv_comm_size);
            }
        }
        else {
            // Second half
            if (rank >= half_procs) color = 0;
            else color = 1;
            inner_root = half_procs;
            recv_root = 0;

            if (!(color)){
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
                MPI_Comm_size(inner_comm, &inner_comm_size);
            }
            else{
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);
                MPI_Comm_size(recv_comm, &recv_comm_size);
            }
        }*/
    }
    else {
        if (rank % 2 == 0) color = 0;
        else color = 1;
        first_root = 0;
        second_root =  1; 
        
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
    
        // Even half 
        /*if (half) {
            if (rank % 2 == 0) color = 0;
            else color = 1;
            inner_root = 0;
            recv_root =  1; 
            
            if (!(color)){
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
                MPI_Comm_size(inner_comm, &inner_comm_size);
            }
            else{
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);
                MPI_Comm_size(recv_comm, &recv_comm_size);
            }
        }
        else {
            // Odd half
            if (rank % 2 != 0) color = 0;
            else color = 1;

            // If number of processes is odd, always add the spare process to the inner product
            if ((num_procs % 2 == 1) && (rank == num_procs - 1)) color = 0;         

            inner_root = 1;
            recv_root = 0;
            
            if (!(color)){
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
                MPI_Comm_size(inner_comm, &inner_comm_size);
            }
            else{
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);
                MPI_Comm_size(recv_comm, &recv_comm_size);
            }
        }*/
    }

    // Get number of values being used in the inner product calculation
    part_global = x.local_n;
    MPI_All_Reduce(MPI_IN_PLACE, &part_global, 1, MPI_INT, MPI_SUM, inner_comm);

    return part_global;
}
