// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_data.hpp"

namespace raptor 
{

/*template<>
void CommData::get_buffer<double>(aligned_vector<double>& buffer, 
        MPI_Comm mpi_comm, const int block_size)
{
    int position = 0;
    int size = size_msgs * block_size;
    if (size > buffer.size()) buffer.resize(size);
    MPI_Unpack(pack_buffer.data(), pack_buffer.size(), &position,
            buffer, size, MPI_DOUBLE, mpi_comm);
    return buffer;
}

template<>
void CommData::get_buffer<int>(aligned_vector<int>& buffer,
        MPI_Comm mpi_comm, const int block_size)
{
    int position = 0;
    int size = size_msgs * block_size;
    if (size > buffer.size()) buffer.resize(size);
    MPI_Unpack(pack_buffer.data(), pack_buffer.size(), &position,
            buffer, size, MPI_INT, mpi_comm);
    return buffer;
}

*/
template<>
aligned_vector<double>& CommData::get_buffer<double>()
{
    return buffer;
}
template<>
aligned_vector<int>& CommData::get_buffer<int>()
{
    return int_buffer;
}

template<>
MPI_Datatype CommData::get_type<int>()
{
    return MPI_INT;
}
template<>
MPI_Datatype CommData::get_type<double>()
{
    return MPI_DOUBLE;
}

template<>
void CommData::send<int>(const int* values, int key, MPI_Comm mpi_comm, const int block_size, 
        std::function<int(int, int)> init_result_func, int init_result_func_val)
{
    int_send(values, key, mpi_comm, block_size, init_result_func, 
            init_result_func_val);
}
template<>
void CommData::send<double>(const double* values, int key, MPI_Comm mpi_comm, const int block_size, 
        std::function<double(double, double)> init_result_func, double init_result_func_val)
{
    double_send(values, key, mpi_comm, block_size, init_result_func,
            init_result_func_val);
}

template<>
void CommData::send<int>(const int* values, int key, MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size)
{
    int_send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
}
template<>
void CommData::send<double>(const double* values, int key, MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size)
{
    double_send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
}


template <>
void CommData::recv<int>(int key, MPI_Comm mpi_comm, 
        const aligned_vector<int>& off_proc_states,
        std::function<bool(int)> compare_func,
        int* s_recv_ptr, int* n_recv_ptr, const int block_size)
{
    int_recv(key, mpi_comm, off_proc_states, compare_func,
            s_recv_ptr, n_recv_ptr, block_size);
}

template <>
void CommData::recv<double>(int key, MPI_Comm mpi_comm, 
        const aligned_vector<int>& off_proc_states,
        std::function<bool(int)> compare_func,
        int* s_recv_ptr, int* n_recv_ptr, const int block_size)
{
    double_recv(key, mpi_comm, off_proc_states, compare_func,
            s_recv_ptr, n_recv_ptr, block_size);
}



}
