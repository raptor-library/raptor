// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_data.hpp"

namespace raptor 
{
template<>
std::vector<double>& CommData::get_buffer<double>(const int block_size)
{
    return buffer;
}
template<>
std::vector<int>& CommData::get_buffer<int>(const int block_size)
{
    return int_buffer;
}
template<> 
std::vector<char>& CommData::get_buffer<char>(const int block_size)
{
    return pack_buffer;
}

template<>
RAPtor_MPI_Datatype CommData::get_type<int>()
{
    return RAPtor_MPI_INT;
}
template<>
RAPtor_MPI_Datatype CommData::get_type<double>()
{
    return RAPtor_MPI_DOUBLE;
}

template<>
void CommData::send<int>(const int* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size, 
        std::function<int(int, int)> init_result_func, int init_result_func_val)
{
    int_send(values, key, mpi_comm, block_size, init_result_func, 
            init_result_func_val);
}
template<>
void CommData::send<double>(const double* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size, 
        std::function<double(double, double)> init_result_func, double init_result_func_val)
{
    double_send(values, key, mpi_comm, block_size, init_result_func,
            init_result_func_val);
}

template<>
void CommData::send<int>(const int* values, int key, RAPtor_MPI_Comm mpi_comm,
            const std::vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size)
{
    int_send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
}
template<>
void CommData::send<double>(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const std::vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size)
{
    double_send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
}


template <>
void CommData::recv<int>(int key, RAPtor_MPI_Comm mpi_comm, 
        const std::vector<int>& off_proc_states,
        std::function<bool(int)> compare_func,
        int* s_recv_ptr, int* n_recv_ptr, const int block_size)
{
    int_recv(key, mpi_comm, off_proc_states, compare_func,
            s_recv_ptr, n_recv_ptr, block_size);
}

template <>
void CommData::recv<double>(int key, RAPtor_MPI_Comm mpi_comm, 
        const std::vector<int>& off_proc_states,
        std::function<bool(int)> compare_func,
        int* s_recv_ptr, int* n_recv_ptr, const int block_size)
{
    double_recv(key, mpi_comm, off_proc_states, compare_func,
            s_recv_ptr, n_recv_ptr, block_size);
}



}
