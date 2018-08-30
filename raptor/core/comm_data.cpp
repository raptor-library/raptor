// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_data.hpp"

namespace raptor 
{

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


}
