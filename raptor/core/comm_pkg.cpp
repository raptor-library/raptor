// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"
#include "core/par_matrix.hpp"

namespace raptor
{
    template<>
    aligned_vector<double>& CommPkg::get_buffer<double>()
    {
        return get_double_buffer();
    }
    template<>
    aligned_vector<int>& CommPkg::get_buffer<int>()
    {
        return get_int_buffer();
    }

    template<>
    aligned_vector<double>& CommPkg::communicate<double>(const double* values,
            const int block_size, const int vblock_size, const int vblock_offset)
    {
        // THIS COMMUNICATE CALLED IN TAP MULT
        init_double_comm(values, block_size, vblock_size, vblock_offset);
        return complete_double_comm(block_size, vblock_size);
    }
    template<>
    aligned_vector<int>& CommPkg::communicate<int>(const int* values,
            const int block_size, const int vblock_size, const int vblock_offset)
    {
        init_int_comm(values, block_size);
        return complete_int_comm(block_size);
    }

    template<>
    void CommPkg::init_comm<double>(const double* values,
            const int block_size, const int vblock_size)
    {
        init_double_comm(values, block_size, vblock_size);
    }
    template<>
    void CommPkg::init_comm<int>(const int* values, const int block_size, const int vblock_size)
    {
        init_int_comm(values, block_size);
    }

    template<>
    aligned_vector<double>& CommPkg::complete_comm<double>(const int block_size, const int vblock_size)
    {
        // THIS COMPLETE COMM USED IN MULT
        // THIS COMPLETE COMM USED IN TAP MULT
        return complete_double_comm(block_size, vblock_size);
    }
    template<>
    aligned_vector<int>& CommPkg::complete_comm<int>(const int block_size, const int vblock_size)
    {
        return complete_int_comm(block_size);
    }

    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<double>& result, 
            const int block_size,
            const int vblock_size,
            std::function<double(double, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_double_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const double* values,
            aligned_vector<int>& result, 
            const int block_size,
            const int vblock_size,
            std::function<int(int, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_double_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<int>& result, 
            const int block_size,
            const int vblock_size,
            std::function<int(int, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_int_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T(const int* values,
            aligned_vector<double>& result, 
            const int block_size,
            const int vblock_size,
            std::function<double(double, int)> result_func,
            std::function<int(int, int)> init_result_func, 
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_int_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T<double>(const double* values,
            const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_double_comm_T(block_size, vblock_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::communicate_T<int>(const int* values,
            const int block_size,
            const int vblock_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
        complete_int_comm_T(block_size, vblock_size, init_result_func, init_result_func_val);
    }

    template<>
    void CommPkg::init_comm_T<double>(const double* values,
            const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        init_double_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::init_comm_T<int>(const int* values,
            const int block_size,
            const int vblock_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        init_int_comm_T(values, block_size, vblock_size, init_result_func, init_result_func_val);
    }

    template<>
    void CommPkg::complete_comm_T<double, double>(aligned_vector<double>& result,
            const int block_size,
            const int vblock_size,
            std::function<double(double, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<double, int>(aligned_vector<int>& result,
            const int block_size,
            const int vblock_size,
            std::function<int(int, double)> result_func,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int, int>(aligned_vector<int>& result,
            const int block_size,
            const int vblock_size,
            std::function<int(int, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int, double>(aligned_vector<double>& result,
            const int block_size,
            const int vblock_size,
            std::function<double(double, int)> result_func,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(result, block_size, vblock_size, result_func, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<double>(const int block_size, const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val)
    {
        complete_double_comm_T(block_size, vblock_size, init_result_func, init_result_func_val);
    }
    template<>
    void CommPkg::complete_comm_T<int>(const int block_size,
            const int vblock_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val)
    {
        complete_int_comm_T(block_size, vblock_size, init_result_func, init_result_func_val);
    }
}


using namespace raptor;

aligned_vector<double>& CommPkg::communicate(ParVector& v, const int block_size)
{
    init_double_comm(v.local->data(), block_size);
    return complete_double_comm(block_size, v.local->b_vecs);
}

void CommPkg::init_comm(ParVector& v, const int block_size, const int vblock_size)
{
    // THIS IS THE INIT COMM BEING CALLED BY MULT    
    // THIS IS THE INIT COMM BEING CALLED BY TAP MULT
    init_double_comm(v.local->data(), block_size, vblock_size, v.local->num_values);
}

