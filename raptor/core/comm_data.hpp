// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_COMMDATA_HPP
#define RAPTOR_CORE_COMMDATA_HPP

#define WITH_RAPtor_MPI 1

#include <mpi.h>
#include "mpi_types.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "utilities.hpp"

/**************************************************************
 *****   CommData Class
 **************************************************************
 **************************************************************/
namespace raptor
{
    // Forward Declaration
class CommData
{
public:
    /**************************************************************
    *****   CommData Class Constructor
    **************************************************************
    ***** Initializes an empty CommData, setting number and size of 
    ***** messages to zero.

    **************************************************************/
    CommData()
    {
        num_msgs = 0;
        size_msgs = 0;
        indptr.emplace_back(0);
    }

    CommData(CommData* data)
    {
        num_msgs = data->num_msgs;
        size_msgs = data->size_msgs;
        std::copy(data->procs.begin(), data->procs.end(),
                std::back_inserter(procs));
        std::copy(data->indptr.begin(), data->indptr.end(), 
                std::back_inserter(indptr));

        if (num_msgs)
        {
            requests.resize(num_msgs);
        }

        if (size_msgs)
        {
            buffer.resize(size_msgs);
            int_buffer.resize(size_msgs);
        }
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    virtual ~CommData()
    {
    };

    virtual void add_msg(int proc, int msg_size, int* msg_indices = NULL) = 0;

    void finalize()
    {
        if (num_msgs)
        {
            requests.resize(num_msgs);
        }
        if (size_msgs)
        {
                buffer.resize(size_msgs);
                int_buffer.resize(size_msgs);
        }
    }

    virtual void probe(int size, int key, RAPtor_MPI_Comm mpi_comm) = 0;

    virtual CommData* copy() = 0;
    virtual CommData* copy(const aligned_vector<int>& col_to_new) = 0;

    template <typename T>
    static RAPtor_MPI_Datatype get_type();


    template <typename T>
    aligned_vector<T>& get_buffer(const int block_size = 1, const int vblock_size = 1);

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const int vblock_size = 1, const int vblock_offset = 0,
            std::function<T(T, T)> init_result_func = &sum_func<T, T>,
            T init_result_func_val = 0);
    virtual void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val,
            const int vblock_size = 1,
            const int vblock_offset = 0) = 0;
    virtual void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val,
            const int vblock_offset = 0) = 0;

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size = 1, const int vblock_size = 1);
    virtual void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1) = 0;
    virtual void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1) = 0;


    virtual void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            const double* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1) = 0;
    virtual void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            double const* const* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1) = 0;
    virtual int get_msg_size(const int* rowptr, 
            const bool has_vals, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1) = 0;


    template <typename T>
    void recv(int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1, const int vblock_size = 1)
    {
        if (num_msgs == 0) return;

        int proc, start, end;
        int size = size_msgs * block_size * vblock_size;
        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);
       
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            RAPtor_MPI_Irecv(&(buf[start*block_size*vblock_size]), (end - start) * block_size * vblock_size, datatype,
                    proc, key, mpi_comm, &(requests[i]));
        }
    }   

    template <typename T>
    void recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1);
    virtual void int_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1) = 0;
    virtual void double_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1) = 0;

    void recv(CSRMatrix* recv_mat, int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const bool vals = true, const int vblock_size = 1)
    {
        if (num_msgs == 0) return;

        int proc, start, end, size;
        int ctr, row_size, row_count;
        int count, recv_size;
        RAPtor_MPI_Status recv_status;
        aligned_vector<char> recv_buffer;

        recv_size = 0;
        row_count = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            size = end - start;
            
            // Recv message of any size from proc
            RAPtor_MPI_Probe(proc, key, mpi_comm, &recv_status);
            RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_PACKED, &count);

            // Resize recv_buffer as needed
            if (count > recv_buffer.size())
            {
                recv_buffer.resize(count);
            }
            RAPtor_MPI_Recv(&(recv_buffer[0]), count, RAPtor_MPI_PACKED, proc, key,
                    mpi_comm, &recv_status);

            // Go through recv, adding indices to matrix recv_mat
            ctr = 0;
            for (int j = 0; j < size; j++)
            {
                RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &row_size, 1, RAPtor_MPI_INT,
                        mpi_comm);
                recv_mat->idx1[row_count + 1] = recv_size + row_size;
                row_count++;
                recv_mat->idx2.resize(recv_size + row_size);
                RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &recv_mat->idx2[recv_size],
                        row_size, RAPtor_MPI_INT, mpi_comm);
                
                if (vals)
                {
                    if (block_size > 1)
                    {
                        BSRMatrix* recv_mat_bsr = (BSRMatrix*) recv_mat;
                        recv_mat_bsr->block_vals.resize(recv_size + row_size);
                        for (int i = 0; i < row_size; i++)
                        {
                            recv_mat_bsr->block_vals[recv_size + i] = new double[block_size];
                            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, 
                                    recv_mat_bsr->block_vals[recv_size + i],
                                    block_size, RAPtor_MPI_DOUBLE, mpi_comm);
                        }
                    }
                    else
                    {
                        recv_mat->vals.resize(recv_size + row_size);
                        RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &recv_mat->vals[recv_size],
                                row_size, RAPtor_MPI_DOUBLE, mpi_comm);
                    }
                }
                recv_size += row_size;
            }
        }
        recv_mat->nnz = recv_mat->idx2.size();
    }

 
    void waitall()
    {
        if (num_msgs)
        {
            RAPtor_MPI_Waitall(num_msgs, requests.data(), RAPtor_MPI_STATUSES_IGNORE);
        }
    }
    void waitall(int n_msgs)
    {
        if (n_msgs)
        {
            RAPtor_MPI_Waitall(n_msgs, requests.data(), RAPtor_MPI_STATUSES_IGNORE);
        }
    }

    void pack_values(const double* values, int row_start, int size, char* send_buffer,
           int bytes, int* ctr, RAPtor_MPI_Comm mpi_comm, int block_size, int vblock_size = 1)
    {
        RAPtor_MPI_Pack(&(values[row_start]), size, RAPtor_MPI_DOUBLE, send_buffer, 
                bytes, ctr, mpi_comm);
    }
    void pack_values(double const* const* values, int row_start, int size, 
            char* send_buffer, int bytes, int* ctr, RAPtor_MPI_Comm mpi_comm, int block_size,
            int vblock_size = 1)
    {
        for (int i = 0; i < size; i++)
        {
            RAPtor_MPI_Pack(values[row_start + i], block_size, RAPtor_MPI_DOUBLE, send_buffer,
                    bytes, ctr, mpi_comm);
        }
    }

    template <typename T>
    void unpack(aligned_vector<T>& buffer, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const int vblock_size = 1)
    {
        if (num_msgs == 0) return;

        int position = 0;
        int flat_size = size_msgs * block_size;
        if (buffer.size() < flat_size) buffer.resize(flat_size);
        RAPtor_MPI_Datatype datatype = get_type<T>();
        RAPtor_MPI_Unpack(pack_buffer.data(), pack_buffer.size(), &position,
                buffer.data(), flat_size, datatype, mpi_comm);
    }

    void reset_buffer()
    {
        pack_buffer.resize(size_msgs);
    }

    int num_msgs;
    int size_msgs;
    aligned_vector<int> procs;
    aligned_vector<int> indptr;
    aligned_vector<RAPtor_MPI_Request> requests;
    aligned_vector<double> buffer;
    aligned_vector<int> int_buffer;
    aligned_vector<char> pack_buffer;

};

class ContigData : public CommData
{
public:
    ContigData() : CommData()
    {
    }

    ContigData(ContigData* data) : CommData(data)
    {

    }

    ~ContigData()
    {
    }

    ContigData* copy()
    {
        return new ContigData(this);
    }
    ContigData* copy(const aligned_vector<int>& col_to_new)
    {
        bool comm_proc;
        int proc, start, end;
        int new_idx;

        ContigData* data = new ContigData();

        data->size_msgs = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            comm_proc = false;
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                new_idx = col_to_new[j];
                if (new_idx != -1)
                {
                    comm_proc = true;
                    data->size_msgs++;
                }
            }
            if (comm_proc)
            {
                data->procs.emplace_back(proc);
                data->indptr.emplace_back(data->size_msgs);
            }
        }
        data->num_msgs = data->procs.size();
        data->finalize();

        return data;
    }

    void add_msg(int proc, int msg_size, int* msg_indices = NULL)
    {
        int last_ptr = indptr[num_msgs];
        procs.emplace_back(proc);
        indptr.emplace_back(last_ptr + msg_size);

        num_msgs++;
        size_msgs += msg_size;
    }

    void probe(int n_recv, int key, RAPtor_MPI_Comm mpi_comm)
    {
        int proc, count, size;
        RAPtor_MPI_Status recv_status;

        size_msgs = 0;
        indptr[0] = 0;
        for (int i = 0; i < n_recv; i++)
        {
            RAPtor_MPI_Recv(&size, 1, RAPtor_MPI_INT, RAPtor_MPI_ANY_SOURCE, key,
                    mpi_comm, &recv_status);
            procs.emplace_back(recv_status.RAPtor_MPI_SOURCE);
            size_msgs += size;
            indptr.emplace_back(size_msgs);
        }
        num_msgs = procs.size();
        finalize();
    }


    void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val,
            const int vblock_size = 1,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func, 
                init_result_func_val);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func,
                init_result_func_val);
    }

    void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size, vblock_size);
    }        

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const int vblock_size = 1, const int vblock_offset = 0,
            std::function<T(T, T)> init_result_func = &sum_func<T, T>,
            T init_result_func_val = 0)
    {
        if (num_msgs == 0) return;

        int start, end;
        int proc, idx;
        int size = size_msgs * block_size * vblock_size;
        aligned_vector<T> send_vals;

        RAPtor_MPI_Datatype datatype = get_type<T>();

        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];

            if (vblock_size > 1)
            {
                send_vals.resize((end-start) * block_size * vblock_size);
                for (int v = 0; v < vblock_size; v++)
                {
                    for (int i = 0; i < (end-start) * block_size; i++)
                    {
                        send_vals[(end-start)*block_size*v + i] = values[start*block_size + v*size_msgs + i];
                    }
                }

                RAPtor_MPI_Isend(&(send_vals[0]), (end-start)*block_size*vblock_size,
                            datatype, proc, key, mpi_comm, &(requests[i]));
            }
            else
            {
                RAPtor_MPI_Isend(&(values[start*block_size*vblock_size]), (end - start)*block_size*vblock_size,
                            datatype, proc, key, mpi_comm, &(requests[i]));
            }
        }
    }


    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        if (num_msgs == 0)
        {
            *n_send_ptr = 0;
            return;
        }

        int n_sends;
        int proc, start, end, idx;
        int ctr, prev_ctr;
        bool comparison;
        int size = size_msgs * block_size; 

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        n_sends = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                comparison = false;
                idx = j * block_size;
                for (int k = 0; k < block_size; k++)
                {
                    if (compare_func(states[idx + k]))
                    {
                        comparison = true;
                        break;
                    }
                }
                if (comparison)
                {
                    for (int k = 0; k < block_size; k++)
                    {
                        buf[ctr++] = values[idx+k];
                    }
                }
            }
            size = ctr - prev_ctr;
            if (size)
            {
                RAPtor_MPI_Isend(&(buf[prev_ctr]), size, datatype, 
                        proc, key, mpi_comm, &(requests[n_sends++]));
                prev_ctr = ctr;
            }
        }        
        *n_send_ptr = n_sends;
    }

    void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            const double* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        send_helper(send_buffer, rowptr, col_indices, values, key,
                mpi_comm, block_size);
    }

    void send(char* send_buffer,
        const int* rowptr,
        const int* col_indices,
        double const* const* values,
        int key, RAPtor_MPI_Comm mpi_comm,
        const int block_size = 1)     
    {
        send_helper(send_buffer, rowptr, col_indices, values, key,
                mpi_comm, block_size);
    }

    int get_msg_size(const int* rowptr, const bool has_vals, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        int start, end;
        int row_start, row_end;
        int num_ints, num_doubles;
        int double_bytes, bytes;

        // Calculate total msg size
        start = indptr[0];
        end = indptr[num_msgs];
        row_start = rowptr[start];
        row_end = rowptr[end];
        num_ints = (row_end - row_start) + (end - start);
        num_doubles = (row_end - row_start) * block_size;
        RAPtor_MPI_Pack_size(num_ints, RAPtor_MPI_INT, mpi_comm, &bytes);

        if (has_vals)
        {
            RAPtor_MPI_Pack_size(num_doubles, RAPtor_MPI_DOUBLE, mpi_comm, &double_bytes);
            bytes += double_bytes;
        }

        return bytes;
    }

    // values can be double* (CSRMatrix) or double** (BSRMatrix)
    template <typename T>
    void send_helper(char* send_buffer,
        const int* rowptr,
        const int* col_indices,
        const T& values,
        int key, RAPtor_MPI_Comm mpi_comm,
        const int block_size = 1,
        const int vblock_size = 1)
    {   
        if (num_msgs == 0) return;

        int start, end, proc;
        int ctr, prev_ctr, size;
        int row_start, row_end;
        int bytes;

        bytes = get_msg_size(rowptr, values, mpi_comm, block_size);

        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row_start = rowptr[j];
                row_end = rowptr[j+1];
                size = row_end - row_start;
                RAPtor_MPI_Pack(&size, 1, RAPtor_MPI_INT, send_buffer, bytes, 
                        &ctr, mpi_comm);
                RAPtor_MPI_Pack(&(col_indices[row_start]), size, RAPtor_MPI_INT,
                        send_buffer, bytes, &ctr, mpi_comm);
                if (values)
                {
                    pack_values(values, row_start, size, send_buffer, bytes, 
                            &ctr, mpi_comm, block_size);
                }
            }
            RAPtor_MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, RAPtor_MPI_PACKED, proc, 
                    key, mpi_comm, &(requests[i]));
            prev_ctr = ctr;
        }
    } 

    void int_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        cond_recv<int>(key, mpi_comm, off_proc_states, compare_func, s_recv_ptr,
                n_recv_ptr, block_size);
    }
    void double_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        cond_recv<double>(key, mpi_comm, off_proc_states, compare_func, s_recv_ptr,
                n_recv_ptr, block_size);
    }

    template <typename T>
    void cond_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
   {
        if (num_msgs == 0)  
        {
            *s_recv_ptr = 0;
            *n_recv_ptr = 0;
            return;
        }

        int n_recvs, ctr, prev_ctr;
        int proc, start, end, idx;
        int size = size_msgs * block_size;

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        n_recvs = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = j * block_size;
                for (int k = 0; k < block_size; k++)
                {
                    if (compare_func(off_proc_states[idx + k]))
                    {
                        ctr += block_size;
                        break;
                    }
                }
            }
            if (ctr - prev_ctr)
            {
                RAPtor_MPI_Irecv(&(buf[prev_ctr]), ctr - prev_ctr, datatype,
                        proc, key, mpi_comm, &(requests[n_recvs++]));
                prev_ctr = ctr;
            }
        }

        *n_recv_ptr = n_recvs;
        *s_recv_ptr = ctr;
    }

}; 

class NonContigData : public CommData
{
public:
    NonContigData() : CommData()
    {
    }

    NonContigData(NonContigData* data) : CommData(data)
    {
        std::copy(data->indices.begin(), data->indices.end(),
                std::back_inserter(indices));
    }

    ~NonContigData()
    {
    }

    NonContigData* copy()
    {
        return new NonContigData(this);
    }

    NonContigData* copy(const aligned_vector<int>& col_to_new)
    {
        bool comm_proc;
        int proc, start, end;
        int idx, new_idx;

        NonContigData* data = new NonContigData();

        data->size_msgs = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            comm_proc = false;
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = indices[j];
                new_idx = col_to_new[idx];
                if (new_idx != -1)
                {
                    comm_proc = true;
                    data->indices.emplace_back(new_idx);
                }
            }
            if (comm_proc)
            {
                data->procs.emplace_back(proc);
                data->indptr.emplace_back(data->indices.size());
            }
        }
        data->size_msgs = data->indices.size();
        data->num_msgs = data->procs.size();
        data->finalize();

        return data;
    }

    void add_msg(int proc,
            int msg_size,
            int* msg_indices = NULL)
    {
        int last_ptr = indptr[num_msgs];
        procs.emplace_back(proc);
        indptr.emplace_back(last_ptr + msg_size);
        if (msg_indices)
        {
            for (int i = 0; i < msg_size; i++)
            {
                indices.emplace_back(msg_indices[i]);
            }
        }

        num_msgs++;
        size_msgs += msg_size;
    }

    void probe(int size, int key, RAPtor_MPI_Comm mpi_comm)
    {
        int proc, count;
        int size_recvd;
        RAPtor_MPI_Status recv_status;

        size_msgs = size;
        indices.resize(size_msgs);
        indptr[0] = 0;
        size_recvd = 0;
        while (size_recvd < size_msgs)
        {
            RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, key, mpi_comm, &recv_status);
            proc = recv_status.RAPtor_MPI_SOURCE;
            RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
            RAPtor_MPI_Recv(&(indices[size_recvd]), count, RAPtor_MPI_INT, proc, 
                    key, mpi_comm, &recv_status);
            size_recvd += count;
            procs.emplace_back(proc);
            indptr.emplace_back(size_recvd);
        }
        num_msgs = procs.size();
        finalize();
    }

    void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val,
            const int vblock_size = 1,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func, 
                init_result_func_val);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func,
                init_result_func_val);
    }
    void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size, vblock_size);
    }     

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const int vblock_size = 1, const int vblock_offset = 0,
            std::function<T(T, T)> init_result_func = &sum_func<T, T>,
            T init_result_func_val = 0)
    {
	if (num_msgs == 0) return;

        int start, end, buf_pos;
        int proc, idx, pos;
        int size = size_msgs * block_size * vblock_size;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        //printf("%d buf size %d\n", rank, size);

        if (vblock_size > 1)
        {
            for (int i = 0; i < num_msgs; i++)
            {
                proc = procs[i];
                start = indptr[i];
                end = indptr[i+1];
                buf_pos = start * block_size * vblock_size;
                for (int j = start; j < end; j++)
                {
                    idx = indices[j] * block_size;
                    pos = buf_pos * block_size;
                    for (int v = 0; v < vblock_size; v++)
                    {
                        for (int k = 0; k < block_size; k++)
                        {
                            buf[pos + v*(end-start) + k] = values[idx + v*vblock_offset + k];
                            /*printf("%d buf[%d] %e = values[%d] %e to %d vblock_offset %d\n", rank, pos + v*(end-start) + k, 
                                    buf[pos + v*(end-start) + k], idx + v*vblock_offset + k, 
                                    values[idx + v*vblock_offset + k], proc, vblock_offset);*/
                        }
                    }
                    buf_pos++;
                }
                RAPtor_MPI_Isend(&(buf[start*block_size*vblock_size]), (end - start) * block_size * vblock_size,
                        datatype, proc, key, mpi_comm, &(requests[i]));
            }
        }
        else
        {
            for (int i = 0; i < num_msgs; i++)
            {
                proc = procs[i];
                start = indptr[i];
                end = indptr[i+1];
                //printf("%d start %d end %d\n", rank, start, end);
                for (int j = start; j < end; j++)
                {
                    idx = indices[j] * block_size;
                    pos = j * block_size;
                    for (int k = 0; k < block_size; k++)
                    {
                        buf[pos + k] = values[idx + k];
                    }
                }
                RAPtor_MPI_Isend(&(buf[start*block_size]), (end - start) * block_size,
                        datatype, proc, key, mpi_comm, &(requests[i]));
            }
            //printf("%d after sends placed in buf\n", rank);
        }
    }

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        if (num_msgs == 0)
        {
            *n_send_ptr = 0;
            return;
        }

        int n_sends;
        int proc, start, end;
        int idx;
        int ctr, prev_ctr;
        bool comparison;
        int size = size_msgs * block_size; 

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        n_sends = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = indices[j] * block_size;
                comparison = false;
                for (int k = 0; k < block_size; k++)
                {
                    // If compare true for any idx in block
                    // Add full block to message
                    if (compare_func(states[idx + k]))
                    {
                        comparison = true;
                        break;
                    }
                }
                if (comparison)
                {
                    for (int k = 0; k < block_size; k++)
                    {
                        buf[ctr++] = values[idx+k];
                    }
                }
            }
            if (ctr - prev_ctr)
            {
                RAPtor_MPI_Isend(&(buf[prev_ctr]), ctr - prev_ctr, datatype, 
                        proc, key, mpi_comm, &(requests[n_sends++]));
                prev_ctr = ctr;
            }
        }

        *n_send_ptr = n_sends;
    }

    void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            const double* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        send_helper(send_buffer, rowptr, col_indices, values, key,
                mpi_comm, block_size);
    }

    void send(char* send_buffer,
        const int* rowptr,
        const int* col_indices,
        double const* const* values,
        int key, RAPtor_MPI_Comm mpi_comm,
        const int block_size = 1)     
    {
        send_helper(send_buffer, rowptr, col_indices, values, key,
                mpi_comm, block_size);
    }

    int get_msg_size(const int* rowptr, const bool has_vals, RAPtor_MPI_Comm mpi_comm,
            const int block_size = 1)
    {
        int start, end;
        int row_start, row_end;
        int num_ints, num_doubles;
        int double_bytes, bytes;

        // Calculate message size
        num_ints = indptr[num_msgs] - indptr[0];
        num_doubles = 0;
        for (aligned_vector<int>::iterator it = indices.begin();
                it != indices.end(); ++it)
        {
            num_doubles += (rowptr[*it+1] - rowptr[*it]);
        }
        num_ints += num_doubles;
        RAPtor_MPI_Pack_size(num_ints, RAPtor_MPI_INT, mpi_comm, &bytes);

        if (has_vals)
        {
            RAPtor_MPI_Pack_size(num_doubles * block_size, RAPtor_MPI_DOUBLE, mpi_comm, &double_bytes);
            bytes += double_bytes;
        }

        return bytes;
    }

    template <typename T>
    void send_helper(char* send_buffer,
        const int* rowptr,
        const int* col_indices,
        const T& values,
        int key, RAPtor_MPI_Comm mpi_comm,
        const int block_size = 1,
        const int vblock_size = 1)     
    {
        if (num_msgs == 0) return;

        int start, end, proc;
        int ctr, prev_ctr, size;
        int row, row_start, row_end;
        int num_ints, num_doubles;
        int double_bytes, bytes;

        // Resize send buffer
        bytes = get_msg_size(rowptr, values, mpi_comm, block_size);

        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = indices[j];
                row_start = rowptr[row];
                row_end = rowptr[row+1];
                size = (row_end - row_start);
                RAPtor_MPI_Pack(&size, 1, RAPtor_MPI_INT, send_buffer, bytes, 
                        &ctr, mpi_comm);
                RAPtor_MPI_Pack(&(col_indices[row_start]), size, RAPtor_MPI_INT, 
                        send_buffer, bytes, &ctr, mpi_comm);
                if (values)
                {                    
                    pack_values(values, row_start, size, send_buffer, bytes, &ctr, 
                            mpi_comm, block_size);
                }
            }
            RAPtor_MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, RAPtor_MPI_PACKED, proc, 
                    key, mpi_comm, &(requests[i]));
            prev_ctr = ctr;
        }
    }

 
    void int_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        cond_recv<int>(key, mpi_comm, off_proc_states, compare_func, s_recv_ptr,
                n_recv_ptr, block_size);
    }
    void double_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
    {
        cond_recv<double>(key, mpi_comm, off_proc_states, compare_func, s_recv_ptr,
                n_recv_ptr, block_size);
    }

    template <typename T>
    void cond_recv(int key, RAPtor_MPI_Comm mpi_comm, 
            const aligned_vector<int>& off_proc_states,
            std::function<bool(int)> compare_func,
            int* s_recv_ptr, int* n_recv_ptr, const int block_size = 1, const int vblock_size = 1)
   {
        if (num_msgs == 0)
        {
            *s_recv_ptr = 0;
            *n_recv_ptr = 0;
            return;
        }

        int n_recvs, ctr, prev_ctr;
        int proc, start, end, idx;
        int size = size_msgs * block_size;

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        n_recvs = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = indices[j] * block_size;
                for (int k = 0; k < block_size; k++)
                {
                    if (compare_func(off_proc_states[idx + k]))
                    {
                        ctr += block_size;
                        break;
                    }
                }
            }
            if (ctr - prev_ctr)
            {
                RAPtor_MPI_Irecv(&(buf[prev_ctr]), ctr - prev_ctr, datatype, proc,
                        key, mpi_comm, &(requests[n_recvs++]));
                prev_ctr = ctr;
            }
        }

        *n_recv_ptr = n_recvs;
        *s_recv_ptr = ctr;    
   }

    aligned_vector<int> indices;

}; 

class DuplicateData : public NonContigData
{
public:
    DuplicateData() : NonContigData()
    {
    }

    DuplicateData(DuplicateData* data) : NonContigData(data)
    {
        std::copy(data->indptr_T.begin(), data->indptr_T.end(),
                std::back_inserter(indptr_T));
    }

    ~DuplicateData()
    {
    }

    DuplicateData* copy()
    {
        return new DuplicateData(this);
    }
    DuplicateData* copy(const aligned_vector<int>& col_to_new)
    {
        bool comm_proc, comm_idx;
        int proc, start, end;
        int idx, new_idx;
        int idx_start, idx_end;

        DuplicateData* data = new DuplicateData();

        data->indptr_T.emplace_back(0);
        for (int i = 0; i < num_msgs; i++)
        {
            comm_proc = false;
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                comm_idx = false;
                idx_start = indptr_T[j];
                idx_end = indptr_T[j+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx = indices[k];
                    new_idx = col_to_new[idx];
                    if (new_idx != -1)
                    {
                        comm_idx = true;
                        data->indices.emplace_back(new_idx);
                    }
                }
                if (comm_idx)
                {
                    comm_proc = true;
                    data->indptr_T.emplace_back(data->indices.size());
                }
            }
            if (comm_proc)
            {
                data->procs.emplace_back(proc);
                data->indptr.emplace_back(data->indptr_T.size() - 1);
            }
        }
        data->size_msgs = data->indptr_T.size() - 1;
        data->num_msgs = data->procs.size();
        data->finalize();

        return data;
    }

    void int_send(const int* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            std::function<int(int, int)> init_result_func,
            int init_result_func_val,
            const int vblock_size = 1,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func, 
                init_result_func_val);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size,
            const int vblock_size,
            std::function<double(double, double)> init_result_func,
            double init_result_func_val,
            const int vblock_offset = 0)
    {
        send(values, key, mpi_comm, block_size, vblock_size, vblock_offset, init_result_func,
                init_result_func_val);
    }
    void int_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size = 1)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size);
    }
    void double_send(const double* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size, const int vblock_size)
    {
        send(values, key, mpi_comm, states, compare_func, n_send_ptr, block_size, vblock_size);
    }     

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm, const int block_size = 1,
            const int vblock_size = 1, const int vblock_offset = 0,
            std::function<T(T, T)> init_result_func = &sum_func<T, T>,
            T init_result_func_val = 0)
    {
        if (num_msgs == 0) return;

        printf("duplicate data non_contig send being called?\n");

        int start, end;
        int proc, idx;
        int idx_start, idx_end;
        int size = size_msgs * block_size;
        int pos;

        RAPtor_MPI_Datatype datatype = get_type<T>();
        aligned_vector<T>& buf = get_buffer<T>();
        if (buf.size() < size) buf.resize(size);

        aligned_vector<T> tmp(block_size);

        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx_start = indptr_T[j];
                idx_end = indptr_T[j+1];
                std::fill(tmp.begin(), tmp.end(), init_result_func_val);
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx = indices[k] * block_size;
                    for (int l = 0; l < block_size; l++)
                    {
                        tmp[l] = init_result_func(tmp[l], values[idx+l]);
                    }
                }
                pos  = j * block_size;
                for (int k = 0; k < block_size; k++)
                {
                    buf[pos + k] = tmp[k];
                }
            }
            RAPtor_MPI_Isend(&(buf[start * block_size]), (end - start) * block_size,
                   datatype, proc, key, mpi_comm, &(requests[i]));
        }
    }

    template <typename T>
    void send(const T* values, int key, RAPtor_MPI_Comm mpi_comm,
            const aligned_vector<int>& states, std::function<bool(int)> compare_func,
            int* n_send_ptr, const int block_size = 1, const int vblock_size = 1)
    {

    }

    void append_val(aligned_vector<double>& vec, const double val, int block_size)
    {
        vec.emplace_back(val);
    }
    void append_val(aligned_vector<double>& vec, const double* val, int block_size)
    {
        for (int i = 0; i < block_size; i++)
            vec.emplace_back(val[i]);
    }
    
    template <typename T>
    void combine_entries(int j, const int* rowptr, const int* col_indices, 
            const T& values, int block_size, aligned_vector<int>& send_indices, 
            aligned_vector<double>& send_values, int* size_ptr)
    {
        int idx_start, idx_end;
        int row_start, row_end;
        int size, row, idx, ctr;
        int pos = 0;

        idx_start = indptr_T[j];
        idx_end = indptr_T[j+1];
        for (int k = idx_start; k < idx_end; k++)
        {
            row = indices[k];
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            for (int l = row_start; l < row_end; l++)
            {
                send_indices.emplace_back(col_indices[l]);
                append_val(send_values, values[l], block_size);
            }
        }
        if (send_indices.size())
        {
            vec_sort(send_indices, send_values);
            size = 1;

            for (int k = 1; k < send_indices.size(); k++)
            {
                ctr = k * block_size;
                if (send_indices[k] != send_indices[size - 1])
                {
                    idx = size * block_size;
                    for (int i = 0; i < block_size; i++)
                    {
                        send_values[idx + i] = send_values[ctr + i];
                    }
                    send_indices[size++] = send_indices[k];
                }
                else
                {
                    idx = (size - 1) * block_size;
                    for (int i = 0; i < block_size; i++)
                    {
                        send_values[idx + i] += send_values[ctr + i];
                    }
                }
            } 
        }
        else size = 0;

        *size_ptr = size;
    }
    
    void combine_entries(int j, const int* rowptr, const int* col_indices, 
            aligned_vector<int>& send_indices, int* size_ptr)
    {
        int idx_start, idx_end;
        int row_start, row_end;
        int size, row;

        idx_start = indptr_T[j];
        idx_end = indptr_T[j+1];
        for (int k = idx_start; k < idx_end; k++)
        {
            row = indices[k];
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            for (int l = row_start; l < row_end; l++)
            {
                send_indices.emplace_back(col_indices[l]);
            }
        }
        if (send_indices.size())
        {
            size = 1;
            std::sort(send_indices.begin(), send_indices.end());
            for (int k = 1; k < send_indices.size(); k++)
            {
                if (send_indices[k] != send_indices[size - 1])
                {
                    send_indices[size++] = send_indices[k];
                }
            }
        }
        else size = 0;

        *size_ptr = size;
    }


    // TODO -- how to communicate block matrices?
    //
    void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            const double* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        send_helper(send_buffer, rowptr, col_indices, values, key, mpi_comm, block_size);
    }
    void send(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            double const* const* values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        send_helper(send_buffer, rowptr, col_indices, values, key, mpi_comm, block_size);
    }

    int get_msg_size(const int* rowptr, const bool has_vals, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        int start, end;
        int row_start, row_end;
        int num_ints, num_doubles;
        int double_bytes, bytes;

        // Calculate message size (upper bound)
        num_ints = indptr[num_msgs] - indptr[0];
        num_doubles = 0;
        for (aligned_vector<int>::iterator it = indices.begin();
                it != indices.end(); ++it)
        {
            num_doubles += (rowptr[*it+1] - rowptr[*it]);
        }
        num_ints += num_doubles;
        RAPtor_MPI_Pack_size(num_ints, RAPtor_MPI_INT, mpi_comm, &bytes);
        if (has_vals)
        {
            RAPtor_MPI_Pack_size(num_doubles * block_size, RAPtor_MPI_DOUBLE, mpi_comm, &double_bytes);
            bytes += double_bytes;
        }

        return bytes;
    }

    template <typename T>
    void send_helper(char* send_buffer,
            const int* rowptr, 
            const int* col_indices,
            const T& values, 
            int key, RAPtor_MPI_Comm mpi_comm, 
            const int block_size = 1)
    {
        if (num_msgs == 0) return;

        int start, end, proc;
        int ctr, prev_ctr, size;
        int row, row_start, row_end;
        int idx_start, idx_end;
        int bytes;

        // Resize send buffer
        bytes = get_msg_size(rowptr, values, mpi_comm, block_size);

        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < num_msgs; i++)
        {
            proc = procs[i];
            start = indptr[i];
            end = indptr[i+1];
            for (int j = start; j < end; j++)
            {
                aligned_vector<int> send_indices;
                aligned_vector<double> send_values;
                
                if (values)
                {
                    combine_entries(j, rowptr, col_indices, values, block_size,
                            send_indices, send_values, &size);
                }
                else
                {
                    combine_entries(j, rowptr, col_indices, send_indices, &size);
                }
                RAPtor_MPI_Pack(&size, 1, RAPtor_MPI_INT, send_buffer, bytes, &ctr, mpi_comm);
                RAPtor_MPI_Pack(send_indices.data(), size, RAPtor_MPI_INT, send_buffer,
                    bytes, &ctr, mpi_comm);

                if (values)
                {
                    pack_values(send_values.data(), row_start, size, send_buffer, bytes, &ctr, 
                            mpi_comm, block_size);
                }
            }
            RAPtor_MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, RAPtor_MPI_PACKED, proc, 
                    key, mpi_comm, &(requests[i]));
            prev_ctr = ctr;
        }
    }

     aligned_vector<int> indptr_T;

}; 

}
#endif

