#ifndef RAPTOR_UTILS_LINALG_MATMULT_H
#define RAPTOR_UTILS_LINALG_MATMULT_H

#include <mpi.h>
#include <float.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
//using Eigen::VectorXd;

#include "core/vector.hpp"
#include "core/par_matrix.hpp"
#include "core/matrix.hpp"
#include "core/par_vector.hpp"

// Structure of a COO Element
struct Element
{
    index_t row;
    index_t col;
    data_t value;
};

/**************************************************************
 *****   Create MPI Type
 **************************************************************
 ***** Creates a custom MPI_Datatype for communincating
 ***** COO elements (struct Element)
 *****
 ***** Parameters
 ***** -------------
 ***** coo_type : MPI_Datatype*
 *****    MPI_Datatype to be returned
 **************************************************************/
void create_mpi_type(MPI_Datatype* coo_type)
{
    index_t blocks[2] = {2, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Aint displacements[2];
    MPI_Aint intex;
    MPI_Type_extent(MPI_INT, &intex);
    displacements[0] = static_cast<MPI_Aint>(0);
    displacements[1] = 2*intex;
    MPI_Type_struct(2, blocks, displacements, types, coo_type);
}

/**************************************************************
 *****   Dot Product
 **************************************************************
 ***** Calculates the dot product of two sparse vectors
 ***** alpha = u^T*v
 *****
 ***** Parameters
 ***** -------------
 ***** size_u : index_t
 *****    Number of elements in vector u
 ***** size_v : index_t 
 *****    Number of elements in vector v
 ***** local_u : index_t* 
 *****    Indices of nonzeros in vector u
 ***** local_v : index_t*
 *****    Indices of nonzeros in vector v
 ***** data_u : data_t*
 *****    Values of nonzeros in vector u
 ***** data_v : data_t*
 *****    Values of nonzeros in vector v
 ***** map_u : UType
 *****    Maps indices of u from local to global
 ***** map_v : VType
 *****    Maps indices of v from local to global
 **************************************************************/
template <typename UType, typename VType>
data_t dot(index_t size_u, index_t size_v, index_t* local_u, 
    index_t* local_v, data_t* data_u, data_t* data_v,
    UType map_u, VType map_v);

/**************************************************************
 *****   Partial Sequential Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs a partial matmult, multiplying Matrix A
 ***** by a single column of B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on left)
 ***** B : Matrix*
 *****    Matrix to have single column multiplied (on right)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_B : BType 
 *****    Maps local columns of B to global columns
 ***** map_C : CType
 *****    Maps local resulting column to global
 ***** col : index_t 
 *****    Column of B to be multiplied
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType  map_A,
        BType map_B, CType map_C, index_t col);

/**************************************************************
 *****   Sequential Matrix-Matrix Multiplication
 **************************************************************
 ***** Performs matrix-matrix multiplication A*B
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled (on left)
 ***** B : Matrix*
 *****    Matrix to be multiplied (on right)
 ***** C : ParMatrix*
 *****    Parallel Matrix result is added to
 ***** map_A : AType
 *****    Maps local rows of A to global rows 
 ***** map_B : BType 
 *****    Maps local columns of B to global columns
 ***** map_C : CType
 *****    Maps local resulting column to global
 **************************************************************/
template <typename AType, typename BType, typename CType>
void seq_mm(Matrix* A, Matrix* B, ParMatrix* C, AType  map_A,
        BType map_B, CType map_C);


/**************************************************************
 *****   Parallel Matrix - Matrix Multiplication
 **************************************************************
 ***** Multiplies together two parallel matrices, outputing
 ***** the result in a new ParMatrix
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multiplied (on left)
 ***** B : ParMatrix* 
 *****    Parallel matrix to be multiplied (on right)
 ***** _C : ParMatrix**
 *****    Parallel Matrix result is inserted into
 **************************************************************/
void parallel_matmult(ParMatrix* A, ParMatrix* B, ParMatrix** _C)
{
    // If process not active, create new 
    // empty matrix, and return
    if (!(A->local_rows))
    {
        *_C = new ParMatrix();
        return;
    }

    // Declare matrix variables
    ParMatrix* C;
    Matrix* A_diag;
    Matrix* A_offd;
    Matrix* B_diag;
    Matrix* B_offd;

    // Declare format variables
    format_t format_ad;
    format_t format_ao;
    format_t format_bd;
    format_t format_bo;

    // Declare communication variables
    ParComm* comm;
    std::vector<index_t> send_procs;
    std::vector<index_t> recv_procs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;
    MPI_Request* send_requests;
    MPI_Status recv_status;
    std::vector<Element>* send_buffer;
    index_t send_proc;
    index_t num_sends;
    index_t num_recvs;
    index_t count;
    index_t n_recv;
    index_t avail_flag;

    // Declare matrix helper variables
    index_t row_start;
    index_t row_end;
    index_t col_start;
    index_t col_end;
    index_t local_row;
    index_t local_col;
    index_t global_row;
    index_t global_col;
    data_t value;
    Element tmp;
    index_t num_cols;

    // Declare temporary (recvd) matrix variables
    Matrix* Btmp;
    std::map<index_t, index_t> global_to_local;
    std::vector<index_t> local_to_global;
    std::vector<index_t> col_nnz;
    index_t tmp_size;

    // Initialize matrices
    C = new ParMatrix(A->global_rows, B->global_cols, A->comm_mat);
    A_diag = A->diag;
    A_offd = A->offd;
    B_diag = B->diag;
    B_offd = B->offd;

    // Set initial formats (to return with same type)
    format_ad = A_diag->format;
    format_bd = B_diag->format;
    if (A->offd_num_cols)
    {
        format_ao = A_offd->format;
    }
    if (B->offd_num_cols)
    {
        format_bo = B_offd->format;
    }

    // Initialize Communication Package
    comm = A->comm;
    send_procs = comm->send_procs;
    recv_procs = comm->recv_procs;
    send_indices = comm->send_indices;
    recv_indices = comm->recv_indices;
    send_requests = new MPI_Request[send_procs.size()];
    send_buffer = new std::vector<Element>[send_procs.size()];
    num_sends = send_procs.size();
    num_recvs = recv_procs.size();

    /* Send elements of B as array of COO structs */
    B_diag->convert(CSR);
    if (B->offd_num_cols)
    {
        B_offd->convert(CSR);
    }

    // Create custom datatype (COO elements)
    MPI_Datatype coo_type;
    create_mpi_type(&coo_type);

    // Commit COO Datatype
    MPI_Type_commit(&coo_type);

    // Send B-values to necessary processors
    for (index_t i = 0; i < num_sends; i++)
    {
        send_proc = send_procs[i];
        std::vector<index_t> send_idx = send_indices[send_proc];
        for (auto row : send_idx)
        {
            // Send diagonal portion of row
            row_start = B_diag->indptr[row];
            row_end = B_diag->indptr[row+1];
            for (index_t j = row_start; j < row_end; j++)
            {
                local_col = B_diag->indices[j] + B->first_col_diag;
                value = B_diag->data[j];
                tmp = {row + B->first_row, local_col, value};
                send_buffer[i].push_back(tmp);
            }

            // Send off-diagonal portion of row
            if (B->offd_num_cols)
            {
                row_start = B_offd->indptr[row];
                row_end = B_offd->indptr[row+1];
                for (index_t j = row_start; j < row_end; j++)
                {
                    local_col = B->local_to_global[B_offd->indices[j]];
                    value = B_offd->data[j];
                    tmp = {row + B->first_row, local_col, value};
                    send_buffer[i].push_back(tmp);
                }
            }
        }
        // Send B-values to distant processor
        MPI_Isend(send_buffer[i].data(), send_buffer[i].size(), coo_type,
            send_proc, 1111, A->comm_mat, &send_requests[i]);
    }

    // Convert A to CSR, B to CSC, and multiply local portion
    A_diag->convert(CSR);
    if (A->offd_num_cols)
    {
        A_offd->convert(CSR);
    }
    B_diag->convert(CSC);
   
    // Multiply A_diag * B_diag
    seq_mm<index_t, index_t, index_t>(A_diag, B_diag, C, 
        A->first_col_diag, B->first_row, B->first_col_diag);

    if (B->offd_num_cols)
    {
        B_offd->convert(CSC);
        
        // Multiply A_diag * B_offd
        seq_mm<index_t, index_t, std::vector<index_t>>(A_diag, B_offd, C, A->first_col_diag, B->first_row, B->local_to_global);
    }

    // Receive messages and multiply
    count = 0;
    n_recv = 0;
    while (n_recv < num_recvs)
    {
        //Probe for messages, and recv any found
        MPI_Iprobe(MPI_ANY_SOURCE, 1111, A->comm_mat, &avail_flag, &recv_status);
        if (avail_flag)
        {
            // Get size of message in buffer
            MPI_Get_count(&recv_status, coo_type, &count);

            // Get process message comes from
            index_t proc = recv_status.MPI_SOURCE;

            // Receive message into buffer
            Element recv_buffer[count];
            MPI_Recv(&recv_buffer, count, coo_type, MPI_ANY_SOURCE, 1111, A->comm_mat, &recv_status);

            // Initialize matrix for received values
            num_cols = 0;
            tmp_size = recv_indices[proc].size();
            Btmp = new Matrix(tmp_size, tmp_size);

            // Add values to Btmp
            for (index_t i = 0; i < count; i++)
            {
                global_row = recv_buffer[i].row;
                global_col = recv_buffer[i].col;
                value = recv_buffer[i].value;

                local_row = A->global_to_local[global_row];
                if (global_to_local.count(global_col) == 0)
                {
                    global_to_local[global_col] = col_nnz.size();
                    local_to_global.push_back(global_col);
                    col_nnz.push_back(1);
                    num_cols++;
                }
                else
                {
                    col_nnz[global_to_local[global_col]]++;
                }
                Btmp->add_value(local_row, global_to_local[global_col], value);
            }

            // Finalize Btmp
            Btmp->resize(tmp_size, num_cols);
            Btmp->finalize(CSC);

            //Multiply one col at a time
            for (index_t col = 0; col < num_cols; col++)
            {
                seq_mm<std::vector<index_t>, std::vector<index_t>, std::vector<index_t>>
                    (A->offd, Btmp, C, A->local_to_global, A->local_to_global,
                        local_to_global, col);
            }

            // Delete Btmp
            global_to_local.clear();
            local_to_global.clear();
            col_nnz.clear();
            delete Btmp;

            // Increment number of recvs
            n_recv++;
        }
    }

    // Free custom COO datatype (finished communication)
    MPI_Type_free(&coo_type);

    // Convert matrices back to original format
    A_diag->convert(format_ad);
    if (A->offd_num_cols)
    {
        A_offd->convert(format_ao);
    }
    B_diag->convert(format_bd);
    if (B->offd_num_cols)
    {
        B_offd->convert(format_bo);
    }

    // Finalize parallel output matrix
    C->finalize(0);
    *_C = C;

    // Wait for sends to finish, then delete buffer
    if (num_sends)
    {
	    MPI_Waitall(send_procs.size(), send_requests, MPI_STATUS_IGNORE);
        delete[] send_buffer;
        delete[] send_requests; 
    }
}   


#endif
