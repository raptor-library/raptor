// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;
/**************************************************************
*****   ParMatrix Add Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or 
***** off-diagonal block. 
*****
***** Parameters
***** -------------
***** row : index_t
*****    Local row of value
***** global_col : index_t 
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/    
void ParMatrix::add_value(
        int row, 
        index_t global_col, 
        data_t value)
{
    if (global_col >= first_local_col && global_col < first_local_col + local_num_cols)
    {
        diag->add_value(row, global_col - first_local_col, value);
    }
    else 
    {
        offd->add_value(row, global_col, value);
    }
}

/**************************************************************
*****   ParMatrix Add Global Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or 
***** off-diagonal block. 
*****
***** Parameters
***** -------------
***** global_row : index_t
*****    Global row of value
***** global_col : index_t 
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/ 
void ParMatrix::add_global_value(
        index_t global_row, 
        index_t global_col, 
        data_t value)
{
    if (global_col >= first_local_col && global_col < first_local_col + local_num_cols)
    {
        diag->add_value(global_row-first_local_row, global_col - first_local_col, value);
    }
    else
    {
        offd->add_value(global_row-first_local_row, global_col, value);
    }
}

/**************************************************************
*****   ParMatrix Finalize
**************************************************************
***** Finalizes the diagonal and off-diagonal matrices.  Sorts
***** the local_to_global indices, and creates the parallel
***** communicator
*****
***** Parameters
***** -------------
***** create_comm : bool (optional)
*****    Boolean for whether parallel communicator should be 
*****    created (default is true)
**************************************************************/
void ParMatrix::finalize(bool create_comm)
{
    offd->condense_cols();

    // Get reference to column mapping
    offd_column_map = offd->get_col_list();
    offd_num_cols = offd_column_map.size();   
    if (offd_num_cols)
    {
        Matrix* tmp = offd;
        offd = new CSCMatrix((COOMatrix*) tmp);
        offd->sort();
        delete tmp;
    }
    else
    {
        delete offd;
    }      

    Matrix* tmp = diag;
    diag = new CSRMatrix((COOMatrix*) tmp);
    diag->sort();
    delete tmp;

    local_nnz = diag->nnz;
    if (offd_num_cols)
        local_nnz += offd->nnz;

    if (create_comm)
        comm = new ParComm(offd_column_map, first_local_row, first_local_col);
    else
        comm = new ParComm();
}

