// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "matrix.hpp"

/**************************************************************
 *****   Matrix Class Constructor
 **************************************************************
 ***** TODO -- copy Matrix?
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to copy
 **************************************************************/
Matrix::Matrix(Matrix* A)
{

}

/**************************************************************
 *****   Matrix Class Constructor
 **************************************************************
 ***** Initializes an empty COO Matrix (without setting dimensions)
 *****
 **************************************************************/
Matrix::Matrix()
{
    nnz = 0;
    format = COO;
}

/**************************************************************
 *****   Matrix Class Destructor
 **************************************************************
 ***** Deletes all arrays/vectors
 *****
 **************************************************************/
Matrix::~Matrix()
{

}

/**************************************************************
 *****   Matrix Reserve
 **************************************************************
 ***** Reserves nnz per each outer (row or column) to add entries 
 ***** without re-allocating memory.
 ***** TODO -- implement for COO, add CSR/CSC functionality
 *****
 ***** Parameters
 ***** -------------
 ***** nnz_per_outer : index_t
 *****    Number of nonzeros per each row (or column if CSC)
 **************************************************************/
void Matrix::reserve(index_t nnz_per_outer)
{
    index_t max_nnz = nnz_per_outer * n_outer;
       
    this->indptr.resize(n_outer + 1);
    this->indices.resize(max_nnz);
    this->data.resize(max_nnz);

    for (index_t i = 0; i < n_outer; i++)
    {
        this->indptr[i] = i*nnz_per_outer;
    }

    this->indptr[n_outer] = n_outer*nnz_per_outer;
}

/**************************************************************
 *****   Matrix Add Value
 **************************************************************
 ***** Inserts a value into the Matrix
 ***** TODO -- functionality for CSR/CSC
 *****
 ***** Parameters
 ***** -------------
 ***** ptr : index_t
 *****    Outer index (Row in CSR, Col in CSC)
 ***** idx : index_t 
 *****    Inner index (Col in CSR, Row in CSC)
 ***** value : data_t
 *****    Value to insert
 **************************************************************/
void Matrix::add_value(index_t row, index_t col, data_t value)
{
    if (format == CSR)
    {
        // TODO -- Add directly to CSR
    }
    else if (format == CSC)
    {
        // TODO -- Add directly to CSC
    }
    else if (format == COO)
    {
        this->row_idx.push_back(row);
        this->col_idx.push_back(col);
        this->data.push_back(value);

        this->nnz++;
    }
}
 
/**************************************************************
 *****   Matrix Resize
 **************************************************************
 ***** Resizes the dimensions of the matrix
 *****
 ***** Parameters
 ***** -------------
 ***** _nrows : index_t
 *****    Number of rows in the matrix
 ***** _ncols : index_t 
 *****    Number of columns in the matrix
 **************************************************************/
void Matrix::resize(index_t _nrows, index_t _ncols)
{
    this->n_rows = _nrows;
    this->n_cols = _ncols;
   
    if (format == CSR)
    {
        this->n_outer = _nrows;
        this->n_inner = _ncols;
        this->indptr.resize(_nrows + 1);
    }
    else if (format == CSC)
    {
        this->indptr.resize(_ncols + 1);
        this->n_outer = _ncols;
        this->n_inner = _nrows;
    }
}

/**************************************************************
 *****   Matrix Finalize
 **************************************************************
 ***** Compresses matrix, sorts the entries, removes any zero
 ***** values, and combines any entries at the same location
 *****
 ***** Parameters
 ***** -------------
 ***** _format : format_t
 *****    Format to convert Matrix to
 **************************************************************/
void Matrix::finalize(format_t _format)
{
    if (this->format == _format)
    {
        if (format == COO)
        {
            // Nothing to do?
        }
        else
        {
            //TODO-- Add support for straight to CSR/CSC
        }
    }

    else convert(_format);
}

/**************************************************************
 *****   Matrix Convert
 **************************************************************
 ***** Converts matrix into given format.  If format is the same
 ***** as that already set, nothing is done.
 *****
 ***** Parameters
 ***** -------------
 ***** _format : format_t
 *****    Format to convert Matrix to
 **************************************************************/
void Matrix::convert(format_t _format)
{
    // Don't convert if format remains the same
    if (format == _format)
    {
        return;
    }

    // Convert between CSR and CSC
    if ((this->format == CSR || this->format == CSC) 
        && (_format == CSR || _format == CSC))
    {
        // Switch inner and outer indices
        index_t n_tmp = this->n_inner;
        this->n_inner = this->n_outer;
        this->n_outer = n_tmp;

        if (nnz == 0)
        {
            indptr.resize(n_outer+1);
            for (index_t i = 0; i < n_outer + 1; i++)
            {
                indptr[i] = 0;
            }
            return;
        }


        // Calculate number of nonzeros per outer idx
        index_t* ptr_nnz = new index_t[this->n_outer]();
        for (index_t i = 0; i < this->nnz; i++)
        {
            ptr_nnz[this->indices[i]]++;      
        }

        // Create vectors to copy sorted data into
        std::vector<index_t> indptr_a;
        std::vector<std::pair<index_t, data_t>> data_pair;
        indptr_a.resize(this->n_outer + 1);
        data_pair.resize(this->nnz);

        // Set the outer pointer (based on nnz per outer idx)
        indptr_a[0] = 0;
        for (index_t ptr = 0; ptr < this->n_outer; ptr++)
        {
            indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
            ptr_nnz[ptr] = 0;
        }

        // Add inner indices and values, grouped by outer idx
        for (index_t i = 0; i < this->n_inner; i++)
        {
            index_t ptr_start = this->indptr[i];
            index_t ptr_end = this->indptr[i+1];

            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t new_ptr = this->indices[j];
                index_t pos = indptr_a[new_ptr] + ptr_nnz[new_ptr]++;
                data_pair[pos].first = i;
                data_pair[pos].second = this->data[j];
            }
        }

        // Sort each outer group (completely sorted matrix)
        for (index_t i = 0; i < this->n_outer; i++)
        {
            index_t ptr_start = indptr_a[i];
            index_t ptr_end = indptr_a[i+1];

            if (ptr_start < ptr_end)
            {
                std::sort(data_pair.begin()+ptr_start, data_pair.begin() + ptr_end, 
                    [](const std::pair<index_t, data_t>& lhs, 
                    const std::pair<index_t, data_t>& rhs) 
                        { return lhs.first < rhs.first; } );       
            }
        }

        // Add entries to compressed vectors 
        // (assume no zeros or duplicate entries)
        this->indptr.resize(this->n_outer + 1);
        index_t ctr = 0;
        for (index_t i = 0; i < this->n_outer; i++)
        {
            this->indptr[i] = ctr;
            index_t ptr_start = indptr_a[i];
            index_t ptr_end = indptr_a[i+1];
    
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                if (fabs(data_pair[j].second) > zero_tol)
                {
                    this->indices[ctr] = data_pair[j].first;
                    this->data[ctr] = data_pair[j].second;
                    ctr++;
                }
            }
        }
        this->indptr[this->n_outer] = ctr;

        indptr_a.clear();
        data_pair.clear();
    
        delete[] ptr_nnz;
    }

    // Convert from COO to a compressed format
    else if (this->format == COO)
    {
        // CSR - rows are outer indices
        if (_format == CSR)
        {
            this->n_outer = this->n_rows;
            this->n_inner = this->n_cols;
            this->indptr = this->row_idx;
            this->indices = this->col_idx;
        }

        // CSC - columns are outer indices
        else if (_format == CSC)
        {
            this->n_outer = this->n_cols;
            this->n_inner = this->n_rows;
            this->indptr = this->col_idx;
            this->indices = this->row_idx;
        }

        if (nnz == 0)
        {
            indptr.resize(n_outer+1);
            for (index_t i = 0; i < n_outer + 1; i++)
            {
                indptr[i] = 0;
            }
            return;
        }

        // Calculate the number of nonzeros per outer idx
        index_t* ptr_nnz = new index_t[this->n_outer]();
        for (index_t i = 0; i < this->nnz; i++)
        {
            ptr_nnz[this->indptr[i]]++;            
        }

        // Create vectors to copy sorted data into
        std::vector<index_t> indptr_a;
        std::vector<std::pair<index_t, data_t>> data_pair;
        indptr_a.resize(this->n_outer + 1);
        data_pair.resize(this->nnz);

        // Set the outer pointer (based on nnz per outer idx)
        indptr_a[0] = 0;
        for (index_t ptr = 0; ptr < this->n_outer; ptr++)
        {
            indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
            ptr_nnz[ptr] = 0;
        }

        // Add inner indices and values, grouped by outer idx
        for (index_t ctr = 0; ctr < this->nnz; ctr++)
        {
            index_t ptr = this->indptr[ctr];
            index_t pos = indptr_a[ptr] + ptr_nnz[ptr]++;
            data_pair[pos].first = this->indices[ctr];
            data_pair[pos].second = this->data[ctr];
        }

        // Sort each outer group (completely sorted matrix)
        for (index_t i = 0; i < this->n_outer; i++)
        {
            index_t ptr_start = indptr_a[i];
            index_t ptr_end = indptr_a[i+1];

            if (ptr_start < ptr_end)
            {
                std::sort(data_pair.begin()+ptr_start, data_pair.begin() + ptr_end, 
                    [](const std::pair<index_t, data_t>& lhs, 
                    const std::pair<index_t, data_t>& rhs) 
                        { return lhs.first < rhs.first; } );       
            }
        }

        // Add entries to compressed vectors
        this->indptr.resize(this->n_outer + 1);
        index_t ctr = 0;
        for (index_t i = 0; i < this->n_outer; i++)
        {
            this->indptr[i] = ctr;
            index_t ptr_start = indptr_a[i];
            index_t ptr_end = indptr_a[i+1];
    
            index_t j = ptr_start;

            // Always add first entry (if greater than zero)
            while (j < ptr_end)
            {
                if (fabs(data_pair[j].second) > zero_tol)
                {
                    this->indices[ctr] = data_pair[j].first;
                    this->data[ctr] = data_pair[j].second;
                    ctr++;
                    j++;
                    break;
                }  
                j++;
            }

            // Only add successive entries if they are different
            // Otherwise combine (add values together)
            for (; j < ptr_end; j++)
            {
                if (data_pair[j].first == this->indices[ctr-1])
                {
                    this->data[ctr-1] += data_pair[j].second;
                }
                else if (fabs(data_pair[j].second) > zero_tol)
                {
                    this->indices[ctr] = data_pair[j].first;
                    this->data[ctr] = data_pair[j].second;
                    ctr++;
                }
            }
        }
        this->indptr[this->n_outer] = ctr;

        // Clear vectors and array
        indptr_a.clear();
        data_pair.clear();
        delete[] ptr_nnz;
    }

    // Set new format
    this->format = _format;
}
