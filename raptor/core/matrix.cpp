// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "matrix.hpp"

using namespace raptor;

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
        indptr.push_back(row);
        indices.push_back(col);
    }
    else if (format == CSC)
    {
        indptr.push_back(col);
        indices.push_back(row);
    }
    data.push_back(value);
    nnz++;
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
    }
    else if (format == CSC)
    {
        this->n_outer = _ncols;
        this->n_inner = _nrows;
    }
}

/**************************************************************
 *****   Matrix ColToLocal
 **************************************************************
 ***** Converts column indices to be local values
 ***** 0 to number of cols
 *****
 ***** Parameters
 ***** -------------
 ***** map : std::map<index_t, index_t>
 *****    Maps global columns to local columns
 **************************************************************/
void Matrix::col_to_local(std::map<index_t, index_t>& map)
{
    if (format == CSR)
    {
        for (index_t i = 0; i < nnz; i++)
        {
            indices[i] = map[indices[i]];
        }
    }
    else
    {
        for (index_t i = 0; i < nnz; i++)
        {
            indptr[i] = map[indptr[i]];
        }
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
void Matrix::finalize()
{
    if (format == CSR)
    {
        n_outer = n_rows;
        n_inner = n_cols;
    }
    else
    {
        n_outer = n_cols;
        n_inner = n_rows;
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

    index_t* ptr_nnz = new index_t[this->n_outer]();
    for (index_t i = 0; i < nnz; i++)
    {
        ptr_nnz[indptr[i]]++;
    }

    // Create vectors to copy sorted data into
    std::vector<index_t> indptr_a;
    std::vector<std::pair<index_t, data_t>> data_pair;
    indptr_a.resize(n_outer + 1);
    data_pair.resize(nnz);

    // Set the outer pointer (based on nnz per outer idx)
    indptr_a[0] = 0;
    for (index_t ptr = 0; ptr < n_outer; ptr++)
    {
        indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
        ptr_nnz[ptr] = 0;
    }

    // Add inner indices and values, grouped by outer idx
    for (index_t ctr = 0; ctr < nnz; ctr++)
    {
        index_t ptr = indptr[ctr];
        index_t pos = indptr_a[ptr] + ptr_nnz[ptr]++;
        data_pair[pos].first = indices[ctr];
        data_pair[pos].second = data[ctr];
    }

    // Sort each outer group (completely sorted matrix)
    for (index_t i = 0; i < n_outer; i++)
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
    indptr.resize(n_outer + 1);
    nnz = 0;
    for (index_t i = 0; i < n_outer; i++)
    {
        indptr[i] = nnz;
        index_t ptr_start = indptr_a[i];
        index_t ptr_end = indptr_a[i+1];

        index_t j = ptr_start;

        // Always add first entry (if greater than zero)
        while (j < ptr_end)
        {
            if (fabs(data_pair[j].second) > zero_tol)
            {
                indices[nnz] = data_pair[j].first;
                data[nnz] = data_pair[j].second;
                nnz++;
                j++;
                break;
            }
            j++;
        }

        // Only add successive entries if they are different
        // Otherwise combine (add values together)
        for (; j < ptr_end; j++)
        {
            if (data_pair[j].first == indices[nnz-1])
            {
                data[nnz-1] += data_pair[j].second;
            }
            else if (fabs(data_pair[j].second) > zero_tol)
            {
                indices[nnz] = data_pair[j].first;
                data[nnz] = data_pair[j].second;
                nnz++;
            }
        }
    }
    indptr[n_outer] = nnz;

    // Delete array
    delete[] ptr_nnz;


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
    // Switch inner and outer indices
    index_t n_tmp = this->n_inner;
    this->n_inner = this->n_outer;
    this->n_outer = n_tmp;

    if (nnz == 0)
    {
        indptr.resize(n_outer+1);
        indices.resize(0);
        data.resize(0);
        for (index_t i = 0; i < n_outer + 1; i++)
        {
            indptr[i] = 0;
        }
        format = _format;
        return;
    }

    // Calculate number of nonzeros per outer idx
    index_t* ptr_nnz = new index_t[n_outer]();
    for (index_t i = 0; i < nnz; i++)
    {
        ptr_nnz[indices[i]]++;      
    }

    // Create vectors to copy sorted data into
    std::vector<index_t> indptr_a;
    std::vector<std::pair<index_t, data_t>> data_pair;
    indptr_a.resize(n_outer + 1);
    data_pair.resize(nnz);

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
    for (index_t i = 0; i < n_outer; i++)
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
    indptr.resize(n_outer + 1);
    index_t ctr = 0;
    for (index_t i = 0; i < n_outer; i++)
    {
        indptr[i] = ctr;
        index_t ptr_start = indptr_a[i];
        index_t ptr_end = indptr_a[i+1];
    
        for (index_t j = ptr_start; j < ptr_end; j++)
        {
            indices[ctr] = data_pair[j].first;
            data[ctr] = data_pair[j].second;
            ctr++;
        }
    }
    indptr[n_outer] = ctr;

    indptr_a.clear();
    data_pair.clear();
  
    delete[] ptr_nnz;

    // Set new format
    format = _format;
}
