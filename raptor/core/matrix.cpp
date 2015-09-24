// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "matrix.hpp"

Matrix::Matrix(Matrix* A)
{

}

Matrix::Matrix()
{
    nnz = 0;
    format = COO;
}

Matrix::~Matrix()
{

}

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


void Matrix::add_value(index_t row, index_t col, data_t value)
{
    //    Insert a single value into the matrix.
    //
    //    Parameters
    //    ----------
    //    ptr : index_t
    //        Outer index of nonzero.
    //    idx : index_t*
    //        Inner index of nonzero.
    //    value : data_t
    //        Data value of nonzero.

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
 
void Matrix::resize(index_t _nrows, index_t _ncols)
{
    //    Resize the dimensions of the matrix.
    //
    //    Parameters
    //    ----------
    //    _nrows : index_t
    //        New number of rows in the matrix.
    //    _ncols : index_t
    //        New number of columns in the matrix.
    //    _nouter : index_t
    //        New number of outer indices in the matrix.

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

void Matrix::finalize(format_t _format)
{
    //    Converts the matrix into a useable, compressed form.  If
    //    the matrix is initialized as a COO matrix (init_coo == 1)
    //    the matrix is converted into the compressed format, and 
    //    the COO matrix is cleared.  If the matrix was initialized 
    //    directly into a compressed format, zeros are removed.  The
    //    vector ptr_nnz is cleared.

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


void Matrix::convert(format_t _format)
{
    if (format == _format)
    {
        return;
    }

    if ((this->format == CSR || this->format == CSC) 
        && (_format == CSR || _format == CSC))
    {
        index_t n_tmp = this->n_inner;
        this->n_inner = this->n_outer;
        this->n_outer = n_tmp;

        index_t* ptr_nnz = new index_t[this->n_outer]();
        // Calculate nnz per outer
        for (index_t i = 0; i < this->nnz; i++)
        {
            ptr_nnz[this->indices[i]]++;      
        }

        std::vector<index_t> indptr_a;
        std::vector<std::pair<index_t, data_t>> data_pair;
        indptr_a.resize(this->n_outer + 1);
        data_pair.resize(this->nnz);

        indptr_a[0] = 0;
        for (index_t ptr = 0; ptr < this->n_outer; ptr++)
        {
            indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
            ptr_nnz[ptr] = 0;
        }

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
    else if (this->format == COO) // Convert COO to CSR/CSC
    {
        if (_format == CSR)
        {
            this->n_outer = this->n_rows;
            this->n_inner = this->n_cols;
            this->indptr = this->row_idx;
            this->indices = this->col_idx;
        }
        else if (_format == CSC)
        {
            this->n_outer = this->n_cols;
            this->n_inner = this->n_rows;
            this->indptr = this->col_idx;
            this->indices = this->row_idx;
        }

        index_t* ptr_nnz = new index_t[this->n_outer]();
        // Calculate nnz per outer
        for (index_t i = 0; i < this->nnz; i++)
        {
            ptr_nnz[this->indptr[i]]++;            
        }

        std::vector<index_t> indptr_a;
        std::vector<std::pair<index_t, data_t>> data_pair;

        indptr_a.resize(this->n_outer + 1);
        data_pair.resize(this->nnz);

        indptr_a[0] = 0;
        for (index_t ptr = 0; ptr < this->n_outer; ptr++)
        {
            indptr_a[ptr+1] = indptr_a[ptr] + ptr_nnz[ptr];
            ptr_nnz[ptr] = 0;
        }

        for (index_t ctr = 0; ctr < this->nnz; ctr++)
        {
            index_t ptr = this->indptr[ctr];
            index_t pos = indptr_a[ptr] + ptr_nnz[ptr]++;
            data_pair[pos].first = this->indices[ctr];
            data_pair[pos].second = this->data[ctr];
        }

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

        this->indptr.resize(this->n_outer + 1);
        index_t ctr = 0;
        for (index_t i = 0; i < this->n_outer; i++)
        {
            this->indptr[i] = ctr;
            index_t ptr_start = indptr_a[i];
            index_t ptr_end = indptr_a[i+1];
    
            index_t j = ptr_start;
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

        indptr_a.clear();
        data_pair.clear();
        delete[] ptr_nnz;
    }

    this->format = _format;
}
