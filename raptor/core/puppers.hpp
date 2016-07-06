// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_CORE_PUPPERS_HPP
#define RAPTOR_CORE_PUPPERS_HPP

#include "types.hpp"
#include "par_matrix.hpp"
#include "par_vector.hpp"
#include "par_comm.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "array.hpp"
#include "level.hpp"
#include "hierarchy.hpp"

/**************************************************************
 *****  Pupper Method for Array Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the Array class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** a_tmp : void*
 *****    Pointer to the Array
 **************************************************************/
template <typename T> 
void pup_array_helper(pup_er p, void* a_tmp)
{
    Array<T>* a_ptr = (Array<T>*) a_tmp;
    Array<T>& a = *a_ptr;

    pup_int(p, &(a.n));
    pup_int(p, &(a.alloc_n));
    
    if (pup_isUnpacking(p))
    {
        if (a.alloc_n)
        {
            a.reserve(a.alloc_n);
        }
    }
    pup_bytes(p, (void*) a.data_ptr, a.alloc_n*sizeof(T));
}

/**************************************************************
 *****  Pupper Method for Array Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the Array class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** a_tmp : void*
 *****    Pointer to the Array
 **************************************************************/
template <typename T>
void pup_array(pup_er p, void* a_tmp)
{
    Array<T>* a_ptr = (Array<T>*) a_tmp;
    Array<T>& a = *a_ptr;

    pup_array_helper<T>(p, &a);

    if (pup_isDeleting(p))
    {
        if (a.alloc_n)
        {
            free(a.data_ptr);
        }
    }
}

/**************************************************************
 *****  Pupper Method for Vector Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the Vector class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** v_tmp : void*
 *****    Pointer to the Vector
 **************************************************************/
void pup_vector_helper(pup_er p, void* v_tmp)
{
    Vector** v_ptr = (Vector**) v_tmp;
    Vector* v = *v_ptr;

    pup_int(p, &(v->size));

    if (v->size)
    {
        pup_array_helper<data_t>(p, &(v->values));
    }
}

/**************************************************************
 *****  Pupper Method for Vector Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the Vector class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** v_tmp : void*
 *****    Pointer to the Vector
 **************************************************************/
void pup_vector(pup_er p, void* v_tmp)
{
    Vector** v_ptr = (Vector**) v_tmp;
    if (pup_isUnpacking(p))
    {
        *v_ptr = new Vector();
    }
    Vector* v = *v_ptr;

    pup_vector_helper(p, &v);

    if (pup_isDeleting(p))
    {
        delete v;
    }
}

/**************************************************************
 *****  Pupper Method for ParVector Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the ParVector class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** v_tmp : void*
 *****    Pointer to the ParVector
 **************************************************************/
void pup_par_vector_helper(pup_er p, void* v_tmp)
{
    ParVector** v_ptr = (ParVector**) v_tmp;
    ParVector* v = *v_ptr;

    pup_int(p, &(v->global_n));
    pup_int(p, &(v->local_n));
    pup_int(p, &(v->first_local));

    if (v->local_n)
    {
        v->local = new Vector(v->local_n);
        pup_vector_helper(p, &(v->local));
    }
}

/**************************************************************
 *****  Pupper Method for ParVector Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the ParVector class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** v_tmp : void*
 *****    Pointer to the ParVector
 **************************************************************/
void pup_par_vector(pup_er p, void* v_tmp)
{
    ParVector** v_ptr = (ParVector**) v_tmp;

    if (pup_isUnpacking(p))
    {
        *v_ptr = new ParVector();
    }
    ParVector* v = *v_ptr;

    pup_par_vector_helper(p, &v);

    if (pup_isDeleting(p))
    {
        delete v;
    }
}

/**************************************************************
 *****  Pupper Method for ParComm Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the ParComm class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParComm
 **************************************************************/
void pup_par_comm_helper(pup_er p, void* c_tmp)
{
    ParComm** c_ptr = (ParComm**) c_tmp;
    ParComm* c = *c_ptr;

    pup_int(p, &(c->num_sends));
    pup_int(p, &(c->num_recvs));
    pup_int(p, &(c->size_sends));
    pup_int(p, &(c->size_recvs));

    pup_array_helper<index_t>(p, &(c->send_procs));
    pup_array_helper<index_t>(p, &(c->send_row_starts));
    pup_array_helper<index_t>(p, &(c->send_row_indices));
    pup_array_helper<index_t>(p, &(c->recv_procs));
    pup_array_helper<index_t>(p, &(c->recv_col_starts));
    pup_array_helper<index_t>(p, &(c->col_to_proc));

    if (pup_isUnpacking(p))
    {
        if (c->num_sends)
        {
            c->send_requests = new MPI_Request[c->num_sends];
            c->send_buffer = new data_t[c->size_sends];
        }
        if (c->num_recvs)
        {
            c->recv_requests = new MPI_Request[c->num_recvs];
            c->recv_buffer = new data_t[c->size_recvs];
        }
    }

    if (c->num_sends)
    {
        pup_bytes(p, (void*) c->send_requests, c->num_sends*sizeof(MPI_Request));
        pup_doubles(p, c->send_buffer, c->size_sends);
    }
    if (c->num_recvs)
    {
        pup_bytes(p, (void*) c->recv_requests, c->num_recvs*sizeof(MPI_Request));
        pup_doubles(p, c->recv_buffer, c->size_recvs);
    }
}

/**************************************************************
 *****  Pupper Method for ParComm Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the ParComm class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParComm
 **************************************************************/
void pup_par_comm(pup_er p, void* c_tmp)
{
    ParComm** c_ptr = (ParComm**) c_tmp;
    if (pup_isUnpacking(p))
    {
        *c_ptr = new ParComm();
    }
    ParComm* c = *c_ptr;

    pup_par_comm_helper(p, &c);

    if (pup_isDeleting(p))
    {
        if (c->num_sends)
        {
            delete[] c->send_requests;
            delete[] c->send_buffer;
        }
        if (c->num_recvs)
        {
            delete[] c->recv_requests;
            delete[] c->recv_buffer;
        }
    }
}

/**************************************************************
 *****  Pupper Method for Matrix Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the Matrix class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the Matrix
 **************************************************************/
void pup_matrix_helper(pup_er p, void* m_tmp)
{
    Matrix** m_ptr = (Matrix**) m_tmp;
    Matrix* m = *m_ptr;

    pup_int(p, &(m->n_rows));
    pup_int(p, &(m->n_cols));
    pup_int(p, &(m->n_outer));
    pup_int(p, &(m->n_inner));
    pup_int(p, &(m->nnz));

    pup_bytes(p, (void*) &(m->format), sizeof(format_t));

    pup_array_helper<index_t>(p, &(m->indptr));
    pup_array_helper<index_t>(p, &(m->indices));
    pup_array_helper<data_t>(p, &(m->data));
}
    
/**************************************************************
 *****  Pupper Method for Matrix Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the Matrix class
 ***** in AMPI.  NOT be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the Matrix
 **************************************************************/
void pup_matrix(pup_er p, void* m_tmp)
{
    Matrix** m_ptr = (Matrix**) m_tmp;

    if (pup_isUnpacking(p))
    {
        *m_ptr = new Matrix();
    }
    Matrix* m = *m_ptr;

    if (pup_isDeleting(p))
    {
        delete m;
    }
}

/**************************************************************
 *****  Pupper Method for ParMatrix Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the ParMatrix class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParMatrix
 **************************************************************/
void pup_par_matrix_helper(pup_er p, void* m_tmp)
{
    ParMatrix** m_ptr = (ParMatrix**) m_tmp;
    ParMatrix* m = *m_ptr;

    pup_int(p, &(m->global_rows));
    pup_int(p, &(m->global_cols));
    pup_int(p, &(m->local_nnz));
    pup_int(p, &(m->local_rows));
    pup_int(p, &(m->local_cols)); 
    pup_int(p, &(m->offd_num_cols));
    pup_int(p, &(m->first_col_diag));
    pup_int(p, &(m->first_row));
    pup_int(p, &(m->comm_mat));

    pup_array_helper<index_t>(p, &(m->local_to_global));
    pup_array_helper<index_t>(p, &(m->global_col_starts));

    if (pup_isUnpacking(p))
    {
        if (m->offd_num_cols)
        {
            m->offd = new Matrix(m->local_rows, m->offd_num_cols, CSC);
        }
        m->diag = new Matrix(m->local_rows, m->local_rows, CSR);
        m->comm = new ParComm();
    }

    pup_matrix_helper(p, &(m->diag));
    pup_par_comm_helper(p, &(m->comm));
    if (m->offd_num_cols)
    {
        pup_matrix_helper(p, &(m->offd));
    }
}

/**************************************************************
 *****  Pupper Method for ParMatrix Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the ParMatrix class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParMatrix
 **************************************************************/
void pup_par_matrix(pup_er p, void* m_tmp)
{
    ParMatrix** m_ptr = (ParMatrix**) m_tmp;

    if (pup_isUnpacking(p))
    {
        *m_ptr = new ParMatrix();
    }

    ParMatrix* m = *m_ptr;

    pup_par_matrix_helper(p, &m);

    if (pup_isDeleting(p))
    {
        delete m;
    }
}

/**************************************************************
 *****  Pupper Method for ParLevel Class (Helper)
 **************************************************************
 ***** Packs, transfers, and unpacks the ParLevel class
 ***** in AMPI.  Must be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParLevel
 **************************************************************/
void pup_par_level_helper(pup_er p, void* l_tmp)
{
    Level** l_ptr = (Level**) l_tmp;
    Level* l = *l_ptr;

    pup_int(p, &(l->idx));
    pup_bytes(p, &(l->coarsest), sizeof(bool));
    pup_bytes(p, &(l->has_vec), sizeof(bool));

    if (pup_isUnpacking(p))
    {
        l->A = new ParMatrix();

        if (l->has_vec)
        {
            l->b = new ParVector();
            l->x = new ParVector();
        }
        else
        {
            l->b = NULL;
            l->x = NULL;
        }
        l->P = NULL;
        l->tmp = NULL;
    }

    pup_par_matrix_helper(p, &(l->A));

    if (l->has_vec)
    {
        pup_par_vector_helper(p, &(l->b));
        pup_par_vector_helper(p, &(l->x));
    }
}

/**************************************************************
 *****  Pupper Method for ParLevel Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the ParLevel class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the ParLevel
 **************************************************************/
void pup_par_level(pup_er p, void* l_tmp)
{
    Level** l_ptr = (Level**) l_tmp;
    
    if (pup_isUnpacking(p))
    {
        *l_ptr = new Level();
    }
    Level* l = *l_ptr;

    pup_par_level_helper(p, &l);

    if (pup_isDeleting(p))
    {
        delete l;
    }

}

/**************************************************************
 *****  Pupper Method for Hierarchy Class 
 **************************************************************
 ***** Packs, transfers, and unpacks the Hierarchy class
 ***** in AMPI.  NOT to be called from another pup method.
 *****
 ***** Parameters
 ***** -------------
 ***** p : pup_er
 *****    Pup AMPI Object
 ***** c_tmp : void*
 *****    Pointer to the Hierarchy
 **************************************************************/
void pup_hierarchy(pup_er p, void* h_tmp)
{
    Hierarchy** h_ptr = (Hierarchy**) h_tmp;
    
    if (pup_isUnpacking(p))
    {
        *h_ptr = new Hierarchy();
    }
    Hierarchy* h = *h_ptr;

    pup_int(p, &(h->num_levels));

    if (pup_isUnpacking(p))
    {
        for (int i = 0; i < h->num_levels; i++)
        {
            h->levels.push_back(new Level());
        }
    }

    for (int i = 0; i < h->num_levels; i++)
    {
        pup_par_level_helper(p, &(h->levels[i]));
    }

    if (pup_isDeleting(p))
    {
        delete h;
    }       
}

#endif
