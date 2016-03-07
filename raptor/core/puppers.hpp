#include "types.hpp"
#include "par_matrix.hpp"
#include "par_vector.hpp"
#include "par_comm.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "array.hpp"

template <typename T> 
void pup_array(pup_er p, void* a_tmp)
{
    Array<T>* a_ptr = (Array<T>*) a_tmp;
    Array<T>& a = *a_ptr;

    pup_int(p, &(a.n));
    pup_int(p, &(a.alloc_n));
    
    if (pup_isUnpacking(p))
    {
        a.data_ptr = (T*) calloc (a.alloc_n, sizeof(T));
    }
    pup_bytes(p, (void*) a.data_ptr, a.alloc_n*sizeof(T));
    if (pup_isPacking(p))
    {
        free(a.data_ptr);
    }
}

void pup_vector(pup_er p, void* v_tmp)
{
    Vector** v_ptr = (Vector**) v_tmp;
    if (pup_isUnpacking(p))
    {
        *v_ptr = new Vector();
    }
    Vector* v = *v_ptr;

    pup_int(p, &(v->size));
    pup_array<data_t>(p, &(v->values));

    if (pup_isPacking(p))
    {
        delete v;
    }
}

void pup_par_vector(pup_er p, void* v_tmp)
{
    ParVector** v_ptr = (ParVector**) v_tmp;

    if (pup_isUnpacking(p))
    {
        *v_ptr = new ParVector();
    }
    ParVector* v = *v_ptr;

    pup_int(p, &(v->global_n));
    pup_int(p, &(v->local_n));
    pup_int(p, &(v->first_local));

    pup_vector(p, &(v->local));
    if (pup_isPacking(p))
    {
        delete v;
    }
}

void pup_par_comm(pup_er p, void* c_tmp)
{
    ParComm** c_ptr = (ParComm**) c_tmp;
    if (pup_isUnpacking(p))
    {
        *c_ptr = new ParComm();
    }
    ParComm* c = *c_ptr;

    pup_int(p, &(c->num_sends));
    pup_int(p, &(c->num_recvs));
    pup_int(p, &(c->size_sends));
    pup_int(p, &(c->size_recvs));

    pup_array<index_t>(p, &(c->send_procs));
    pup_array<index_t>(p, &(c->send_row_starts));
    pup_array<index_t>(p, &(c->send_row_indices));
    pup_array<index_t>(p, &(c->recv_procs));
    pup_array<index_t>(p, &(c->recv_col_starts));
    pup_array<index_t>(p, &(c->col_to_proc));

    if (pup_isUnpacking(p))
    {
        c->send_requests = new MPI_Request[c->num_sends];
        c->recv_requests = new MPI_Request[c->num_recvs];
        c->send_buffer = new data_t[c->size_sends];
        c->recv_buffer = new data_t[c->size_recvs];
    }
    pup_bytes(p, (void*) c->send_requests, c->num_sends*sizeof(MPI_Request));
    pup_bytes(p, (void*) c->recv_requests, c->num_recvs*sizeof(MPI_Request));
    pup_doubles(p, c->send_buffer, c->size_sends);
    pup_doubles(p, c->recv_buffer, c->size_recvs);
    if (pup_isPacking(p))
    {
        delete[] c->send_requests;
        delete[] c->recv_requests;
        delete[] c->send_buffer;
        delete[] c->recv_buffer;
    }
}

void pup_matrix(pup_er p, void* m_tmp)
{
    Matrix** m_ptr = (Matrix**) m_tmp;

    if (pup_isUnpacking(p))
    {
        *m_ptr = new Matrix();
    }
    Matrix* m = *m_ptr;

    pup_int(p, &(m->n_rows));
    pup_int(p, &(m->n_cols));
    pup_int(p, &(m->n_outer));
    pup_int(p, &(m->n_inner));
    pup_int(p, &(m->nnz));

    pup_bytes(p, (void*) &(m->format), sizeof(format_t));

    pup_array<index_t>(p, &(m->indptr));
    pup_array<index_t>(p, &(m->indices));
    pup_array<data_t>(p, &(m->data));
}

void pup_par_matrix(pup_er p, void* m_tmp)
{
    ParMatrix** m_ptr = (ParMatrix**) m_tmp;

    if (pup_isUnpacking(p))
    {
        *m_ptr = new ParMatrix();
    }

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

    pup_array<index_t>(p, &(m->local_to_global));
    pup_array<index_t>(p, &(m->global_col_starts));

    if (pup_isUnpacking(p))
    {
        m->diag = new Matrix();
        m->offd = new Matrix();
        m->diag_elmts = new data_t[m->local_rows];
        m->comm = new ParComm();
        for (int i = 0; i < m->local_to_global.size(); i++)
        {
            m->global_to_local[m->local_to_global[i]] = i;
        }
    }

    pup_matrix(p, &(m->diag));
    pup_matrix(p, &(m->offd));
    pup_par_comm(p, &(m->comm));
    pup_doubles(p, m->diag_elmts, m->local_rows);

    if (pup_isPacking(p))
    {
        delete m->diag;
        delete m->offd;
        delete m->comm;
        delete[] m->diag_elmts;
        m->global_to_local.clear();
        delete m;
    }
}

void pup_par_level(pup_er p, void* l_tmp)
{
    Level** l_ptr = (Level**) l_tmp;
    
    if (pup_isUnpacking(p))
    {
        *l_ptr = new Level();
    }
    Level* l = *l_ptr;

    pup_int(p, &(l->idx));
    pup_bytes(p, &(l->coarsest), sizeof(bool));
    pup_bytes(p, &(l->has_vec), sizeof(bool));

    pup_par_matrix(p, &(l->A));

    if (!(l->coarsest))
    {
        pup_par_matrix(p, &(l->P));
        pup_par_vector(p, &(l->tmp));
    }

    if (l->has_vec)
    {
        pup_par_vector(p, &(l->x));
        pup_par_vector(p, &(l->b));
    }

    if (pup_isPacking(p))
    {
        delete l;
    }
}



