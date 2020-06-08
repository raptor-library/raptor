// Copyright (c) 2015-2017, RAPtor Developer Team
    // License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
    #ifndef RAPTOR_CORE_PARVECTOR_HPP
    #define RAPTOR_CORE_PARVECTOR_HPP

    #include "assert.h"

    #include <mpi.h>
    #include <math.h>

    #include "mpi_types.hpp"
#ifndef NO_CUDA
    #include "cuda/vector_cuda.hpp"
#else
    #include "serial/vector.hpp"
#endif

    /**************************************************************
     *****   ParVector Class
     **************************************************************
     ***** This class constructs a parallel vector, containing 
     ***** values for a local portion
     *****
     ***** Attributes
     ***** -------------
     ***** local : Vector*
     *****    Local portion of the parallel vector
     ***** global_n : index_t
     *****    Number of entries in the global vector
     ***** local_n : index_t
     *****    Dimension of the local portion of the vector
     ***** 
     ***** Methods
     ***** -------
     ***** set_const_value(data_t alpha)
     *****    Sets the local vector to a constant value
     ***** set_rand_values()
 *****    Sets each element of the local vector to a random value
 ***** axpy(Vector& y, data_t alpha)
 *****    Performs axpy on local portion of vector
 ***** scale(data_t alpha)
 *****    Multiplies entries of the local vector by a constant
 ***** norm(index_t p)
 *****    Calculates the p-norm of the global vector
 **************************************************************/
namespace raptor
{
    class ParBVector;

    class ParVector
    {
    public:
        /**************************************************************
        *****   ParVector Class Constructor
        **************************************************************
        ***** Sets the dimensions of the global vector and initializes
        ***** an empty local vector of the given size
        *****
        ***** Parameters
        ***** -------------
        ***** glbl_n : index_t
        *****    Number of entries in global vector
        ***** lcl_n : index_t
        *****    Number of entries of global vector stored locally
        **************************************************************/
        ParVector(index_t glbl_n, int lcl_n, bool form_vec = true)
        {
            if (form_vec)
            {
                local = new Vector(lcl_n);
                resize(glbl_n, lcl_n);
            }
        }

        ParVector(const ParVector& x)
        {
            local = new Vector();
            copy(x);
        }

        /**************************************************************
        *****   ParVector Class Constructor
        **************************************************************
        ***** Creates an empy ParVector (local_n = 0)
        **************************************************************/
        ParVector()
        {
            local_n = 0;
            local = new Vector(local_n);
        }

        /**************************************************************
        *****   ParVector Class Destructor
        **************************************************************
        ***** Deletes the local vector
        **************************************************************/
        ~ParVector()
        {
            delete local;
        }

        void resize(index_t glbl_n, int lcl_n)
        {
            global_n = glbl_n;
            local_n = lcl_n;
            local->resize(local_n);
        }

        void copy(const ParVector& x)
        {
            global_n = x.global_n;
            local_n = x.local_n;
            local->copy(*(x.local));
        }

        /**************************************************************
        *****   ParVector Set Constant Value
        **************************************************************
        ***** Sets each element of the local vector to a constant value
        *****
        ***** Parameters
        ***** -------------
        ***** alpha : data_t
        *****    Value to set each element of local vector to
        **************************************************************/
        void set_const_value(data_t alpha);

        /**************************************************************
        *****   ParVector Set Random Values
        **************************************************************
        ***** Sets each element of the local vector to a random value
        **************************************************************/
        void set_rand_values(int seed = 0);

        /**************************************************************
        *****   Vector AXPY
        **************************************************************
        ***** Multiplies the local vector by a constant, alpha, and then
        ***** sums each element with corresponding entry of Y
        *****
        ***** Parameters
        ***** -------------
        ***** y : ParVector* y
        *****    ParVector to be summed with
        ***** alpha : data_t
        *****    Constant value to multiply each element of vector by
        **************************************************************/
        void axpy(ParVector& y, data_t alpha, double* comp_t = NULL);

        /**************************************************************
        *****   Vector Scale
        **************************************************************
        ***** Multiplies the local vector by a constant, alpha
        *****
        ***** Parameters
        ***** -------------
        ***** alpha : data_t
        *****    Constant value to multiply each element of vector by
        **************************************************************/
        void scale(data_t alpha);

        /**************************************************************
        *****   Vector Norm
        **************************************************************
        ***** Calculates the P norm of the global vector (for a given P)
        *****
        ***** Parameters
        ***** -------------
        ***** p : index_t
        *****    Determines which p-norm to calculate
        **************************************************************/
        data_t norm(index_t p, data_t* norms = NULL);

        data_t inner_product(ParVector& x, data_t* inner_prods = NULL, double* comp_t = NULL);  

        const data_t& operator[](const int index) const
        {
            return local->values[index];
        }

        data_t& operator[](const int index)
        {
            return local->values[index];
        }

        void split(ParVector& W, int t);
        void split_contig(ParVector& W, int t, int first_local);

        void add_val(data_t val, index_t vec, index_t global_n, index_t first_local);

        Vector* local;
        int global_n;
        int local_n;
    };

    class ParBVector : public ParVector
    {

    public:
        ParBVector(index_t glbl_n, int lcl_n, int vecs_in_block)
            : ParVector(glbl_n, lcl_n, false)
        {
            local = new BVector(lcl_n, vecs_in_block);
            global_n = glbl_n;
            local_n = lcl_n;
        }

        ParBVector() : ParVector()
        {
        }

        void scale(data_t alpha, data_t* alphas = NULL);
        void axpy_ij(ParBVector& y, index_t i, index_t j, data_t alpha, double* comp_t = NULL);
        data_t norm(index_t p, data_t* norms = NULL);

        data_t inner_product(ParBVector& x, data_t* inner_prods = NULL, double* comp_t = NULL);
        data_t inner_product(ParBVector& x, index_t i, index_t j, double* comp_t = NULL);

        void mult_T(ParVector& x, Vector& b, double* comp_t = NULL);
        void mult(Vector& x, ParVector& b, double* comp_t = NULL);

        void sum_cols(ParVector& b, double* comp_t = NULL);

        void append(ParBVector& P);
    };

}
#endif
