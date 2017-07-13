// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARVECTOR_HPP
#define RAPTOR_CORE_PARVECTOR_HPP

#include <mpi.h>
#include <math.h>

#include "types.hpp"
#include "vector.hpp"
#include "comm_pkg.hpp"

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
 ***** first_local : index_t
 *****    Position of local vector inside the global vector
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
    class ParComm;
    class TAPComm;

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
        ***** first_lcl : index_t
        *****    Position of local vector inside global vector
        **************************************************************/
        ParVector(index_t glbl_n, int lcl_n, index_t first_lcl)
        {
            set_size(glbl_n, lcl_n, first_lcl);
        }

        /**************************************************************
        *****   ParVector Class Constructor
        **************************************************************
        ***** Creates an empy ParVector (local_n = 0)
        **************************************************************/
        ParVector()
        {
            local_n = 0;
        }

        /**************************************************************
        *****   ParVector Class Destructor
        **************************************************************
        ***** Deletes the local vector
        **************************************************************/
        ~ParVector()
        {
        }

        void set_size(index_t glbl_n, int lcl_n, index_t first_lcl)
        {
            global_n = glbl_n;
            local_n = lcl_n;
            first_local = first_lcl;
            local.set_size(local_n);
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
        void set_rand_values();

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
        void axpy(ParVector* y, data_t alpha);

        /**************************************************************
        *****   Vector Copy
        **************************************************************
        ***** Copies values of local vector in y into local 
        *****
        ***** Parameters
        ***** -------------
        ***** y : ParVector* y
        *****    ParVector to be copied
        **************************************************************/
        void copy(ParVector* y);
        
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
        data_t norm(index_t p);

        Vector& communicate(CommPkg* comm_pkg, MPI_Comm comm = MPI_COMM_WORLD);
        void init_comm(CommPkg* comm_pkg, MPI_Comm comm = MPI_COMM_WORLD);
        Vector& complete_comm(CommPkg* comm_pkg);


        Vector local;
        int global_n;
        int local_n;
        int first_local;
    };

}
#endif
