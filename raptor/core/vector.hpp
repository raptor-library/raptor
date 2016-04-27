// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "types.hpp"
#include "array.hpp"
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <cmath>

/**************************************************************
 *****   Vector Class
 **************************************************************
 ***** This class constructs a vector, supporting simple linear
 ***** algebra operations.
 *****
 ***** Attributes
 ***** -------------
 ***** values : std::vector<data_t>
 *****    Array of vector values
 ***** size : index_t
 *****    Dimension of vector
 ***** 
 ***** Methods
 ***** -------
 ***** set_const_value(data_t alpha)
 *****    Sets the vector to a constant value
 ***** set_rand_values()
 *****    Sets each element of the vector to a random value
 ***** axpy(Vector& y, data_t alpha)
 *****    Multiplies each element by a constant, alpha, and then
 *****    adds corresponding values from y
 ***** scale(data_t alpha)
 *****    Multiplies entries of vector by a constant
 ***** norm(index_t p)
 *****    Calculates the p-norm of the vector
 ***** data()
 *****    Returns the data values as a data_t*
 **************************************************************/

namespace raptor
{
class Vector
{

public:
    /**************************************************************
    *****   Vector Class Constructor
    **************************************************************
    ***** Initializes an empty vector of the given size
    *****
    ***** Parameters
    ***** -------------
    ***** len : index_t
    *****    Size of the vector
    **************************************************************/
    Vector(index_t len)
    {
        values.resize(len);
        size = len;
    }

    /**************************************************************
    *****   Vector Class Constructor
    **************************************************************
    ***** Initializes an empty vector without setting the size
    **************************************************************/
    Vector()
    {
        size = 0;
    }

    /**************************************************************
    *****   Vector Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~Vector()
    {

    }

    /**************************************************************
    *****   Vector Set Constant Value
    **************************************************************
    ***** Initializes the vector to a constant value
    *****
    ***** Parameters
    ***** -------------
    ***** alpha : data_t
    *****    Constant value to set each element of vector to
    **************************************************************/
    void set_const_value(data_t alpha);

    
    /**************************************************************
    *****   Vector Set Random Values
    **************************************************************
    ***** Initializes each element of the vector to a random
    ***** value
    **************************************************************/
    void set_rand_values();
    
    /**************************************************************
    *****   Vector AXPY
    **************************************************************
    ***** Multiplies the vector by a constant, alpha, and then
    ***** sums each element with corresponding entry of Y
    *****
    ***** Parameters
    ***** -------------
    ***** y : Vector&
    *****    Vector to be summed with
    ***** alpha : data_t
    *****    Constant value to multiply each element of vector by
    **************************************************************/
    void axpy(Vector& y, data_t alpha);
    
    /**************************************************************
    *****   Vector Scale
    **************************************************************
    ***** Multiplies each element of the vector by a constant value
    *****
    ***** Parameters
    ***** -------------
    ***** alpha : data_t
    *****    Constant value to set multiply element of vector by
    **************************************************************/
    void scale(data_t alpha);
    
    /**************************************************************
    *****   Vector Norm
    **************************************************************
    ***** Calculates the P norm of the vector (for a given P)
    *****
    ***** Parameters
    ***** -------------
    ***** p : index_t
    *****    Determines which p-norm to calculate
     **************************************************************/
    data_t norm(index_t p);

    /**************************************************************
    *****   Vector Data
    **************************************************************
    ***** Returns pointer to vector entries
    *****
    ***** Returns
    ***** -------------
    ***** data_t*
    *****    Pointer to values of vector
    **************************************************************/
    data_t* data();

    Array<data_t> values;
    index_t size;
};

}


#endif
