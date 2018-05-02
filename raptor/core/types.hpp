// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TYPES_HPP_
#define RAPTOR_CORE_TYPES_HPP_

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <set>

#include <cstdint>
#include <vector>
#include <stdexcept>

using namespace std;

#define zero_tol 1e-16
#define MPI_INDEX_T MPI_INT
#define MPI_DATA_T MPI_DOUBLE

#ifdef WITH_MPI
#include <mpi.h>
#endif

template <typename T, std::size_t Alignment>
        class AlignAllocator
{
public:

        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T value_type;
        typedef std::size_t size_type;
        typedef ptrdiff_t difference_type;

        T * address(T& r) const
        {
                return &r;
        }

        const T * address(const T& s) const
        {
                return &s;
        }

        std::size_t max_size() const
        {
                return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
        }


        template <typename U>
                struct rebind
                {
                        typedef AlignAllocator<U, Alignment> other;
                } ;

        bool operator!=(const AlignAllocator& other) const
        {
                return !(*this == other);
        }

        void construct(T * const p, const T& t) const
        {
                void * const pv = static_cast<void *>(p);

                new (pv) T(t);
        }

        void destroy(T * const p) const
        {
                p->~T();
        }

        bool operator==(const AlignAllocator& other) const
        {
                return true;
        }

        AlignAllocator() { }

        AlignAllocator(const AlignAllocator&) { }

        template <typename U> AlignAllocator(const AlignAllocator<U, Alignment>&) { }

        ~AlignAllocator() { }

        T * allocate(const std::size_t n) const
        {
                if (n == 0) {
                        return NULL;
                }

                if (n > max_size())
                {
                        throw std::length_error("AlignAllocator<T>::allocate() - Integer overflow.");
                }

                void *pv;
                int ret = posix_memalign(&pv, Alignment, n*sizeof(T));

                if (ret != 0)
                {
                        throw std::bad_alloc();
                }

                return static_cast<T *>(pv);
        }

        void deallocate(T * const p, const std::size_t n) const
        {
                free(p);
        }


        template <typename U>
                T * allocate(const std::size_t n, const U * /* const hint */) const
        {
                return allocate(n);
        }

private:
        AlignAllocator& operator=(const AlignAllocator&);
};



namespace raptor
{
    using data_t = double;
    using index_t = int;
    template <typename T>
    using aligned_vector = std::vector<T, AlignAllocator<T, 16>>;
    enum format_t {CSR, CSC, COO};
    enum coarsen_t {RS, CLJP, Falgout, PMIS, HMIS};
    enum interp_t {Direct, Classical, Extended};
    enum relax_t {Jacobi, SOR, SSOR};

}

#endif
