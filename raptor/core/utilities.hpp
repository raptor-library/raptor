#ifndef RAPTOR_CORE_UTILITIES_HPP
#define RAPTOR_CORE_UTILITIES_HPP

#include "core/types.hpp"

using namespace raptor;

// BLAS LU routine that is used for coarse solve
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, 
        int* ipiv, int* info);
extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, 
        int *LDA, int *IPIV, double *B, int *LDB, int *INFO );


template <typename T, typename U>
void vec_sort(std::vector<T>& vec1, std::vector<U>& vec2, int start = 0, int end = -1)
{
    vec1.shrink_to_fit();
    vec2.shrink_to_fit();

    int k, prev_k;
    int n = vec1.size();
    if (end < 0) end = n;
    int size = end - start;

    std::vector<int> p(size);
    std::vector<bool> done(size, false);

    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
            [&](const int i, const int j)
            {
                return vec1[i+start] < vec1[j+start];
            });
    for (int i = 0; i < size; i++)
    {
        if (done[i]) continue;
        done[i] = true;
        prev_k = i;
        k = p[i];
        while (i != k)
        {
            std::swap(vec1[prev_k + start], vec1[k + start]);
            std::swap(vec2[prev_k + start], vec2[k + start]);
            done[k] = true;
            prev_k = k;
            k = p[k];
        }
    }
}

template <typename T, typename U>
void vec_sort(std::vector<T>& vec1, std::vector<T>& vec2, 
        std::vector<U>& vec3,
        int start = 0, int end = -1)
{
    vec1.shrink_to_fit();
    vec2.shrink_to_fit();
    vec3.shrink_to_fit();

    int k, prev_k;
    int n = vec1.size();
    if (end < 0) end = n;
    int size = end - start;

    std::vector<int> p(size);
    std::vector<bool> done(size, false);

    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
            [&](const int i, const int j)
            {
                int idx1 = i + start;
                int idx2 = j + start;
                if (vec1[idx1] == vec1[idx2]) 
                    return vec2[idx1] < vec2[idx2];
                else
                    return vec1[idx1] < vec1[idx2];
            });
    for (int i = 0; i < size; i++)
    {
        if (done[i]) continue;
        done[i] = true;
        prev_k = i;
        k = p[i];
        int idx1 = prev_k + start;
        int idx2 = k + start;
        while (i != k)
        {
            std::swap(vec1[idx1], vec1[idx2]);
            std::swap(vec2[idx1], vec2[idx2]);
            std::swap(vec3[idx1], vec3[idx2]);
            done[k] = true;
            prev_k = k;
            k = p[k];
        }
    }
}



#endif
