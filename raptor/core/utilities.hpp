#ifndef RAPTOR_CORE_UTILITIES_HPP
#define RAPTOR_CORE_UTILITIES_HPP

#include "core/types.hpp"

using namespace raptor;

template <typename T, typename U>
void vec_sort(aligned_vector<T>& vec1, aligned_vector<U>& vec2, int start = 0, int end = -1)
{
    int k, prev_k;
    int n = vec1.size();
    if (end < 0) end = n;
    int size = end - start;

    aligned_vector<int> p(size);
    aligned_vector<bool> done(size, false);

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
void vec_sort(aligned_vector<T>& vec1, aligned_vector<T>& vec2, 
        aligned_vector<U>& vec3,
        int start = 0, int end = -1)
{
    int k, prev_k;
    int n = vec1.size();
    if (end < 0) end = n;
    int size = end - start;

    aligned_vector<int> p(size);
    aligned_vector<bool> done(size, false);

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
