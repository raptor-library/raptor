#ifndef RAPTOR_CORE_UTILITIES_HPP
#define RAPTOR_CORE_UTILITIES_HPP

#include <limits>
#include <type_traits>
#include <cassert>

#include "types.hpp"

// BLAS LU routine that is used for coarse solve
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda,
        int* ipiv, int* info);
extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A,
        int *LDA, int *IPIV, double *B, int *LDB, int *INFO );
extern "C" void dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK, int *LWORK, int *info);

namespace raptor {
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
        while (i != k)
        {
            std::swap(vec1[prev_k + start], vec1[k + start]);
            std::swap(vec2[prev_k + start], vec2[k + start]);
            std::swap(vec3[prev_k + start], vec3[k + start]);
            done[k] = true;
            prev_k = k;
            k = p[k];
        }
    }
}


enum extents : std::size_t {
  dynamic_extent = std::numeric_limits<std::size_t>::max()
};
template <std::size_t E>
struct extent_storage
{
	extent_storage(std::size_t) {}
	constexpr std::size_t value() const { return E; }
};
template <>
struct extent_storage<dynamic_extent>
{
	constexpr std::size_t value() const { return e; }
	std::size_t e;
};


template <class T, std::size_t Extent = dynamic_extent>
struct span {
	using element_type = T;
	using value_type = typename std::remove_cv<T>::type;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using iterator = T*;
	using reverse_iterator = std::reverse_iterator<iterator>;

	static constexpr std::size_t extent = Extent;


	template<std::size_t E = Extent,
	         class = typename std::enable_if<E == dynamic_extent || E == 0>::type>
	span() : b(nullptr), ext{0} {}

	constexpr span(pointer p, size_type s) : b(p), ext{s} {}

	constexpr span(std::vector<T> & v) :
		span(v.data(), v.size()) {}

	constexpr iterator begin() const noexcept {
		return b;
	}

	constexpr iterator end() const noexcept {
		return b + size();
	}

	constexpr reverse_iterator rbegin() const noexcept {
		return reverse_iterator(end());
	}

	constexpr reverse_iterator rend() const noexcept {
		return reverse_iterator(begin());
	}

	constexpr reference front() const {
		return *b;
	}

	constexpr reference back() const {
		return *(b + (size() - 1));
	}

	constexpr reference operator[](size_type idx) const {
		return begin()[idx];
	}

	constexpr pointer data() const noexcept {
		return b;
	}

	constexpr size_type size() const noexcept {
		return ext.value();
	}

	constexpr size_type size_bytes() const noexcept {
		return sizeof(T)*size();
	}

	[[nodiscard]] constexpr bool empty() const noexcept {
		return size() == 0;
	}

	template<size_type Count>
	constexpr span<T, Count> first() const noexcept {
		return {b, Count};
	}

	constexpr span<T, dynamic_extent> first(size_type count) const noexcept {
		return {data(), count};
	}

	template<size_type Count>
	constexpr span<T, Count> last() const noexcept {
		return {data() + (size() - Count), Count};
	}

	constexpr span<T, dynamic_extent> last(size_type count) const noexcept {
		return {data() + (size() - count), count};
	}

protected:
	pointer b;
	extent_storage<Extent> ext;
};

}
#endif
