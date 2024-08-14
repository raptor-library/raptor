// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <type_traits>

#include "core/par_matrix.hpp"
#include "core/matrix_traits.hpp"

using namespace raptor;

// Declare Private Methods
ParCSRMatrix* classical_strength(ParCSRMatrix* A, double theta, bool tap_amg, int num_variables,
        int* variables);
ParCSRMatrix* symmetric_strength(ParCSRMatrix* A, double theta, bool tap_amg);

namespace raptor::classical {
namespace {
void init_strength(const Matrix & A, Matrix & S) {
	    if (A.nnz) {
		    S.idx2.resize(A.nnz);
		    S.vals.resize(A.nnz);
	    }
	    S.idx1[0] = 0;
	    S.nnz = 0;
}
void finalize_strength(const ParCSRMatrix & A, ParCSRMatrix & S) {
	auto finalize_vecs = [](Matrix & mat) {
		mat.idx2.resize(mat.nnz);
		mat.idx2.shrink_to_fit();
		mat.vals.resize(mat.nnz);
		mat.vals.shrink_to_fit();
	};
	finalize_vecs(*S.on_proc);
	finalize_vecs(*S.off_proc);

	S.local_nnz = S.on_proc->nnz + S.off_proc->nnz;

	S.on_proc_column_map = A.on_proc_column_map;
    S.local_row_map = A.get_local_row_map();
    S.off_proc_column_map = A.off_proc_column_map;

    S.comm = A.comm;
    S.tap_comm = A.tap_comm;
    S.tap_mat_comm = A.tap_mat_comm;

    if (S.comm) S.comm->num_shared++;
    if (S.tap_comm) S.tap_comm->num_shared++;
    if (S.tap_mat_comm) S.tap_mat_comm->num_shared++;
}
struct positive_coupling {
	static constexpr inline double init = std::numeric_limits<double>::lowest();
	static constexpr bool comp(double a, double b) { return a > b; }
	static constexpr double strongest(double a, double b) { return std::max(a, b); }
};
struct negative_coupling {
	static constexpr inline double init = std::numeric_limits<double>::max();
	static constexpr bool comp(double a, double b) { return a < b; }
	static constexpr double strongest(double a, double b) { return std::min(a, b); }
};
template <strength_norm norm_type> struct norm_coupling {};
template<> struct norm_coupling<strength_norm::abs>
{
	static constexpr inline double init = std::numeric_limits<double>::min();
	static constexpr bool comp(double a, double b) { return std::abs(a) > b; }
	static constexpr double strongest(double a, double b) { return std::max(std::abs(a), b); }
};

template<class T>
struct mat_args {
	T & mat;
	const int * variables;
	int beg;
};
template <class T>
mat_args(T&, const int*, int)->mat_args<T>;

template <class P>
constexpr double value(CSRMatrix & mat, int i) {
	return mat.vals[i];
}
template <class P>
constexpr double value(BSRMatrix & mat, int i) {
	auto bvals = span<double>(mat.block_vals[i], mat.b_size);
	auto curr = P::init;
	for (auto val : bvals)
		if (P::comp(val, curr)) curr = val;
	return curr;
}
template <class P, bool filter, class T>
constexpr double strongest_element(int i, int row_var, const mat_args<T> & a) {
	auto curr = P::init;
	for (int j = a.beg; j < a.mat.idx1[i+1]; ++j) {
		auto col = a.mat.idx2[j];
		if constexpr (filter)
			if (row_var != a.variables[col]) continue;
		auto val = value<P>(a.mat, j);
		if (P::comp(val, curr))
			curr = val;
	}
	return curr;
}
template <class P, class T>
constexpr double strongest_connection(int row, int row_var, int num_variables,
                                      mat_args<T> on_proc, mat_args<T> off_proc) {
	if (num_variables == 1) {
		return P::strongest(
			strongest_element<P, false>(row, row_var, on_proc),
			strongest_element<P, false>(row, row_var, off_proc));
	} else {
		return P::strongest(
			strongest_element<P, true>(row, row_var, on_proc),
			strongest_element<P, true>(row, row_var, off_proc));
	}
}
template<class T>
struct append_args : mat_args<T> { Matrix & soc; };

template <class P, bool filter, class T>
constexpr void add_connections(int row, int row_var, double threshold,
                               const append_args<T> & args) {
	for (int j = args.beg; j < args.mat.idx1[row+1]; ++j) {
		auto col = args.mat.idx2[j];
		if constexpr (filter)
			if (row_var != args.variables[col]) continue;
		auto val = value<P>(args.mat, j);
		if (P::comp(val, threshold)) {
			args.soc.idx2[args.soc.nnz] = col;
			args.soc.vals[args.soc.nnz] = val;
			++args.soc.nnz;
		}
	}
}
template <class P, class T>
constexpr void add_strong_connections(int row, int row_var, int num_variables, double threshold,
                                      append_args<T> on_proc, append_args<T> off_proc) {
	if (num_variables == 1) {
		add_connections<P, false>(row, row_var, threshold, on_proc);
		add_connections<P, false>(row, row_var, threshold, off_proc);
	} else {
		add_connections<P, true>(row, row_var, threshold, on_proc);
		add_connections<P, true>(row, row_var, threshold, off_proc);
	}
}


void hybrid_strength(ParCSRMatrix & A, ParCSRMatrix & S,
                     double theta, int num_variables, int *variables, int *off_variables) {
	auto & on_proc = dynamic_cast<CSRMatrix&>(*A.on_proc);
	auto & off_proc = dynamic_cast<CSRMatrix&>(*A.off_proc);
	for (int i = 0; i < A.local_num_rows; i++)
    {
	    auto row_start_on = A.on_proc->idx1[i];
	    auto row_end_on = A.on_proc->idx1[i+1];
	    auto row_start_off = A.off_proc->idx1[i];
	    auto row_end_off = A.off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
	        double diag;
            if (A.on_proc->idx2[row_start_on] == i)
            {
	            diag = A.on_proc->vals[row_start_on];
                row_start_on++;
            }
            else
            {
                diag = 0.0;
            }

            auto row_var = (num_variables > 1) ? variables[i] : -1;
            // Find value with max magnitude in row
            auto row_scale = [&]() {
	            auto get_row_scale = [&](auto comp) {
		            using P = decltype(comp);
		            return strongest_connection<P, CSRMatrix>(i, row_var, num_variables,
		                                                      {on_proc, variables, row_start_on},
		                                                      {off_proc, off_variables, row_start_off});
	            };
	            if (diag < 0.0)
		            return get_row_scale(positive_coupling{});
	            else
		            return get_row_scale(negative_coupling{});
            }();

            // Multiply row max magnitude by theta
            auto threshold = row_scale * theta;

            // Always add diagonal
            S.on_proc->idx2[S.on_proc->nnz] = i;
            S.on_proc->vals[S.on_proc->nnz] = diag;
            S.on_proc->nnz++;

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to
            // row_max * theta
            auto add_row = [&](auto comp) {
	            using P = decltype(comp);
	            add_strong_connections<P, CSRMatrix>(i, row_var, num_variables, threshold,
	                                                 {{on_proc, variables, row_start_on}, *S.on_proc},
	                                                 {{off_proc, off_variables, row_start_off}, *S.off_proc});
            };
            if (diag < 0)
	            add_row(positive_coupling{});
            else
	            add_row(negative_coupling{});
        }
        S.on_proc->idx1[i+1] = S.on_proc->nnz;
        S.off_proc->idx1[i+1] = S.off_proc->nnz;
    }
} // hybrid_strength

template<strength_norm snorm, class T, is_bsr_or_csr<T> = true>
void norm_strength(T & A, ParCSRMatrix & S,
                   double theta, int num_variables, int *variables, int *off_variables) {
	using seq_t = sequential_matrix_t<T>;
	auto & diag = dynamic_cast<seq_t&>(*A.on_proc);
	auto & offd = dynamic_cast<seq_t&>(*A.off_proc);

	using P = norm_coupling<snorm>;
	for (int i = 0; i < A.local_num_rows; i++)
    {
	    auto row_start_on = A.on_proc->idx1[i];
	    auto row_end_on = A.on_proc->idx1[i+1];
	    auto row_start_off = A.off_proc->idx1[i];
	    auto row_end_off = A.off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
	        bool has_zero_diag = true;
	        if (A.on_proc->idx2[row_start_on] == i) {
                row_start_on++;
                has_zero_diag = false;
	        }

            auto row_var = (num_variables > 1) ? variables[i] : -1;
            // Find value with max magnitude in row
            auto row_scale = strongest_connection<P, seq_t>(i, row_var, num_variables,
                                                            {diag, variables, row_start_on},
                                                            {offd, off_variables, row_start_off});

            // Multiply row max magnitude by theta
            auto threshold = row_scale * theta;

            // Always add diagonal
            S.on_proc->idx2[S.on_proc->nnz] = i;
            if constexpr (is_bsr_v<T>)
	            S.on_proc->vals[S.on_proc->nnz] = has_zero_diag ? 0 :
		            value<P>(diag, row_start_on - 1);
            else
	            S.on_proc->vals[S.on_proc->nnz] = has_zero_diag ? 0 : A.on_proc->vals[row_start_on - 1];
            S.on_proc->nnz++;

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to
            // row_max * theta
            add_strong_connections<P, seq_t>(i, row_var, num_variables, threshold,
                                             {{diag, variables, row_start_on}, *S.on_proc},
                                             {{offd, off_variables, row_start_off}, *S.off_proc});
        }
        S.on_proc->idx1[i+1] = S.on_proc->nnz;
        S.off_proc->idx1[i+1] = S.off_proc->nnz;
    }
} // norm_strength
} // namespace
} // namespace raptor::classical

ParCSRMatrix* classical_strength(ParCSRMatrix* A, double theta, bool tap_amg, int num_variables,
        int* variables)
{
    CommPkg* comm = A->comm;
    if (tap_amg)
    {
        comm = A->tap_comm;
    }

    ParCSRMatrix* S = new ParCSRMatrix(A->partition, A->global_num_rows, A->global_num_cols,
            A->local_num_rows, A->on_proc_num_cols, A->off_proc_num_cols);

    int* off_variables = NULL;
    if (num_variables > 1)
    {
        std::vector<int>& recvbuf = comm->communicate(variables);
        off_variables = recvbuf.data();
    }

    // A and S will be sorted
    A->sort();
    A->on_proc->move_diag();

    classical::init_strength(*A->on_proc, *S->on_proc);
    classical::init_strength(*A->off_proc, *S->off_proc);
    auto bsr = dynamic_cast<ParBSRMatrix*>(A);
    if (!bsr) {
	    classical::hybrid_strength(*A, *S, theta, num_variables, variables, off_variables);
    } else {
	    classical::norm_strength<strength_norm::abs>(*bsr, *S, theta, num_variables,
	                                                 variables, off_variables);
    }

    classical::finalize_strength(*A, *S);

    return S;
}

// TODO -- currently this assumes all diags are same sign...
ParCSRMatrix* symmetric_strength(ParCSRMatrix* A, double theta, bool tap_amg)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val;
    double row_scale;
    double threshold;
    double diag;

    CommPkg* comm = A->comm;
    if (tap_amg)
    {
        comm = A->tap_comm;
    }

    std::vector<int> neg_diags;
    std::vector<double> row_scales;
    if (A->local_num_rows)
    {
        row_scales.resize(A->local_num_rows, 0);
        neg_diags.resize(A->local_num_rows);
    }

    ParCSRMatrix* S = new ParCSRMatrix(A->partition, A->global_num_rows, A->global_num_cols,
            A->local_num_rows, A->on_proc_num_cols, A->off_proc_num_cols);

    A->sort();
    A->on_proc->move_diag();

    if (A->on_proc->nnz)
    {
        S->on_proc->idx2.resize(A->on_proc->nnz);
        S->on_proc->vals.resize(A->on_proc->nnz);
        S->on_proc->nnz = 0;
    }
    if (A->off_proc->nnz)
    {
        S->off_proc->idx2.resize(A->off_proc->nnz);
        S->off_proc->vals.resize(A->off_proc->nnz);
        S->off_proc->nnz = 0;
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_start_on = A->on_proc->idx1[i];
        row_end_on = A->on_proc->idx1[i+1];
        row_start_off = A->off_proc->idx1[i];
        row_end_off = A->off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
            if (A->on_proc->idx2[row_start_on] == i)
            {
                diag = A->on_proc->vals[row_start_on];
                row_start_on++;
            }
            else
            {
                diag = 0.0;
            }

            // Find value with max magnitude in row
            if (diag < 0.0)
            {
                neg_diags[i] = 1;
                row_scale = -RAND_MAX;
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = A->on_proc->vals[j];
                    if (val > row_scale)
                    {
                        row_scale = val;
                    }
                }
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = A->off_proc->vals[j];
                    if (val > row_scale)
                    {
                        row_scale = val;
                    }
                }
            }
            else
            {
                neg_diags[i] = 0;
                row_scale = RAND_MAX;
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = A->on_proc->vals[j];
                    if (val < row_scale)
                    {
                        row_scale = val;
                    }
                }
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = A->off_proc->vals[j];
                    if (val < row_scale)
                    {
                        row_scale = val;
                    }
                }
            }

            // Multiply row max magnitude by theta
            row_scales[i] = row_scale * theta;
        }
    }

    std::vector<double>& off_proc_row_scales = comm->communicate(row_scales);
    std::vector<int>& off_proc_neg_diags = comm->communicate(neg_diags);

    S->on_proc->idx1[0] = 0;
    S->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_start_on = A->on_proc->idx1[i];
        row_end_on = A->on_proc->idx1[i+1];
        row_start_off = A->off_proc->idx1[i];
        row_end_off = A->off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
            bool neg_diag = neg_diags[i];
            threshold = row_scales[i];

            // Always add diagonal
            S->on_proc->idx2[S->on_proc->nnz] = i;
            S->on_proc->vals[S->on_proc->nnz] = A->on_proc->vals[row_start_on++];
            S->on_proc->nnz++;

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to
            // row_max * theta
            for (int j = row_start_on; j < row_end_on; j++)
            {
                val = A->on_proc->vals[j];
                col = A->on_proc->idx2[j];
                if ((neg_diag && val > threshold) || (!neg_diag && val < threshold)
                        || (neg_diags[col] && val > row_scales[col])
                        || (!neg_diags[col] && val < row_scales[col]))
                {
                    S->on_proc->idx2[S->on_proc->nnz] = col;
                    S->on_proc->vals[S->on_proc->nnz] = val;
                    S->on_proc->nnz++;
                }
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                val = A->off_proc->vals[j];
                col = A->off_proc->idx2[j];
                if ((neg_diag && val > threshold) || (!neg_diag && val < threshold)
                        || (off_proc_neg_diags[col] && val > off_proc_row_scales[col])
                        || (!off_proc_neg_diags[col] && val < off_proc_row_scales[col]))
                {
                    S->off_proc->idx2[S->off_proc->nnz] = col;
                    S->off_proc->vals[S->off_proc->nnz] = val;
                    S->off_proc->nnz++;
                }
            }
        }
        S->on_proc->idx1[i+1] = S->on_proc->nnz;
        S->off_proc->idx1[i+1] = S->off_proc->nnz;
    }
    S->on_proc->idx2.resize(S->on_proc->nnz);
    S->on_proc->idx2.shrink_to_fit();
    S->off_proc->idx2.resize(S->off_proc->nnz);
    S->off_proc->idx2.shrink_to_fit();

    S->on_proc->vals.resize(S->on_proc->nnz);
    S->on_proc->vals.shrink_to_fit();
    S->off_proc->vals.resize(S->off_proc->nnz);
    S->off_proc->vals.shrink_to_fit();

    S->local_nnz = S->on_proc->nnz + S->off_proc->nnz;

    S->on_proc_column_map = A->get_on_proc_column_map();
    S->local_row_map = A->get_local_row_map();
    S->off_proc_column_map = A->get_off_proc_column_map();

    S->comm = A->comm;
    S->tap_comm = A->tap_comm;
    S->tap_mat_comm = A->tap_mat_comm;

    if (S->comm) S->comm->num_shared++;
    if (S->tap_comm) S->tap_comm->num_shared++;
    if (S->tap_mat_comm) S->tap_mat_comm->num_shared++;

    return S;
}


// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
ParCSRMatrix* ParCSRMatrix::strength(strength_t strength_type,
        double theta, bool tap_amg, int num_variables, int* variables)
{
    switch (strength_type)
    {
        case Classical:
            return classical_strength(this, theta, tap_amg, num_variables, variables);
        case Symmetric:
            return symmetric_strength(this, theta, tap_amg);
        default :
            return NULL;
    }

    return NULL;
}
