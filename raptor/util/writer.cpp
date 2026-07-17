#include <array>
#include <sstream>
#include <fstream>

#include "raptor/core/matrix_traits.hpp"
#include "raptor/util/writer.hpp"

namespace raptor {

namespace impl {

template <class T> struct mat_type;
template <> struct mat_type<raptor::ParCSRMatrix> : std::integral_constant<int, 0> {};
template <> struct mat_type<raptor::ParBSRMatrix> : std::integral_constant<int, 1> {};
template <> struct mat_type<raptor::BSRMatrix> : std::integral_constant<int, 2> {};
template <> struct mat_type<raptor::CSRMatrix> : std::integral_constant<int, 3> {};


void write_header(const char *fname, const raptor::ParCSRMatrix & mat) {
	std::ofstream ofile(std::string(fname) + ".hdr", std::ios_base::binary);

	int nprocs; MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	std::array<int, 4> buf{
		mat_type<raptor::ParCSRMatrix>::value,
		mat.global_num_rows,
		mat.global_num_cols,
		nprocs};

	ofile.write(reinterpret_cast<const char*>(buf.data()), sizeof(int)*buf.size());
}


void write_header(const char *fname, const raptor::ParBSRMatrix & mat) {
	std::ofstream ofile(std::string(fname) + ".hdr", std::ios_base::binary);

	int nprocs; MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	auto & diag = dynamic_cast<raptor::BSRMatrix&>(*mat.on_proc);
	std::array<int, 6> buf{
		mat_type<raptor::ParBSRMatrix>::value,
		mat.global_num_rows,
		mat.global_num_cols,
		nprocs,
		diag.b_rows,
		diag.b_cols};

	ofile.write(reinterpret_cast<const char*>(buf.data()), sizeof(int)*buf.size());
}


void write_header(const char *fname, const raptor::BSRMatrix & mat) {
	std::ofstream ofile(std::string(fname) + ".hdr", std::ios_base::binary);
	std::array<int, 5> buf{
		mat_type<raptor::BSRMatrix>::value,
		mat.n_rows,
		mat.n_cols,
		mat.b_rows,
		mat.b_cols};

	ofile.write(reinterpret_cast<const char*>(buf.data()), sizeof(int)*buf.size());
}


void write_header(const char *fname, const raptor::CSRMatrix & mat) {
	std::ofstream ofile(std::string(fname) + ".hdr", std::ios_base::binary);
	std::array<int, 3> buf{
		mat_type<raptor::CSRMatrix>::value,
		mat.n_rows,
		mat.n_cols};

	ofile.write(reinterpret_cast<const char*>(buf.data()), sizeof(int)*buf.size());
}


void write_rowptr(std::ostream & out,
                  const raptor::Matrix & diag,
                  const raptor::Matrix & offd) {
	for (int i = 0; i < diag.n_rows + 1; ++i) {
		int ptr = diag.idx1[i] + offd.idx1[i];
		out.write(reinterpret_cast<const char *>(&ptr), sizeof(int));
	}
}


template<class T>
void write_rows(std::ostream & out,
                const T & diag, const T & offd,
                const std::vector<int> & diag_colmap, const std::vector<int> & offd_colmap) {
	out.write(reinterpret_cast<const char*>(&diag.n_rows), sizeof(diag.n_rows));
	write_rowptr(out, diag, offd);

	for (int i = 0; i < diag.n_rows; ++i) {
		auto write_colinds = [&](const T & mat, const std::vector<int> & colmap) {
			for (int j = mat.idx1[i]; j < mat.idx1[i + 1]; ++j) {
				int gcol = colmap[mat.idx2[j]];
				out.write(reinterpret_cast<const char *>(&gcol), sizeof(int));
			}
		};

		write_colinds(diag, diag_colmap);
		write_colinds(offd, offd_colmap);
	}
	for (int i = 0; i < diag.n_rows; ++i) {
		auto write_values = [&](const T & mat) {
			if constexpr (is_bsr_v<T>) {
				for (int j = mat.idx1[i]; j < mat.idx1[i + 1]; ++j) {
					out.write(reinterpret_cast<const char *>(mat.block_vals[j]),
					          mat.b_size * sizeof(double));
				}
			} else {
				out.write(reinterpret_cast<const char *>(&mat.vals[mat.idx1[i]]),
				          (mat.idx1[i + 1] - mat.idx1[i]) * sizeof(double));
			}
		};

		write_values(diag);
		write_values(offd);
	}
}


std::ofstream get_rank_file(const char * fname) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::ostringstream rank_file;
	rank_file << fname << '.' << rank;
	return std::ofstream(rank_file.str(), std::ios_base::binary);
}


template <class T, is_bsr_or_csr<T> = true>
void write(const char * fname, const T & mat) {
	write_header(fname, mat);

	auto cast = [](const auto & m) -> const auto & {
		return dynamic_cast<const sequential_matrix_t<T> &>(m);
	};

	auto ofile = get_rank_file(fname);
	write_rows(ofile, cast(*mat.on_proc), cast(*mat.off_proc),
	           mat.on_proc_column_map, mat.off_proc_column_map);
}


template<class T, is_seq_bsr_or_csr<T> = true>
void write(const char * fname, const T & mat) {
	write_header(fname, mat);

	std::ofstream ofile(fname, std::ios_base::binary);
	ofile.write(reinterpret_cast<const char*>(&mat.n_rows), sizeof(mat.n_rows));
	ofile.write(reinterpret_cast<const char *>(mat.idx1.data()), mat.idx1.size() * sizeof(int));
	ofile.write(reinterpret_cast<const char*>(mat.idx2.data()), mat.idx2.size() * sizeof(int));
	if constexpr (is_bsr_v<T>) {
		for (int i = 0; i < mat.n_rows; ++i) {
			for (int j = mat.idx1[i]; j < mat.idx1[i + 1]; ++j) {
				ofile.write(reinterpret_cast<const char *>(mat.block_vals[j]),
				            mat.b_size * sizeof(double));
			}
		}
	} else {
		ofile.write(reinterpret_cast<const char*>(mat.vals.data()), mat.vals.size() * sizeof(double));
	}
}
} // namespace impl

void write(const char *fname, const ParCSRMatrix &mat) {
	impl::write(fname, mat);
}

void write(const char *fname, const ParBSRMatrix & mat) {
	impl::write(fname, mat);
}

void write(const char *fname, const BSRMatrix &mat) { impl::write(fname, mat); }
void write(const char *fname, const CSRMatrix &mat) { impl::write(fname, mat); }



} // namespace raptor
