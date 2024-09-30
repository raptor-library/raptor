#include <array>
#include <sstream>
#include <fstream>

#include "raptor/core/matrix_traits.hpp"
#include "raptor/util/writer.hpp"

namespace raptor {

namespace impl {
void write_header(const char *fname, const raptor::ParCSRMatrix & mat) {
	std::ofstream ofile(std::string(fname) + ".hdr", std::ios_base::binary);

	int nprocs; MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	std::array<int, 4> buf{
		0, // csr
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
		1, // bsr
		mat.global_num_rows,
		mat.global_num_cols,
		nprocs,
		diag.b_rows,
		diag.b_cols};

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
} // namespace impl

void write(const char *fname, const ParCSRMatrix &mat) {
	impl::write(fname, mat);
}

void write(const char *fname, const ParBSRMatrix & mat) {
	impl::write(fname, mat);
}

} // namespace raptor
