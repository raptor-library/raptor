#include "gtest/gtest.h"

#include "raptor/raptor.hpp"
#include "raptor/ruge_stuben/par_air_solver.hpp"
#include "raptor/tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}

#if 0
TEST(TestOnePointInterp, TestsInRuge_Stuben) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	constexpr std::size_t n{16};
	std::vector<int> grid;

	grid.resize(1, n);

	std::vector<double> stencil{{-1., 2, -1}};

	auto A = par_stencil_grid(stencil.data(), grid.data(), 1);
	auto get_split = [](const Matrix &, const std::vector<int> & colmap) {
		std::vector<int> split(colmap.size(), 0);

		std::size_t i{0};
		for (auto col : colmap) split[i++] = (col % 2 == 0) ? Selected : Unselected;

		return split;
	};
	splitting_t split{get_split(*A->on_proc, A->on_proc_column_map),
	                  get_split(*A->off_proc, A->off_proc_column_map)};

	auto S = A->copy();

	{ // delete a connection to force interpolating from off_proc
		if (rank == 1) {
			auto & diag = *S->on_proc;
			auto & colmap = S->on_proc_column_map;
			auto & rowmap = S->local_row_map;
			int r = diag.idx1.size() - 2;
			for (int off = diag.idx1[r]; off < diag.idx1[r+1]; ++off) {
				if (colmap[diag.idx2[off]] == rowmap[r] - 1) {
					diag.vals[off] = 0;
				}
			}
		}
	}
	auto P = one_point_interpolation(*A, *S, split);

	auto & diag = *P->on_proc;
	auto & offd = *P->off_proc;
	auto & rowmap = P->local_row_map;
	EXPECT_EQ(diag.n_rows, 4);
	if (rank == 1) {
		EXPECT_EQ(P->off_proc_num_cols, 1);
	} else {
		EXPECT_EQ(P->off_proc_num_cols, 0);
	}
	for (int i = 0; i < P->local_num_rows; ++i) {
		auto r = rowmap[i];
		for (int off = diag.idx1[i]; off < diag.idx1[i+1]; ++off) {
			auto c = P->on_proc_column_map[diag.idx2[off]];
			if (r % 2 == 0) {
				ASSERT_EQ(c, r);
				ASSERT_EQ(diag.vals[off], 1);
			} else {
				if (r == 7) {
					ASSERT_EQ(c, r+1);
				} else {
					ASSERT_EQ(c, r-1);
				}
				ASSERT_EQ(diag.vals[off], 1);
			}
		}
		for (int off = offd.idx1[i]; off < offd.idx1[i+1]; ++off) {
			ASSERT_EQ(rank, 1);
			ASSERT_EQ(r, 7);
			auto c = P->off_proc_column_map[offd.idx2[off]];
			ASSERT_EQ(c, 8);
		}
	}
}

TEST(TestLocalAIR, TestsInRuge_Stuben) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	constexpr std::size_t n{17};
	std::vector<int> grid;

	grid.resize(1, n);

	std::vector<double> stencil{{-1., 2, -1}};

	auto A = par_stencil_grid(stencil.data(), grid.data(), 1);
	auto get_split = [](const Matrix &, const std::vector<int> & colmap) {
		std::vector<int> split(colmap.size(), 0);

		std::size_t i{0};
		for (auto col : colmap) split[i++] = (col % 2 == 0) ? Selected : Unselected;

		return split;
	};
	splitting_t split{get_split(*A->on_proc, A->on_proc_column_map),
	                  get_split(*A->off_proc, A->off_proc_column_map)};
	auto S = A->copy();

	auto R = local_air(*A, *S, split);

	using expect_t = std::map<int, double>;
	auto get_expected = [](int row) -> expect_t {
		if (row == 0)
			return {{0, 1}, {1, 0.5}};
		else if (row == 16)
			return {{16, 1}, {15, 0.5}};
		else
			return {
				{row - 1, 0.5},
				{row, 1},
				{row + 1, 0.5}};
	};
	auto rowmap = [first_row = A->partition->first_local_col](int i) {
		auto first_selected = (first_row % 2 == 0) ? first_row : first_row + 1;
		return (first_selected / 2 + i) * 2;
	};
	for (int i = 0; i < R->on_proc->n_rows; ++i) {
		auto row = rowmap(i);
		auto expected = get_expected(row);
		expect_t cols;
		auto add_cols = [&](const Matrix & mat, const auto & colmap) {
			for (std::size_t off = mat.idx1[i]; off < mat.idx1[i + 1]; ++off) {
				cols[colmap[mat.idx2[off]]] = mat.vals[off];
			}
		};
		add_cols(*R->on_proc, R->on_proc_column_map);
		add_cols(*R->off_proc, R->off_proc_column_map);
		ASSERT_EQ(expected, cols);
	}
}

TEST(TestLocalAIRBSR, TestsInRuge_Stuben) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	constexpr std::size_t n{16};
	// constexpr std::size_t n{5};
	std::vector<int> grid;

	grid.resize(1, n);

	std::vector<double> stencil{{-1., 2, -1}};

	auto A = par_stencil_grid(stencil.data(), grid.data(), 1);
	auto Absr = new raptor::ParBSRMatrix(A->partition, 3, 3);
	[](auto & ... mats) {
		((mats.idx1[0] = 0),...);
	}(*Absr->on_proc, *Absr->off_proc);
	for (int i = 0; i < A->local_num_rows; ++i) {
		auto copy = [=](const Matrix & src, Matrix & dst, auto && colmap) {
			auto & bsr = dynamic_cast<BSRMatrix &>(dst);
			for (int j = src.idx1[i]; j < src.idx1[i+1]; ++j) {
				dst.idx2.push_back(colmap(src.idx2[j]));
				bsr.block_vals.push_back(new double[bsr.b_size]());
				for (int k = 0; k < bsr.b_size; ++k)
					bsr.block_vals.back()[k] = src.vals[j];
			}
			dst.idx1[i+1] = dst.idx2.size();
		};
		copy(*A->on_proc, *Absr->on_proc, [](auto c) { return c; });
		copy(*A->off_proc, *Absr->off_proc, [&](auto c) { return A->off_proc_column_map[c]; });
	}
	[](auto & ... mats) {
		((mats.nnz = mats.idx2.size()),...);
	}(*Absr->on_proc, *Absr->off_proc);
	Absr->finalize();

	auto get_split = [](const Matrix &, const std::vector<int> & colmap) {
		std::vector<int> split(colmap.size(), 0);

		std::size_t i{0};
		for (auto col : colmap) split[i++] = (col % 2 == 0) ? Selected : Unselected;

		return split;
	};
	splitting_t split{get_split(*A->on_proc, A->on_proc_column_map),
	                  get_split(*A->off_proc, A->off_proc_column_map)};

	{
		int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (rank == 1)
			split.on_proc[2] = Unselected;
	}

	auto S = A->copy();

	auto R = local_air(*Absr, *S, split);
}
#endif

void dump(const ParCSRMatrix & A) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::ofstream ofile("coarse-" + std::to_string(rank));
	auto write_row = [&](int i, const Matrix & mat, const std::vector<int> & colmap) {
		const auto & rowmap = A.local_row_map;
		for (int j = mat.idx1[i]; j < mat.idx1[i+1]; ++j) {
			ofile << rowmap[i] << " " << colmap[mat.idx2[j]] << " " << std::scientific << mat.vals[j] << '\n';

		}
	};
	for (int i = 0; i < A.local_num_rows; ++i) {
		write_row(i, *A.on_proc, A.on_proc_column_map);
		write_row(i, *A.off_proc, A.off_proc_column_map);
	}
}

ParCSRMatrix * gen(std::size_t n) {
	auto A = new ParCSRMatrix(n, n);

	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size; MPI_Comm_size(MPI_COMM_WORLD, &size);

	auto num_local = A->partition->local_num_rows;
	auto row_start = A->partition->first_local_row;
	auto init = [&](Matrix & mat, int ncols) {
		mat.n_rows = num_local;
		mat.n_cols = ncols;
		mat.nnz = 0;
		mat.idx1.resize(num_local+1);
		mat.idx2.reserve(num_local*2);
		mat.vals.reserve(.3*num_local*2);
	};
	init(*A->on_proc, num_local);
	init(*A->off_proc, n);

	A->on_proc->idx1[0] = 0;
	A->off_proc->idx1[0] = 0;
	for (int i = 0; i < num_local; ++i) {
		A->add_value(i, i + row_start, 1.);
		if (!(rank == 0 && i == 0))
			A->add_value(i, i + row_start - 1, -1.);

		auto set_rowptr = [i](Matrix & mat) {
			mat.idx1[i+1] = mat.idx2.size();
		};

		set_rowptr(*A->on_proc);
		set_rowptr(*A->off_proc);
	}

	A->on_proc->nnz = A->on_proc->idx2.size();
	A->off_proc->nnz = A->off_proc->idx2.size();

	A->finalize();

	return A;
}
TEST(UpwindAdvection, TestsInRuge_Stuben) {
	// constexpr std::size_t n{100};
	// std::vector<int> grid;
	// grid.resize(1, n);
	// std::vector<double> stencil{{-1, 1, 0}};
	// auto A = par_stencil_grid(stencil.data(), grid.data(), 1);
	auto A = gen(100);

	// dump(*A);
	ParAIRSolver ml;
	// ParRugeStubenSolver ml;
	ml.max_levels = 2;
	ml.setup(A);
	dump(*ml.levels[1]->A);
}
