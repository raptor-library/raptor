#include "gtest/gtest.h"

#include "raptor/raptor.hpp"
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
#endif


TEST(TestLocalAIR, TestsInRuge_Stuben) {
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	constexpr std::size_t n{16};
	// constexpr std::size_t n{5};
	std::vector<int> grid;

	grid.resize(1, n);

	std::vector<double> stencil{{-1., 2, -1}};

	auto A = par_stencil_grid(stencil.data(), grid.data(), 1);
	// {
	// 	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// 	if (rank == 1) {
	// 		auto & offd = *A->off_proc;
	// 		auto [rowptr, colind, values] = offd.vecs();
	// 		for (int r = 0; r < A->local_num_rows; ++r) {
	// 			for (int off = rowptr[r]; off < rowptr[r+1]; ++off) {
	// 				auto & gcol = A->off_proc_column_map[colind[off]];
	// 				std::cout << "here: " << r << " " << A->off_proc_column_map[colind[off]] << std::endl;
	// 			}
	// 		}
	// 	}
	// }
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

	auto R = local_air_interpolation(*A, *S, split);

}
