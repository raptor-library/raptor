#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/util/writer.hpp"
#include "raptor/tests/par_compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
}


void comparse_single_block(const ParCSRMatrix & A, const ParBSRMatrix & B) {
}

void handler(int signum, siginfo_t *si, void *vcontext) {
	std::cout << "SEGFAULT" << std::endl;
}


TEST(ParBSRMatrixMult, TestsInUtil)
{
	int num_procs;
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	int grid[] = {10 * num_procs, 2, 5};
	double * stencil = laplace_stencil_27pt();
	ParCSRMatrix * A = par_stencil_grid(stencil, grid, 3);
	ParCSRMatrix * B = par_stencil_grid(stencil, grid, 3);

	auto C = A->mult(B);

	auto Ab = A->to_ParBSR(10, 10);
	auto Bb = B->to_ParBSR(10, 10);
	auto Cb = Ab->mult(Bb);
	auto C_bsr = Cb->to_ParCSR();
	compare(C, C_bsr);

	delete C_bsr;
	delete Cb;
	delete Bb;
	delete Ab;
	delete C;
	delete B;
	delete A;
	delete[] stencil;
}

TEST(ParBSRMatrixMultT, TestsInUtil)
{
	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	if (num_procs < 2)
	{
		GTEST_SKIP() << "at least two MPI ranks required";
	}

	// every rank owns two block rows
	const int local_block_rows = 2;
	const int global_block_rows = local_block_rows * num_procs;
	const int global_block_cols = num_procs;
	ParBSRMatrix* A_bsr = new ParBSRMatrix(
			global_block_rows, global_block_cols, 2, 4);
	ParBSRMatrix* B_bsr = new ParBSRMatrix(
			global_block_rows, global_block_cols, 2, 4);

	BSRMatrix* A_on = static_cast<BSRMatrix*>(A_bsr->on_proc);
	BSRMatrix* A_off = static_cast<BSRMatrix*>(A_bsr->off_proc);
	BSRMatrix* B_on = static_cast<BSRMatrix*>(B_bsr->on_proc);
	BSRMatrix* B_off = static_cast<BSRMatrix*>(B_bsr->off_proc);
	double A_diag[] = {2, 0, 1, 0, 0, 2, 0, 1};
	double A_next[] = {-1, 0, 0, 0, 0, -1, 0, 0};
	double B_diag[] = {3, 0, 0, 1, 0, 3, 1, 0};
	double B_next[] = {-1, 0, 0, 0, 0, -1, 0, 0};

	for (int i = 0; i < local_block_rows; i++)
	{
		A_on->add_value(i, 0, A_diag);
		B_on->add_value(i, 0, B_diag);

		const int next_col = rank + 1;
		if (next_col < global_block_cols)
		{
			A_off->add_value(i, next_col, A_next);
			B_off->add_value(i, next_col, B_next);
		}
		A_on->idx1[i + 1] = A_on->idx2.size();
		A_off->idx1[i + 1] = A_off->idx2.size();
		B_on->idx1[i + 1] = B_on->idx2.size();
		B_off->idx1[i + 1] = B_off->idx2.size();
	}
	A_bsr->finalize();
	B_bsr->finalize();

	ParCSRMatrix* A = A_bsr->to_ParCSR();
	ParCSRMatrix* B = B_bsr->to_ParCSR();
	ParCSRMatrix* C = A->mult_T(B);

	ParBSRMatrix* C_bsr = A_bsr->mult_T(B_bsr);

	// true B^T A
	// [diag | upper             ]
	// [lower| diag+previous_diag]
	// 2B^T_diag A_diag
	const double diag[] = {
		12,  0, 6, 0,
		 0, 12, 0, 6,
		 0,  4, 0, 2,
		 4,  0, 2, 0
	};
	// 2B^T_next A_next
	const double previous_diag[] = {
		2, 0, 0, 0,
		0, 2, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	// 2B^T_next A_diag
	const double lower[] = {
		-4,  0, -2,  0,
		 0, -4,  0, -2,
		 0,  0,  0,  0,
		 0,  0,  0,  0
	};
	// 2B^T_diag A_next
	const double upper[] = {
		-6,  0, 0, 0,
		 0, -6, 0, 0,
		 0, -2, 0, 0,
		-2,  0, 0, 0
	};

	ASSERT_EQ(C_bsr->local_num_rows, 1);
	const int global_row = C_bsr->local_row_map[0];
	std::vector<double> expected(global_block_cols * 16, 0.0);
	std::copy(diag, diag + 16, expected.begin() + global_row * 16);
	if (global_row > 0)
	{
		for (int i = 0; i < 16; i++)
		{
			expected[global_row * 16 + i] += previous_diag[i];
			expected[(global_row - 1) * 16 + i] = lower[i];
		}
	}
	if (global_row + 1 < global_block_cols)
	{
		std::copy(upper, upper + 16,
				expected.begin() + (global_row + 1) * 16);
	}

	BSRMatrix* C_on = static_cast<BSRMatrix*>(C_bsr->on_proc);
	BSRMatrix* C_off = static_cast<BSRMatrix*>(C_bsr->off_proc);
	const int expected_blocks = 1 + (global_row > 0)
			+ (global_row + 1 < global_block_cols);
	ASSERT_EQ(C_on->idx2.size() + C_off->idx2.size(), expected_blocks);
	ASSERT_EQ(C_on->nnz, C_on->idx2.size());
	ASSERT_EQ(C_off->nnz, C_off->idx2.size());

	std::vector<double> actual(global_block_cols * 16, 0.0);
	for (int j = C_on->idx1[0]; j < C_on->idx1[1]; j++)
	{
		const int global_col = C_bsr->on_proc_column_map[C_on->idx2[j]];
		std::copy(C_on->block_vals[j], C_on->block_vals[j] + 16,
				actual.begin() + global_col * 16);
	}
	for (int j = C_off->idx1[0]; j < C_off->idx1[1]; j++)
	{
		const int global_col = C_bsr->off_proc_column_map[C_off->idx2[j]];
		std::copy(C_off->block_vals[j], C_off->block_vals[j] + 16,
				actual.begin() + global_col * 16);
	}

	// bsr vs actual
	for (int i = 0; i < global_block_cols * 16; i++)
	{
		ASSERT_NEAR(actual[i], expected[i], 1e-10);
	}

	ParCSRMatrix* C_from_bsr = C_bsr->to_ParCSR();

	// csr vs bsr result
	compare(C, C_from_bsr);

	delete C_from_bsr;
	delete C_bsr;
	delete B_bsr;
	delete A_bsr;
	delete C;
	delete B;
	delete A;
}
