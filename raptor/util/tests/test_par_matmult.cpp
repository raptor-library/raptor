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
	int grid[] = {10, 10, 10};
	double * stencil = laplace_stencil_27pt();
	ParCSRMatrix * A = par_stencil_grid(stencil, grid, 3);
	ParCSRMatrix * B = par_stencil_grid(stencil, grid, 3);

	auto C = A->mult(B);

	auto Ab = A->to_ParBSR(10, 10);
	auto Bb = B->to_ParBSR(10, 10);
	auto Cb = Ab->mult(Bb);
	auto C_bsr = Cb->to_ParCSR();
	compare(C, C_bsr);
}
