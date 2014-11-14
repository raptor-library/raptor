#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "linalg/seq/csrmatrix.h"
#include "linalg/seq/vector.h"

namespace {
    class CSRMatrixTest : public ::testing::Test {
        protected:
            CSRMatrixTest() {
                using namespace linalg::seq;
                int i;
                mat0 = new CSRMatrix(5, 5, 7);
                for (i=0;i<mat0->num_rows()+1;i++) {
                    if (i < 2)
                        mat0->rowptr()(i) = i*2;
                    else
                        mat0->rowptr()(i) = i+2;
                }
                for (i=0; i < mat0->nnz(); i++) {
                    if (i < 2)
                        mat0->colind()(i) = i;
                    else
                        mat0->colind()(i) = i-2;
                    mat0->data()(i) = (i+1)*2;
                }
            }

            virtual ~CSRMatrixTest() {
                delete mat0;
            }

            virtual void SetUp() {
            }

            virtual void TearDown() {
            }

            linalg::seq::CSRMatrix* mat0;
    };

    TEST_F(CSRMatrixTest, MatVec) {
        using namespace linalg::seq;
        Vector vec(mat0->num_rows());
        vec.set(2);
        Vector out(mat0->num_rows());
        mat0->mult(vec, &out);
        for (int i=0;i<out.size();i++) {
            if (i < 2)
                EXPECT_EQ(2*((2*i+1)*2 + (2*i+2)*2), out(i));
            else
                EXPECT_EQ(4*((i+1)+2), out(i));
        }
    }
}  // namespace
