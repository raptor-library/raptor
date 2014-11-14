#include <cstdint>
#include "linalg/par/vector.h"
#include "linalg/seq/vector.h"
#include <gtest/gtest.h>
#include <mpi.h>

namespace {
    class ParVectorTest : public ::testing::Test {
        protected:
            ParVectorTest() {
                using namespace linalg::par;
                v0 = new Vector(MPI_COMM_WORLD, 10);
                v0->set(2);
            }

            virtual ~ParVectorTest() {
                delete v0;
            }

            virtual void SetUp() {
            }

            virtual void TearDown() {
            }

            linalg::par::Vector* v0;
    };

    TEST_F(ParVectorTest, Scale) {
        using namespace linalg::par;
        v0->scale(3);
        const linalg::seq::Vector &lvec = v0->local();
        for (int i=0;i<lvec.size();i++){
            EXPECT_EQ(6, lvec(i));
        }

    }
}  // namespace
