#include <cstdint>
#include "linalg/seq/vector.h"
#include <gtest/gtest.h>

namespace {
    class SeqVectorTest : public ::testing::Test {
        protected:
            SeqVectorTest() {
                using namespace linalg::seq;
                v0 = new Vector(10);
                v0->set(2);
            }

            virtual ~SeqVectorTest() {
                delete v0;
            }

            virtual void SetUp() {
            }

            virtual void TearDown() {
            }

            linalg::seq::Vector* v0;
    };

    TEST_F(SeqVectorTest, Scale) {
        using namespace linalg::seq;
        v0->scale(3);
        for (int i=0;i<v0->size();i++){
            EXPECT_EQ(6, (*v0)(i));
        }

    }

    TEST_F(SeqVectorTest, Abs)
    {
        using namespace linalg::seq;
        v0->set(-3);
        if (v0->size() > 0) EXPECT_EQ(-3, (*v0)(0));
        v0->abs();
        for (int i=0;i<v0->size();i++){
            EXPECT_EQ(3, (*v0)(i));
        }
    }

    TEST_F(SeqVectorTest, Dot)
    {
        using namespace linalg::seq;
        Vector v1(v0->size());
        v1.set(3);
        sys::data_t val = v0->dot(&v1);
        EXPECT_EQ(2*3*v0->size(), val);
    }
}  // namespace
