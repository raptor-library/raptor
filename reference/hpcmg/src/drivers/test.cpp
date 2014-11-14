#include "linalg.h"
#include "sys.h"

int main()
{
    using namespace linalg::seq;
    CSRMatrix mat(4,4,8);
    Vector &v = mat.data();
    v.set(4);
    const Vector &vec = mat.data();
    std::cout << vec(0) << std::endl;

    std::cout << v << std::endl;
    return 0;
}
