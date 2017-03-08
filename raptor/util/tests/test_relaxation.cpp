#include <assert.h>
#include <math.h>
#include <core/types.hpp>
#include <core/matrix.hpp>

using namespace raptor;

int main(int argc, char* argv[])
{
    int n_rows = 4;
    int n_cols = 4;
    double data[16] = {2.0, -1.0,  0.0,  0.0, -1.0,  2.0, -1.0,  0.0,
             0.0, -1.0,  2.0, -1.0, 0.0,  0.0, -1.0,  2.0};

    CSRMatrix A(n_rows, n_cols, data);
    A.sort();
    A.print();

    Vector x(A.n_rows);
    Vector b(A.n_rows);
    Vector r(A.n_rows);
    double r_norm = 0;
    double prev_r_norm = 0;
    
    x.set_const_value(1.0);
    b.set_rand_values();
    A.residual(x, b, r);
    prev_r_norm = r.norm(2);
    printf("INIT RNorm = %e\n", prev_r_norm);
    for (int i = 0; i < 100; i++)
    {
        A.jacobi(x, b, r, 1.0);
        A.residual(x, b, r);
        r_norm = r.norm(2);
        printf("Jacobi[%d]: RNorm = %e\n", i, r_norm);
        if (r_norm < 1e-08) break;
        prev_r_norm = r_norm;
    }

    x.set_const_value(1.0);
    b.set_rand_values();
    A.residual(x, b, r);
    prev_r_norm = r.norm(2);
    printf("INIT RNorm = %e\n", prev_r_norm);
    for (int i = 0; i < 100; i++)
    {
        A.gauss_seidel(x, b);
        A.residual(x, b, r);
        r_norm = r.norm(2);
        printf("GaussSeidel[%d]: RNorm = %e\n", i, r_norm);
        if (r_norm < 1e-08) break;
        assert(prev_r_norm >= r_norm);
        prev_r_norm = r_norm;
    }

    x.set_const_value(1.0);
    b.set_rand_values();
    A.residual(x, b, r);
    prev_r_norm = r.norm(2);
    printf("INIT RNorm = %e\n", prev_r_norm);
    for (int i = 0; i < 100; i++)
    {
        A.SOR(x, b);
        A.residual(x, b, r);
        r_norm = r.norm(2);
        printf("SOR[%d]: RNorm = %e\n", i, r_norm);
        if (r_norm < 1e-08) break;
        assert(r_norm <= prev_r_norm);
        prev_r_norm = r_norm;
    }

}
