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

    CSRMatrix S(n_rows, n_cols, n_rows);


    // Test Classical Strength of Connection for Various Theta Values
    printf("Classical Strength Matrix (Theta = 0.0)\n");
    A.classical_strength(&S, 0.0);
    S.print();

    printf("Classical Strength Matrix (Theta = 0.5)\n");
    A.classical_strength(&S, 0.5);
    S.print();

    printf("Classical Strength Matrix (Theta = 1.0)\n");
    A.classical_strength(&S, 1.0);
    S.print();


    // Test Symmetric Strength of Connection for Various Theta Values
    printf("Symmetric Strength Matrix (Theta = 0.0)\n");
    A.symmetric_strength(&S, 0.0);
    S.print();

    printf("Symmetric Strength Matrix (Theta = 0.5)\n");
    A.symmetric_strength(&S, 0.5);
    S.print();

    printf("Symmetric Strength Matrix (Theta = 1.0)\n");
    A.symmetric_strength(&S, 1.0);
    S.print();
}

