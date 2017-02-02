#include <mpi.h>
#include <math.h>
#include <core/types.hpp>
#include <core/matrix.hpp>

using namespace raptor;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get Local Process Rank, Number of Processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0)
    {
        COOMatrix* A = new COOMatrix(25, 25, 5);
        A->add_value(0, 1, 1.0);
        A->add_value(21, 9, 2.0);
        A->add_value(1, 0, 1.0);
        A->add_value(5, 3, 2.0);
        A->add_value(0, 0, 1.0);
        A->add_value(7, 11, 1.0);
        A->add_value(0, 0, 0.5);
        A->add_value(21, 21, 4.0);
        A->add_value(11, 11, 2.0);

        printf("Original COO Matrix:\n");
        A->print();

        printf("\nCOO converted to CSR Matrix:\n");
        CSRMatrix* Acsr = new CSRMatrix(A);
        Acsr->print();

        printf("\nCOO converted to CSC Matrix:\n");
        CSCMatrix* Acsc = new CSCMatrix(A);
        Acsc->print();

        printf("\nCSR converted to CSC Matrix: \n");
        CSCMatrix* Acsr_csc = new CSCMatrix(Acsr);
        Acsr_csc->print();

        printf("\nCSC converted to CSR Matrix: \n");
        CSRMatrix* Acsc_csr = new CSRMatrix(Acsc);
        Acsc_csr->print();

        printf("\nSorted COO Matrix:\n");
        A->sort();
        A->print();

        printf("\nSorted CSR Matrix:\n");
        Acsr->sort();
        Acsr->print();

        printf("\nSorted CSC Matrix:\n");
        Acsc->sort();
        Acsc->print();

        delete Acsc_csr;
        delete Acsr_csc;

        printf("\nCreate X, B and Initialize X = 1.0\n");                
        Vector* x = new Vector(A->n_cols);
        Vector* b = new Vector(A->n_rows);
        x->set_const_value(1.0);
        
        printf("\nVector B = A(COO) * X\n");        
        A->mult(x, b);
        b->print("B");

        printf("\nVector B = A(CSR) * X\n");                
        b->set_const_value(1.0);
        Acsr->mult(x, b);
        b->print("B");

        printf("\nVector B = A(CSC) * X\n");        
        b->set_const_value(1.0);
        Acsc->mult(x, b);
        b->print("B");

        printf("\nChange a couple values in B\n");                
        b->data()[2] += 0.1;
        b->data()[4] += 0.1;
        Vector* r = new Vector(A->n_rows);
        
        printf("\nVector R = B - A(COO) * X\n");                
        A->residual(x, b, r);
        r->print("R");

        delete x;
        delete b;
        delete r;

        printf("\nCOO Matrix with Condensed Cols:\n");
        A->condense_cols();
        A->print();

        printf("\nCOO Matrix with Condensed Rows:\n");
        A->condense_rows();
        A->print();

        printf("\nCSR Matrix with Condensed Cols:\n");
        Acsr->condense_cols();
        Acsr->print();

        printf("\nCSR Matrix with Condensed Rows:\n");
        Acsr->condense_rows();
        Acsr->print();

        printf("\nCSC Matrix with Condensed Cols:\n");
        Acsc->condense_cols();
        Acsc->print();

        printf("\nCSC Matrix with Condensed Rowss:\n");
        Acsc->condense_rows();
        Acsc->print();

        delete Acsr;
        delete Acsc;
        delete A;

    }

    MPI_Finalize();
}
   
