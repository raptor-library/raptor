#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Get filename and symmetry for matrix to read in
    char *fname;
    if (argc == 2) {
       fname = argv[1];
    }
    else {
        printf("Input <matrix filename>\n");
        exit(-1);
    }

    // Needed for partial inner products
    int first_root = 0, second_root = 0, color = 0, part_global;
    MPI_Comm contig_comm = MPI_COMM_NULL;
    MPI_Comm striped_comm = MPI_COMM_NULL;

    // Setup problem to solve
    ParCSRMatrix* A = readParMatrix(fname);
    
    ParVector x_contig(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_striped(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_true(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    std::vector<double> residuals_true;
    std::vector<double> residuals_contig;
    std::vector<double> residuals_striped;

    x_true.set_const_value(1.0);
    A->mult(x_true, b);
    x_true.set_const_value(0.0);
    BiCGStab(A, x_true, b, residuals_true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test contiguous first half
    //create_partial_inner_comm(inner_comm, color, first_root, second_root, part_global, 0);
    x_contig.set_const_value(1.0);
    A->mult(x_contig, b);
    x_contig.set_const_value(0.0);
    PI_BiCGStab(A, x_contig, b, residuals_contig, contig_comm, color, first_root, second_root, part_global, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test striped even procs
    //create_partial_inner_comm(striped_comm, color, first_root, second_root, part_global, 1);
    x_striped.set_const_value(1.0);
    A->mult(x_striped, b);
    x_striped.set_const_value(0.0);
    PI_BiCGStab(A, x_striped, b, residuals_striped, striped_comm, color, first_root, second_root, part_global, 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // Write out residuals to file
    FILE *f;
    if (rank == 0) {
       f = fopen("PartInner_Contig_BiCGStab_Res.txt", "w");
       fprintf(f, "CFD %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_contig.size(); i++) {
           fprintf(f, "%lf \n", residuals_contig[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_Striped_BiCGStab_Res.txt", "w");
       fprintf(f, "CFD %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_striped.size(); i++) {
           fprintf(f, "%lf\n", residuals_striped[i]);
       }
       fclose(f);
       
       f = fopen("PartInner_True_BiCGStab_Res.txt", "w");
       fprintf(f, "CFD %d x %d\n", A->global_num_rows, A->global_num_cols);
       for (int i=0; i<residuals_true.size(); i++) {
           fprintf(f, "%lf\n", residuals_true[i]);
       }
       fclose(f);

    }

    // Write out solutions to file
    f = fopen("PartInner_Contig_BiCGStab_x.txt", "w");
    for (int i = 0; i < num_procs; i++) {
        if (rank == i) {
            for (int j = 0; j < x_contig.local_n; j++) fprintf(f, "%lf \n", x_contig.local[j]);
        }
    }
    fclose(f);

    f = fopen("PartInner_Striped_BiCGStab_x.txt", "w");
    for (int i = 0; i < num_procs; i++) {
        if (rank == i) {
            for (int j = 0; j < x_striped.local_n; j++) fprintf(f, "%lf \n", x_striped.local[j]);
        }
    }
    fclose(f);

    f = fopen("PartInner_True_BiCGStab_x.txt", "w");
    for (int i = 0; i < num_procs; i++) {
        if (rank == i) {
            for (int j = 0; j < x_true.local_n; j++) fprintf(f, "%lf \n", x_true.local[j]);
        }
    }
    fclose(f);
    
    /*if (rank == 0) printf("Testing Contiguous Solution\n");
    for (int i = 0; i < x_true.local_n; i++) {
        assert(fabs(x_true.local[i] - x_contig.local[i]) < 1e-04);
        printf("%lf %lf\n", 1e-04, fabs(x_true.local[i] - x_contig.local[i]));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) printf("Testing Striped Solution\n");
    for (int i = 0; i < x_true.local_n; i++) {
        assert(fabs(x_true.local[i] - x_striped.local[i]) < 1e-04);
        printf("%lf %lf\n", 1e-04, fabs(x_true.local[i] - x_striped.local[i]));
    }*/

    MPI_Comm_free(&contig_comm);
    MPI_Comm_free(&striped_comm);

    delete A;

    MPI_Finalize();

    return 0;
}
