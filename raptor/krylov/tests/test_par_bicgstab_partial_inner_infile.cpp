#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
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
    double frac;
    if (argc == 3) {
       fname = argv[1];
       frac  = atof(argv[2]);
    }
    else {
        printf("Input <matrix filename> <fraction for partial inner product>\n");
        exit(-1);
    }

    // Needed for partial inner products
    int inner_color, root_color, inner_root, procs_in_group, part_global;
    MPI_Comm inner_comm = MPI_COMM_NULL;
    MPI_Comm root_comm = MPI_COMM_NULL;

    // Setup problem to solve
    ParCSRMatrix* A = readParMatrix(fname);
    
    ParVector x_part(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector x_true(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals_true;
    aligned_vector<double> residuals_pre;
    aligned_vector<double> residuals_part;
    aligned_vector<double> residuals_prepart;
    
    // Setup AMG hierarchy
    ParMultilevel *ml;
    ml = new ParSmoothedAggregationSolver(0.0);
    ml->max_levels = 3;
    ml->setup(A);

    x_true.set_const_value(1.0);
    A->mult(x_true, b);
    x_true.set_const_value(0.0);
    BiCGStab(A, x_true, b, residuals_true);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Preconditioned Solution
    x_true.set_const_value(1.0);
    A->mult(x_true, b);
    x_true.set_const_value(0.0);
    Pre_BiCGStab(A, ml, x_true, b, residuals_pre);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test contiguous first half
    //create_partial_inner_comm(inner_comm, color, first_root, second_root, part_global, 0);
    x_part.set_const_value(1.0);
    A->mult(x_part, b);
    x_part.set_const_value(0.0);
    PI_BiCGStab(A, x_part, b, residuals_part, inner_comm, root_comm, frac, inner_color, root_color, inner_root, procs_in_group,
                part_global);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test preconditioned half
    aligned_vector<double> sas_inner_prods;
    aligned_vector<double> asas_inner_prods;

    x_part.set_const_value(1.0);
    A->mult(x_part, b);
    x_part.set_const_value(0.0);
    PrePI_BiCGStab(A, ml, x_part, b, residuals_prepart, sas_inner_prods, asas_inner_prods, inner_comm, root_comm,
                   frac, inner_color, root_color, inner_root, procs_in_group, part_global);

    MPI_Barrier(MPI_COMM_WORLD);

    // Write out residuals to file
    /*FILE *f;
    if (rank == 0) {
        const char *start_fname = "_PartInner_";
        const char *end_fname = "_BiCGStab_Res.txt";
        char fname_buffer[512];
        sprintf(fname_buffer, "%s%s%f%s", fname, start_fname, frac, end_fname);
        printf("%s\n", fname_buffer);
        f = fopen(fname_buffer, "w");
        fprintf(f, "%d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<residuals_part.size(); i++) {
            fprintf(f, "%lf \n", residuals_part[i]);
        }
        fclose(f);
       
        sprintf(fname_buffer, "%s%s", fname, end_fname);
        f = fopen(fname_buffer, "w");
        fprintf(f, "%d x %d\n", A->global_num_rows, A->global_num_cols);
        for (int i=0; i<residuals_true.size(); i++) {
            fprintf(f, "%lf\n", residuals_true[i]);
        }
        fclose(f);
    }*/
    
    // Write out residuals to file
    if (rank == 0) {
        FILE *f;
        const char *start_fname = "_PartInner_";
        const char *end_fname = "_BiCGStab_Res.txt";
        char fname_buffer[512];
        sprintf(fname_buffer, "%s%s%f%s", fname, start_fname, frac, end_fname);
        f = fopen(fname_buffer, "w");
        for (int i=0; i<residuals_part.size(); i++) {
            fprintf(f, "%lf \n", residuals_part[i]);
        }
        fclose(f);
       
        const char *start_fname2 = "_PrePartInner_"; 
        sprintf(fname_buffer, "%s%s%f%s", fname, start_fname2, frac, end_fname);
        f = fopen(fname_buffer, "w");
        for (int i=0; i<residuals_prepart.size(); i++) {
            fprintf(f, "%lf \n", residuals_prepart[i]);
        }
        fclose(f);
       
        sprintf(fname_buffer, "%s%s", fname, end_fname);
        f = fopen(fname_buffer, "w");
        for (int i=0; i<residuals_true.size(); i++) {
            fprintf(f, "%lf\n", residuals_true[i]);
        }
        fclose(f);
        
        const char *start_fname3 = "_Pre";
        sprintf(fname_buffer, "%s%s%s", fname, start_fname3, end_fname);
        f = fopen(fname_buffer, "w");
        for (int i=0; i<residuals_pre.size(); i++) {
            fprintf(f, "%lf\n", residuals_pre[i]);
        }
        fclose(f);
    }

    // Write out solutions to file
    /*f = fopen("PartInner_Contig_BiCGStab_x.txt", "w");
    for (int i = 0; i < num_procs; i++) {
        if (rank == i) {
            for (int j = 0; j < x_part.local_n; j++) fprintf(f, "%lf \n", x_part.local[j]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    fclose(f);

    f = fopen("PartInner_True_BiCGStab_x.txt", "w");
    for (int i = 0; i < num_procs; i++) {
        if (rank == i) {
            for (int j = 0; j < x_true.local_n; j++) fprintf(f, "%lf \n", x_true.local[j]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    fclose(f);*/
    
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

    MPI_Comm_free(&inner_comm);
    MPI_Comm_free(&root_comm);

    delete A;

    MPI_Finalize();

    return 0;
}
