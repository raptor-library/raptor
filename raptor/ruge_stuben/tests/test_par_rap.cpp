#include <assert.h>

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char name[128];
    int ctr = 0;
    int start, end;

    ParCSRMatrix* A;
    ParCSRMatrix* P;
    ParCSRMatrix* Ac;
    ParCSRMatrix* AP;
    ParCSCMatrix* AP_csc;
    ParCSRMatrix* Ac_rap;

    snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);
    while (FILE *file = fopen(name, "r")) 
    {
        fclose(file);
        
        snprintf(name, sizeof(name), "../../tests/rss_laplace_A%d.mtx", ctr);
        if (ctr == 0)
        {
            A = readParMatrix(name, MPI_COMM_WORLD, 1, 1);
        }
        else
        {
            A = readParMatrix(name, MPI_COMM_WORLD, 1, 0);
        }

        snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);        
        P = readParMatrix(name, MPI_COMM_WORLD, 1, 0);

        snprintf(name, sizeof(name), "../../tests/rss_laplace_A%d.mtx", ctr+1);        
        Ac = readParMatrix(name, MPI_COMM_WORLD, 1, 0);

        AP = A->mult(P);
        AP_csc = new ParCSCMatrix(AP);
        Ac_rap = P->mult_T(AP_csc);

        Ac->sort();
        Ac_rap->sort();

        assert(Ac->global_num_rows == Ac_rap->global_num_rows);
        assert(Ac->global_num_cols == Ac_rap->global_num_cols);
        assert(Ac->local_num_rows == Ac_rap->local_num_rows);
        assert(Ac->on_proc_num_cols == Ac_rap->on_proc_num_cols);

        assert(Ac->on_proc->idx1[0] == Ac_rap->on_proc->idx1[0]);
        assert(Ac->off_proc->idx1[0] == Ac_rap->off_proc->idx1[0]);
        for (int i = 0; i < Ac->local_num_rows; i++)
        {
            assert(Ac->on_proc->idx1[i+1] == Ac_rap->on_proc->idx1[i+1]);
            start = Ac->on_proc->idx1[i];
            end = Ac->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                assert(Ac->on_proc->idx2[j] == Ac_rap->on_proc->idx2[j]);
                assert(fabs(Ac->on_proc->vals[j] - Ac_rap->on_proc->vals[j]) < 1e-06);
            }

            assert(Ac->off_proc->idx1[i+1] == Ac_rap->off_proc->idx1[i+1]);
            start = Ac->off_proc->idx1[i];
            end = Ac->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                assert(Ac->off_proc_column_map[Ac->off_proc->idx2[j]] == 
                        Ac_rap->off_proc_column_map[Ac_rap->off_proc->idx2[j]]);
                assert(fabs(Ac->off_proc->vals[j] - Ac_rap->off_proc->vals[j]) < 1e-06);
            }
        }

        ctr++;
        snprintf(name, sizeof(name), "../../tests/rss_laplace_P%d.mtx", ctr);

        delete A;
        delete P;
        delete Ac;
        delete AP;
        delete AP_csc;
        delete Ac_rap;
    }


    snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);
    while (FILE *file = fopen(name, "r")) 
    {
        fclose(file);
        
        snprintf(name, sizeof(name), "../../tests/rss_aniso_A%d.mtx", ctr);
        if (ctr == 0)
        {
            A = readParMatrix(name, MPI_COMM_WORLD, 1, 1);
        }
        else
        {
            A = readParMatrix(name, MPI_COMM_WORLD, 1, 0);
        }

        snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);        
        P = readParMatrix(name, MPI_COMM_WORLD, 1, 0);

        snprintf(name, sizeof(name), "../../tests/rss_aniso_A%d.mtx", ctr+1);        
        Ac = readParMatrix(name, MPI_COMM_WORLD, 1, 0);

        AP = A->mult(P);
        AP_csc = new ParCSCMatrix(AP);
        Ac_rap = P->mult_T(AP_csc);

        Ac->sort();
        Ac_rap->sort();

        assert(Ac->global_num_rows == Ac_rap->global_num_rows);
        assert(Ac->global_num_cols == Ac_rap->global_num_cols);
        assert(Ac->local_num_rows == Ac_rap->local_num_rows);
        assert(Ac->on_proc_num_cols == Ac_rap->on_proc_num_cols);

        assert(Ac->on_proc->idx1[0] == Ac_rap->on_proc->idx1[0]);
        assert(Ac->off_proc->idx1[0] == Ac_rap->off_proc->idx1[0]);
        for (int i = 0; i < Ac->local_num_rows; i++)
        {
            assert(Ac->on_proc->idx1[i+1] == Ac_rap->on_proc->idx1[i+1]);
            start = Ac->on_proc->idx1[i];
            end = Ac->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                assert(Ac->on_proc->idx2[j] == Ac_rap->on_proc->idx2[j]);
                assert(fabs(Ac->on_proc->vals[j] - Ac_rap->on_proc->vals[j]) < 1e-06);
            }

            assert(Ac->off_proc->idx1[i+1] == Ac_rap->off_proc->idx1[i+1]);
            start = Ac->off_proc->idx1[i];
            end = Ac->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                assert(Ac->off_proc_column_map[Ac->off_proc->idx2[j]] == 
                        Ac_rap->off_proc_column_map[Ac_rap->off_proc->idx2[j]]);
                assert(fabs(Ac->off_proc->vals[j] - Ac_rap->off_proc->vals[j]) < 1e-06);
            }
        }

        ctr++;
        snprintf(name, sizeof(name), "../../tests/rss_aniso_P%d.mtx", ctr);

        delete A;
        delete P;
        delete Ac;
        delete AP;
        delete AP_csc;
        delete Ac_rap;
    }
}
