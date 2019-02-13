// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "trilinos_wrapper.hpp"

Epetra_Vector* epetra_convert(raptor::ParVector& x_rap,
                       RAPtor_MPI_Comm comm_mat)
{
    int rank;
    MPI_Comm_rank(comm_mat, &rank);
    Epetra_MpiComm Comm(comm_mat);

    Epetra_Map* Map = new Epetra_Map(x_rap.global_n, x_rap.local_n, 0, Comm);
    Epetra_Vector* x = new Epetra_Vector(Copy, *Map, x_rap.local.values.data());
    return x;
}

Epetra_CrsMatrix* epetra_convert(raptor::ParCSRMatrix* A_rap,
                       RAPtor_MPI_Comm comm_mat)
{
    int rank;
    MPI_Comm_rank(comm_mat, &rank);

    int nnz, start, end;

    Epetra_MpiComm Comm(comm_mat);

    // Initialize Map of Rows to Processes
    Epetra_Map rowMap(A_rap->global_num_rows, A_rap->local_num_rows, 0, Comm);

    // Calculate size of each local row
    aligned_vector<int> rowSizes(A_rap->local_num_rows);
    for (int i = 0; i < A_rap->local_num_rows; i++)
        rowSizes[i] = ((A_rap->on_proc->idx1[i+1] - A_rap->on_proc->idx1[i]) 
            + (A_rap->off_proc->idx1[i+1] - A_rap->off_proc->idx1[i]));

    // Initialize Epetra CSR Matrix
    const bool staticProfile = true;
    Epetra_CrsMatrix* A = new Epetra_CrsMatrix(Copy, rowMap, rowSizes.data(), staticProfile);

    aligned_vector<int> indices;
    aligned_vector<double> values;
    for (int i = 0; i < A_rap->local_num_rows; i++)
    {
        if (rowSizes[i] > indices.size())
        {
            indices.resize(rowSizes[i]);
            values.resize(rowSizes[i]);
        }

        // Add row indices (global) and values to arrays
        nnz = 0;
        start = A_rap->on_proc->idx1[i];
        end = A_rap->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            indices[nnz] = A_rap->on_proc_column_map[A_rap->on_proc->idx2[j]];
            values[nnz++] = A_rap->on_proc->vals[j];
        }
        start = A_rap->off_proc->idx1[i];
        end = A_rap->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            indices[nnz] = A_rap->off_proc_column_map[A_rap->off_proc->idx2[j]];
            values[nnz++] = A_rap->off_proc->vals[j];
        }
       
        // Fill CSR Matrix with Values
        A->InsertGlobalValues(A_rap->local_row_map[i], nnz, values.data(), indices.data());
    }   

    A->FillComplete();
    return A;
}

AztecOO* create_ml_hierarchy(Epetra_CrsMatrix* Ae, Epetra_Vector* xe, Epetra_Vector* be, int dim, int* incoords)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_iter = 100;
    double tol = 1e-07;

    RCP<Epetra_CrsMatrix> A = rcp(new Epetra_CrsMatrix(*Ae));
    RCP<Epetra_Vector> B = rcp(new Epetra_Vector(*be));
    RCP<Epetra_Vector> X = rcp(new Epetra_Vector(*xe));
    X->PutScalar(0.0);

    RCP<Epetra_MultiVector> xcoords = Teuchos::null;
    if (incoords)
    {
        Epetra_Map Map = A->RowMap();
        Epetra_MultiVector* coords = new Epetra_MultiVector(Map, dim);
        int MyLength = coords->MyLength();
        int NumVectors = coords->NumVectors();
        int idx = 0;
        for (int i = 0; i < MyLength; i++)
        {
            for (int j = 0; j < NumVectors; j++)
            {
                (*coords)[j][i] = incoords[idx++];
            }
        }
        xcoords = rcp(coords);
    }

    RCP<MueLu::EpetraOperator> mueLuPreconditioner;
    Teuchos::ParameterList paramList;
    paramList.set("verbosity", "none");
    paramList.set("multigrid algorithm", "sa");
    paramList.set("aggregation: type", "uncoupled");
    paramList.set("coarse: max size", 50);
    paramList.set("repartition: enable", true);
    mueLuPreconditioner = MueLu::CreateEpetraPreconditioner(A, paramList, xcoords);

    Epetra_LinearProblem Problem(A.get(), X.get(), B.get());
    AztecOO* solver = new AztecOO(Problem);
    solver->SetPrecOperator(mueLuPreconditioner.get());
    solver->SetAztecOption(AZ_solver, AZ_fixed_pt);
    solver->SetAztecOption(AZ_output, 0);

    solver->Iterate(num_iter, tol);

    // Print relative residual norm
    Epetra_Vector* ex = X.get();
    Epetra_Vector* eb = B.get();
    Epetra_Vector er(A->RowMap());
    Epetra_CrsMatrix* eA = A.get();
    eA->Multiply(false, *ex, er);
    er.Update(-1, *eb, 1);
    double rnorm = 0.0;
    er.Norm2(&rnorm);
    if (rank == 0) printf("Residual Norm: %e\n", rnorm);

    return solver;
}

