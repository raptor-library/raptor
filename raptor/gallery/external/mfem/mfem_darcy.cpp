// TODO -- right now just makes random vectors
// Lines 11-151 of this code were taken from MFEM example, ex2p.cpp (lines 76-216)

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const mfem::Vector & x, mfem::Vector & u);
double pFun_ex(const mfem::Vector & x);
void fFun(const mfem::Vector & x, mfem::Vector & f);
double gFun(const mfem::Vector & x);
double f_natural(const mfem::Vector & x);

void mfem_darcy(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, const char* mesh_file, int num_elements, int order, MPI_Comm comm_mat)
{
    StopWatch chrono;

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh *mesh;
    ifstream imesh(mesh_file);
    if (!imesh)
    {
        return;
    }
    mesh = new Mesh(imesh, 1, 1);
    imesh.close();
    int dim = mesh->Dimension();

    // 4. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 10,000 elements.
    {
        int ref_levels =
            (int)floor(log((1.0*num_elements)/mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(comm_mat, *mesh);
    delete mesh;
    {
        int par_ref_levels = 2;
        for (int l = 0; l < par_ref_levels; l++)
            pmesh->UniformRefinement();
    }

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
    FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, l2_coll);

    int dimR = R_space->GlobalTrueVSize();
    int dimW = W_space->GlobalTrueVSize();

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    mfem::Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = W_space->GetVSize();
    block_offsets.PartialSum();

    mfem::Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient k(1.0);

    VectorFunctionCoefficient fcoeff(dim, fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    VectorFunctionCoefficient ucoeff(dim, uFun_ex);
    FunctionCoefficient pcoeff(pFun_ex);

    // 9. Define the parallel grid function and parallel linear forms, solution
    //    vector and rhs.
    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();
    fform->ParallelAssemble(trueRhs.GetBlock(0));

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(W_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform->Assemble();
    gform->ParallelAssemble(trueRhs.GetBlock(1));

    // 10. Assemble the finite element matrices for the Darcy operator
    //
    //                            D = [ M  B^T ]
    //                                [ B   0  ]
    //     where:
    //
    //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
    //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
    ParBilinearForm *mVarf(new ParBilinearForm(R_space));
    ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));

    HypreParMatrix *M, *B;

    mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
    mVarf->Assemble();
    mVarf->Finalize();
    M = mVarf->ParallelAssemble();

    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();
    B = bVarf->ParallelAssemble();
    (*B) *= -1;

    hypre_ParCSRMatrix* M_hypre = M->StealData();
    hypre_ParCSRMatrix* B_hypre = B->StealData();

    // TODO -- these matrices are temporary... don't need comm_pkg
    raptor::ParMatrix* M_raptor = convert(M_hypre, comm_mat);
    raptor::ParMatrix* B_raptor = convert(B_hypre, comm_mat);

    // Create Darcy Matrix A manually
    int nnz = M_raptor->diag->nnz + B_raptor->diag->nnz;
    if (M_raptor->offd_num_cols) nnz += M_raptor->offd->nnz;
    if (B_raptor->offd_num_cols) nnz += B_raptor->offd->nnz;

    // B is not square... this is important
    int global_rows = M_raptor->global_rows + B_raptor->global_rows;
    int global_cols = M_raptor->global_cols + B_raptor->global_rows;
    int local_rows = M_raptor->local_rows + B_raptor->local_rows;
    int local_cols = M_raptor->local_cols + B_raptor->local_rows;
    int first_row = M_raptor->first_row + B_raptor->first_row;
    int first_col_diag = M_raptor->first_col_diag + B_raptor->first_row;

    raptor::ParMatrix* A_raptor = new raptor::ParMatrix(global_rows, global_cols,
        local_rows, local_cols, first_row, first_col_diag);

    int B_global_row_starts[num_procs] = {0};
    MPI_Allgather(&(B_raptor->first_row), 1, MPI_INT, 
           B_global_row_starts, 1, MPI_INT, MPI_COMM_WORLD);

    // Send columns of B to processes that hold equivalent rows
    int ctr = 0;
    int start_ctr, send_proc;
    int send_start, send_end;
    int global_col, global_row;
    int col_start, col_end;
    int row_start, row_end;
    int recv_proc;
    int recv_start, recv_end;
    int count;
    MPI_Status recv_status;
    double value;
    int local_rows_m = M_raptor->local_rows;
    int local_cols_m = M_raptor->local_cols;
    int col_b, row_b, col_m;
    int proc;
    int global_col_B, global_row_B;
    int* diag_col_proc;
    int* offd_col_proc;

    int tag = 4567;
    MPI_Datatype coo_type;
    create_coo_type(&coo_type);
    MPI_Type_commit(&coo_type);

    // Find which processes hold the rows that correspond to my 
    // global columns of B (must send to these procs)
    if (B_raptor->local_cols)
        diag_col_proc = new int[B_raptor->local_cols];
    if (B_raptor->offd_num_cols)
        offd_col_proc = new int[B_raptor->offd_num_cols];

    proc = 0;
    for (int i = 0; i < B_raptor->local_cols; i++)
    {
        global_col_B = i + B_raptor->first_col_diag;
        while (proc + 1 < num_procs &&
                M_raptor->global_col_starts[proc + 1] <= global_col_B)
            proc++;
        diag_col_proc[i] = proc;
    }
    proc = 0;
    for (int i = 0; i < B_raptor->offd_num_cols; i++)
    {
        global_col_B = B_raptor->local_to_global[i];
        while(proc + 1 < num_procs &&
                M_raptor->global_col_starts[proc + 1] <= global_col_B)
            proc++;
        offd_col_proc[i] = proc;
    }

    // Create send_data, containing the coo_data to send to each process
    // Also, count the num of processes I must send to
    std::vector<coo_data>* send_data = new std::vector<coo_data>[num_procs];
    int* send_procs = new int[num_procs]();
    int* recv_procs = new int[num_procs];
    int nnz_off_proc = 0;
    for (int i = 0; i < B_raptor->local_rows; i++)
    {
        row_start = B_raptor->diag->idx1[i];
        row_end = B_raptor->diag->idx1[i+1];
        global_row_B = i + B_raptor->first_row;
        for (int j = row_start; j < row_end; j++)
        {
            col_b = B_raptor->diag->idx2[j];
            global_col_B = col_b + B_raptor->first_col_diag;

            // Want to find which proc hold row of M equal to global_col_B
            proc = diag_col_proc[col_b];

            if (proc == myid)
            {
                A_raptor->add_value(global_col_B - M_raptor->first_row,
                        global_row_B + M_raptor->first_row + M_raptor->local_rows,
                        B_raptor->diag->vals[j]);
            }
            else
            {
                send_data[proc].push_back({global_col_B, global_row_B, 
                        B_raptor->diag->vals[j]});
                send_procs[proc] = 1;
            }
        }
    }
    for (int i = 0; i < B_raptor->offd_num_cols; i++)
    {
        col_start = B_raptor->offd->idx1[i];
        col_end = B_raptor->offd->idx1[i+1];
        global_col_B = B_raptor->local_to_global[i];
        proc = offd_col_proc[i];
        if (proc == myid)
        {   for (int j = col_start; j < col_end; j++)
            {
                global_row_B = B_raptor->offd->idx2[j] + B_raptor->first_row;
                A_raptor->add_value(global_col_B - M_raptor->first_row,
                        global_row_B + M_raptor->first_row + M_raptor->local_rows,
                        B_raptor->offd->vals[j]);
            }

        }
        else
        {
            for (int j = col_start; j < col_end; j++)
            {
                global_row_B = B_raptor->offd->idx2[j] + B_raptor->first_row;
                send_data[proc].push_back({global_col_B, global_row_B,
                        B_raptor->offd->vals[j]});
                send_procs[proc] = 1;
            }
        }
    }

    // Allreduce to find the number of messages I must recv
    MPI_Allreduce(send_procs, recv_procs, num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int num_recvs = recv_procs[myid];
    delete[] send_procs;
    delete[] recv_procs;

    // Send the data to each process that needs it
    MPI_Request* send_requests = new MPI_Request[num_procs];
    int num_sends = 0;
    for (int i = 0; i < num_procs; i++)
    {
        if (send_data[i].size())
        {
            MPI_Isend(send_data[i].data(), send_data[i].size(), coo_type, 
                    i, tag, MPI_COMM_WORLD, &(send_requests[num_sends++])); 
        }
    }

    // Recv messages of any size, from any source
    // until 'num_recvs' messages have been recvd
    // And add the recvd data to the appropriate row/col of A_raptor
    for (int i = 0; i < num_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, coo_type, &count);
        coo_data recv_data[count];
        MPI_Recv(&recv_data, count, coo_type, recv_status.MPI_SOURCE, tag, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            coo_data& tmp = recv_data[i];
            int row = tmp.row - M_raptor->first_row;
            int col = tmp.col + M_raptor->global_col_starts[recv_proc + 1];
            A_raptor->add_value(row, col,  tmp.value);
        }
    }

    // Wait for all sends to complete and delete sent data 
    MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    delete[] send_data;
    delete[] send_requests;
    MPI_Type_free(&coo_type);

    // THESE ARE ALL COLUMNS OF M
    //
    // Adding value from m -- local row is local row of m
    // Adding value from B -- local row is local row of b + local_num(m)
    // Global col from m == global col of m + first_col_diag of b
    // Global col from b == global col of b + first_col_diag of m + num cols m
    for (int i = 0; i < M_raptor->local_rows; i++)
    {
        row_start = M_raptor->diag->idx1[i];
        row_end = M_raptor->diag->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = A_raptor->first_col_diag + M_raptor->diag->idx2[j];
            A_raptor->add_value(i, 
                    global_col, 
                    M_raptor->diag->vals[j]);
        }
    }
    for (int i = 0; i < B_raptor->local_rows; i++)
    {
        row_start = B_raptor->diag->idx1[i];
        row_end = B_raptor->diag->idx1[i+1];
        global_row = A_raptor->first_row + local_rows_m + i;
        for (int j = row_start; j < row_end; j++)
        {
            col_b = B_raptor->diag->idx2[j];
            global_col = A_raptor->first_col_diag + B_raptor->diag->idx2[j];
            A_raptor->add_value(i + local_rows_m, 
                    global_col,
                    B_raptor->diag->vals[j]);
        }
    }

    // Global column in A = global column in M plus number of columns in B
    // that lie on processes prior to process holding equivalent row in M
    for (int i = 0; i < M_raptor->comm->num_recvs; i++)
    {
        recv_proc = M_raptor->comm->recv_procs[i];
        recv_start = M_raptor->comm->recv_col_starts[i];
        recv_end = M_raptor->comm->recv_col_starts[i+1];
        for (int col = recv_start; col < recv_end; col++)
        {
            col_start = M_raptor->offd->idx1[col];
            col_end = M_raptor->offd->idx1[col+1];
            global_col = M_raptor->local_to_global[col] 
                + B_global_row_starts[recv_proc];
            for (int j = col_start; j < col_end; j++)
            {
                A_raptor->add_value(M_raptor->offd->idx2[j],
                        global_col, M_raptor->offd->vals[j]);
            }
        }
    }

    // B isn't square, so commpkg probably isnt correct
    for (int i = 0; i < B_raptor->offd_num_cols; i++)
    {
        col_start = B_raptor->offd->idx1[i];
        col_end = B_raptor->offd->idx1[i+1];
        proc = offd_col_proc[i];
        global_col = B_raptor->local_to_global[i] + B_global_row_starts[recv_proc];
        for (int j = col_start; j < col_end; j++)
        {
            A_raptor->add_value(B_raptor->offd->idx2[j] + local_rows_m, 
                    global_col, B_raptor->offd->vals[j]);
        }
    }

    if (B_raptor->local_cols)
        delete[] diag_col_proc;
    if (B_raptor->offd_num_cols)
        delete[] offd_col_proc;

    A_raptor->finalize();

    raptor::ParVector* b_raptor = new raptor::ParVector(global_rows, local_rows, 
           first_row);
    raptor::ParVector* x_raptor = new raptor::ParVector(global_rows, local_rows, 
           first_row);
    raptor::data_t* b_raptor_data = b_raptor->local->data();
    raptor::data_t* x_raptor_data = x_raptor->local->data();
    for (int i = 0; i < A_raptor->local_rows; i++)
    {
        x_raptor_data[i] = 1.0;
        b_raptor_data[i] = 0.0;
    }

    hypre_ParCSRMatrixDestroy(M_hypre);
    hypre_ParCSRMatrixDestroy(B_hypre);
    delete M_raptor;
    delete B_raptor;

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete B;
    delete M;
    delete mVarf;
    delete bVarf;
    delete W_space;
    delete R_space;
    delete l2_coll;
    delete hdiv_coll;
    delete pmesh;

    *A_raptor_ptr = A_raptor;
    *x_raptor_ptr = x_raptor;
    *b_raptor_ptr = b_raptor;
}

void uFun_ex(const mfem::Vector & x, mfem::Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const mfem::Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const mfem::Vector & x, mfem::Vector & f)
{
    f = 0.0;
}

double gFun(const mfem::Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const mfem::Vector & x)
{
    return (-pFun_ex(x));
}
