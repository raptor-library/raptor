#include "gallery/external/mfem_wrapper.hpp"

using namespace mfem;


void rhs_func(const mfem::Vector& p, mfem::Vector& rhs);
void sol_func(const mfem::Vector& p, mfem::Vector& sol);
double freq, kappa;

// Create an MFEM Grad Div system and convert to Raptor format
raptor::ParCSRMatrix* mfem_grad_div(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, int order, int seq_n_refines, 
        int par_n_refines, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int mesh_dim;
    int space_dim;
    int boundary_n;
    freq = 1.0;
    kappa = freq * M_PI;

    Mesh* mesh;
    ParMesh* par_mesh;
    FiniteElementCollection* collection;
    ParFiniteElementSpace* space;

    mesh = new Mesh(mesh_file, 1, 1);
    mesh_dim = mesh->Dimension();
    space_dim = mesh->SpaceDimension();


    // Uniform refinement on serial mesh
    for (int i = 0; i < seq_n_refines; i++)
    {
        mesh->UniformRefinement();
    }

    // Uniform refinement on parallel mesh
    par_mesh = new ParMesh(comm, *mesh);
    delete mesh;
    for (int i = 0; i < par_n_refines; i++)
    {
        par_mesh->UniformRefinement();
    }
    par_mesh->ReorientTetMesh();

    // Get dims
    boundary_n = par_mesh->bdr_attributes.Max();

    // Form finite element collection / space
    collection = new RT_FECollection(order-1, mesh_dim);
    space = new ParFiniteElementSpace(par_mesh, collection);

    Array<int> dofs;
    if (par_mesh->bdr_attributes.Size())
    {
        Array<int> bdry(boundary_n);
        bdry = 1;
        space->GetEssentialTrueDofs(bdry, dofs);
    }

    VectorFunctionCoefficient rhs(space_dim, rhs_func);
    ParLinearForm* b = new ParLinearForm(space);
    b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(rhs));
    b->Assemble();

    ParGridFunction x(space);
    VectorFunctionCoefficient sol(space_dim, sol_func);
    x.ProjectCoefficient(sol);

    Coefficient* alpha = new ConstantCoefficient(1.0);
    Coefficient* beta = new ConstantCoefficient(1.0);
    ParBilinearForm* a = new ParBilinearForm(space);
    a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));

    FiniteElementCollection* hfec = new DG_Interface_FECollection(order-1, mesh_dim);
    ParFiniteElementSpace* hfes = new ParFiniteElementSpace(par_mesh, hfec);
    a->EnableHybridization(hfes, new NormalTraceJumpIntegrator(), dofs);
    a->Assemble();

    HypreParMatrix A;
    mfem::Vector B, X;
    a->FormLinearSystem(dofs, x, *b, A, X, B);
    A.SetOwnerFlags(-1, -1, -1);
    hypre_ParCSRMatrix* A_hypre = A.StealData();

    raptor::ParCSRMatrix* A_raptor = convert(A_hypre, comm);
    x_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows,
            A_raptor->partition->first_local_row);
    b_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows,
            A_raptor->partition->first_local_row);
    double* x_data = X.GetData();
    double* b_data = B.GetData();
    for (int i = 0; i < A_raptor->local_num_rows; i++)
    {
        x_raptor[i] = x_data[i];
        b_raptor[i] = b_data[i];
    }
    
    delete hfes;
    delete hfec;
    delete a;
    delete alpha;
    delete beta;
    delete b;
    delete space;
    delete collection;
    delete par_mesh;

    return A_raptor;
}

void sol_func(const mfem::Vector &p, mfem::Vector& sol)
{
    int mesh_dim = p.Size();
    double a = p(0);
    double b = p(1);

    sol(0) = cos(kappa*a)*sin(kappa*b);
    sol(1) = cos(kappa*b)*sin(kappa*a);
    if (mesh_dim == 3)
    {
        sol(3) = 0.0;
    }
}

void rhs_func(const mfem::Vector& p, mfem::Vector& rhs)
{
    int mesh_dim = p.Size();
    double a = p(0);
    double b = p(1);
    double temp = 1 + 2*kappa*kappa;
    
    rhs(0) = temp*cos(kappa*a)*sin(kappa*b);
    rhs(1) = temp*cos(kappa*b)*sin(kappa*a);
    if (mesh_dim == 3)
    {
        rhs(2) = 0.0;
    }
}

