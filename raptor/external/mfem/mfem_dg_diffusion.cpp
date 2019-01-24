#include "external/mfem_wrapper.hpp"

using namespace mfem;

// Create MFEM Laplacian System and convert to RAPtor format
raptor::ParCSRMatrix* mfem_dg_diffusion(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor, const char* mesh_file, 
        const int order, const int seq_n_refines, const int par_n_refines,
        const MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    double sigma = -1.0;
    double kappa = (order + 1) * (order + 1);

    int mesh_dim;
    int boundary_n;

    Mesh* mesh;
    ParMesh* par_mesh;
    FiniteElementCollection* collection;
    ParFiniteElementSpace* space;

    mesh = new Mesh(mesh_file, 1, 1);
    mesh_dim = mesh->Dimension();

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

    // Form finite element collection / space
    boundary_n = par_mesh->bdr_attributes.Max();
    collection = new DG_FECollection(order, mesh_dim);
    space = new ParFiniteElementSpace(par_mesh, collection);

    ParLinearForm* b = new ParLinearForm(space);
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero, one, sigma, kappa));
    b->Assemble();

    ParGridFunction x(space);
    x = 0.0;

    ParBilinearForm* a = new ParBilinearForm(space);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));
    a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
    a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
    a->Assemble();
    a->Finalize();

    HypreParMatrix* A = a->ParallelAssemble();
    HypreParVector* B = b->ParallelAssemble();
    HypreParVector* X = x.ParallelProject();

    A->SetOwnerFlags(-1, -1, -1);
    hypre_ParCSRMatrix* A_hypre = A->StealData();
    raptor::ParCSRMatrix* A_raptor = convert(A_hypre, comm);
    x_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
    b_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
    hypre_ParVector* x_hypre = X->StealParVector();
    hypre_ParVector* b_hypre = B->StealParVector();
    double* x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_hypre));
    double* b_data = hypre_VectorData(hypre_ParVectorLocalVector(b_hypre));
    for (int i = 0; i < A_raptor->local_num_rows; i++)
    {
        x_raptor[i] = x_data[i];
        b_raptor[i] = b_data[i];
    }

    delete a;
    delete b;
    delete space;
    delete collection;
    delete par_mesh;

    return A_raptor;
}


