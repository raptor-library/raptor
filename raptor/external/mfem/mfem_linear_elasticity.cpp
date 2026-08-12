#include "external/mfem_wrapper.hpp"

// basically MFEM ex2p
// two material constants are set to be 1 and 1 
using namespace mfem;

class ElasticityBilinearForm : public ParBilinearForm
{
  public:
    ElasticityBilinearForm(ParFiniteElementSpace* space)
        : ParBilinearForm(space)
    {
    }

    void get_system_vector(const ParGridFunction& grid_function,
            Vector& system_vector) const
    {
        if (static_cond)
        {
            static_cond->ReduceSolution(grid_function, system_vector);
        }
        else
        {
            grid_function.GetTrueDofs(system_vector);
        }
    }
};

static void get_rigid_body_candidates(
        const ElasticityBilinearForm& bilinear_form,
        ParFiniteElementSpace* space,
        ParFiniteElementSpace* system_space,
        const Array<int>& boundary_marker, int mesh_dim,
        std::vector<double>& candidates, int& num_candidates)
{
    Array<int> essential_dofs;
    system_space->GetEssentialTrueDofs(boundary_marker, essential_dofs);

    num_candidates = mesh_dim * (mesh_dim + 1) / 2;
    int local_size = system_space->GetTrueVSize();
    candidates.resize(num_candidates * local_size);

    for (int candidate = 0; candidate < num_candidates; candidate++)
    {
        VectorFunctionCoefficient coefficient(mesh_dim,
                [mesh_dim, candidate](const Vector& coordinate, Vector& value)
                {
                    value.SetSize(mesh_dim);
                    value = 0.0;

                    if (candidate < mesh_dim)
                    {
                        value(candidate) = 1.0;
                    }
                    else if (mesh_dim == 2)
                    {
                        value(0) = coordinate(1);
                        value(1) = -coordinate(0);
                    }
                    else
                    {
                        switch (candidate - mesh_dim)
                        {
                            case 0:
                                value(0) = coordinate(1);
                                value(1) = -coordinate(0);
                                break;
                            case 1:
                                value(1) = coordinate(2);
                                value(2) = -coordinate(1);
                                break;
                            case 2:
                                value(2) = coordinate(0);
                                value(0) = -coordinate(2);
                                break;
                        }
                    }
                });

        ParGridFunction mode(space);
        Vector true_mode;
        mode.ProjectCoefficient(coefficient);
        bilinear_form.get_system_vector(mode, true_mode);
        true_mode.SetSubVector(essential_dofs, 0.0);

        for (int i = 0; i < local_size; i++)
        {
            candidates[candidate * local_size + i] = true_mode(i);
        }
    }
}

// Create an MFEM Linear Elasticity Matrix and convert to Raptor format
raptor::ParCSRMatrix* mfem_linear_elasticity(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor, int* num_variables,
        const char* mesh_file, int order, int seq_n_refines, 
        int par_n_refines, RAPtor_MPI_Comm comm,
        std::vector<double>* rigid_body_candidates,
        int* num_rigid_body_candidates, bool static_cond, 
        double material_contrast)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(comm, &rank);
    RAPtor_MPI_Comm_size(comm, &num_procs);

    int mesh_dim;
    int par_mesh_n;
    int boundary_n;

    Mesh* mesh;
    ParMesh* par_mesh;
    FiniteElementCollection* collection;
    ParFiniteElementSpace* space;

    mesh = new Mesh(mesh_file, 1, 1);
    mesh_dim = mesh->Dimension();

    MFEM_VERIFY(mesh->attributes.Max() >= 2 &&
            mesh->bdr_attributes.Max() >= 2,
            "Linear elasticity requires a mesh with at least two material "
            "attributes and two boundary attributes.");
    MFEM_VERIFY(material_contrast > 0.0,
            "Material contrast must be positive.");

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

    // Get dims
    par_mesh_n = par_mesh->attributes.Max();
    boundary_n = par_mesh->bdr_attributes.Max();

    // Form finite element collection / space
    collection = new H1_FECollection(order, mesh_dim);
    space = new ParFiniteElementSpace(par_mesh, collection, mesh_dim, Ordering::byVDIM);

    Array<int> dofs;
    Array<int> bdry(boundary_n);
    bdry = 0;
    bdry[0] = 1;
    space->GetEssentialTrueDofs(bdry, dofs);

    VectorArrayCoefficient force(mesh_dim);
    for (int i = 0; i < mesh_dim-1; i++)
    {
        force.Set(i, new ConstantCoefficient(0.0));
    }
    mfem::Vector pull_force(boundary_n);
    pull_force = 0.0;
    pull_force(1) = -1.0e-2;
    force.Set(mesh_dim-1, new PWConstCoefficient(pull_force));

    ParLinearForm* b = new ParLinearForm(space);
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(force));
    b->Assemble();

    ParGridFunction x(space);
    x = 0.0;

    mfem::Vector lambda(par_mesh_n);
    lambda = 1.0;
    lambda(0) = lambda(1) * material_contrast;
    PWConstCoefficient lambda_func(lambda);
    mfem::Vector mu(par_mesh_n);
    mu = 1.0;
    mu(0) = mu(1) * material_contrast;
    PWConstCoefficient mu_func(mu);

    ElasticityBilinearForm *a = new ElasticityBilinearForm(space);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    HypreParMatrix A;
    mfem::Vector B, X;
    a->FormLinearSystem(dofs, x, *b, A, X, B);

    if (rigid_body_candidates != nullptr)
    {
        ParFiniteElementSpace* system_space = a->SCParFESpace();
        if (system_space == nullptr)
        {
            system_space = space;
        }
        MFEM_VERIFY(system_space->GetTrueVSize() == X.Size(),
                "Rigid-body candidates do not match the linear system.");

        int candidate_count;
        get_rigid_body_candidates(*a, space, system_space, bdry,
                mesh_dim, *rigid_body_candidates, candidate_count);
        if (num_rigid_body_candidates != nullptr)
        {
            *num_rigid_body_candidates = candidate_count;
        }
    }

    A.SetOwnerFlags(-1, -1, -1);
    hypre_ParCSRMatrix* A_hypre = A.StealData();

    raptor::ParCSRMatrix *A_raptor = raptor::convert(A_hypre, comm);
    x_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
    b_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
    double* x_data = X.GetData();
    double* b_data = B.GetData();
    for (int i = 0; i < A_raptor->local_num_rows; i++)
    {
        x_raptor[i] = x_data[i];
        b_raptor[i] = b_data[i];
    }
    *num_variables = mesh_dim;

    delete a;
    delete b;
   
    delete space;
    delete collection;
    delete par_mesh;

    return A_raptor;
}
