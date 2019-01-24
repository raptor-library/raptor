#include "external/mfem_wrapper.hpp"

using namespace mfem;

// Create an MFEM Linear Elasticity Matrix and convert to Raptor format
raptor::ParCSRMatrix* mfem_adaptive_laplacian(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor, const char* mesh_file, int order, 
        int max_dofs, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int mesh_dim;
    int space_dim;
    int boundary_n;

    Mesh* mesh;
    raptor::ParCSRMatrix* A_raptor = NULL;
 
    mesh = new Mesh(mesh_file, 1, 1);
    mesh_dim = mesh->Dimension();
    space_dim = mesh->SpaceDimension();

    // Uniform refinement on parallel mesh
    if (mesh->NURBSext)
    {
        printf("NURBS\n");
        mesh->UniformRefinement();
        mesh->SetCurvature(2);
    }
    mesh->EnsureNCMesh();

    ParMesh par_mesh(comm, *mesh);
    delete mesh;

    // Get dims
    boundary_n = par_mesh.bdr_attributes.Max();
    Array<int> bdry(boundary_n);
    bdry = 1;

    // Form finite element collection and space
    H1_FECollection collection(order, mesh_dim);
    ParFiniteElementSpace space(&par_mesh, &collection);

    // Form linear / bilinear forms (A, rhs)
    ParBilinearForm a(&space);
    ParLinearForm b(&space);
    ConstantCoefficient one(1.0);
    BilinearFormIntegrator *integrator = new DiffusionIntegrator(one);
    a.AddDomainIntegrator(integrator);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));

    // Form grid function x (solution)
    ParGridFunction x(&space);
    x = 0;

    L2_FECollection flux_collection(order, mesh_dim);
    ParFiniteElementSpace flux_space(&par_mesh, &flux_collection, space_dim);
    RT_FECollection smooth_flux_collection(order-1, mesh_dim);
    ParFiniteElementSpace smooth_flux_space(&par_mesh, &smooth_flux_collection);
    L2ZienkiewiczZhuEstimator estimator(*integrator, x, flux_space, smooth_flux_space);

    ThresholdRefiner refiner(estimator);
    refiner.SetTotalErrorFraction(0.7);

    max_dofs = 10000000;
    while(1)
    {
        a.Assemble();
        b.Assemble();

        Array<int> dofs;
        space.GetEssentialTrueDofs(bdry, dofs);

        HypreParMatrix A;
        mfem::Vector B, X;
        const int copy_interior = 1;
        a.FormLinearSystem(dofs, x, b, A, X, B, copy_interior);

        // Solve current system with AMG
        HypreBoomerAMG amg;
        amg.SetPrintLevel(0);
        CGSolver pcg(A.GetComm());
        pcg.SetPreconditioner(amg);
        pcg.SetOperator(A);
        pcg.SetRelTol(1e-6);
        pcg.SetMaxIter(200);
        pcg.SetPrintLevel(3);
        pcg.Mult(B, X);
        a.RecoverFEMSolution(X, b, x);

        refiner.Apply(par_mesh);
        if (space.GlobalTrueVSize() > max_dofs || refiner.Stop())
        {
            A.SetOwnerFlags(-1, -1, -1);
            hypre_ParCSRMatrix* A_hypre = A.StealData();
            A_raptor = convert(A_hypre, comm);
            x_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
            b_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows);
            double* x_data = X.GetData();
            double* b_data = B.GetData();
            for (int i = 0; i < A_raptor->local_num_rows; i++)
            {
                x_raptor[i] = x_data[i];
                b_raptor[i] = b_data[i];
            }

            break;
        }

        space.Update();
        x.Update();

        // Load balance mesh, updating solution (only for nonconforming meshes)
        if (par_mesh.Nonconforming())
        {
            par_mesh.Rebalance();
            space.Update();
            x.Update();
        }

        a.Update();
        b.Update();
    }

    return A_raptor;
}


