// Lines 11-151 of this code were taken from MFEM example, ex2p.cpp (lines 76-216)

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void mfem_linear_elasticity(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, const char* mesh_file, int num_elements, int order, MPI_Comm comm_mat)
{
    int myid, num_procs;
    MPI_Comm_rank(comm_mat, &myid);
    MPI_Comm_size(comm_mat, &num_procs);

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      MPI_Finalize();
      return;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      if (myid == 0)
         cerr << "\nInput mesh should have at least two materials and "
              << "two boundary attributes! (See schematic in ex2.cpp)\n"
              << endl;
      MPI_Finalize();
      return;
   }

   // 4. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext && order > mesh->NURBSext->GetOrder())
      mesh->DegreeElevate(order - mesh->NURBSext->GetOrder());

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log((1.0*num_elements)/mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
      for (int l = 0; l < num_elements; l++)
         mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(comm_mat, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   if (pmesh->NURBSext)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   }
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl
           << "Assembling: " << flush;

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
      f.Set(i, new ConstantCoefficient(0.0));
   {
      mfem::Vector pull_force(pmesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myid == 0)
      cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu. The boundary conditions are
   //     implemented by marking only boundary attribute 1 as essential. After
   //     serial/parallel assembly we extract the corresponding parallel matrix.
   mfem::Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   mfem::Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   if (myid == 0)
      cout << "matrix ... " << flush;
   a->Assemble();
   mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   if (myid == 0)
      cout << "done." << endl;

   // 11. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   hypre_ParCSRMatrix *A_hypre = A->StealData();
   double* b_hypre = B->GetData();
   double* x_hypre = X->GetData();

   raptor::ParMatrix *A_raptor = convert(A_hypre, comm_mat);
   raptor::ParVector* b_raptor = new raptor::ParVector(A_raptor->global_rows, A_raptor->local_rows, A_raptor->first_row);
   raptor::ParVector* x_raptor = new raptor::ParVector(A_raptor->global_rows, A_raptor->local_rows, A_raptor->first_row);

   raptor::data_t* x_raptor_data = x_raptor->local->data();
   raptor::data_t* b_raptor_data = b_raptor->local->data();
   for (int i = 0; i < A_raptor->local_rows; i++)
   {
      x_raptor_data[i] = x_hypre[i];
      b_raptor_data[i] = b_hypre[i];
   }
   
   delete B;
   delete X;
   delete A;
   //remove_shared_ptrs(A_hypre);
   hypre_ParCSRMatrixDestroy(A_hypre);

   *A_raptor_ptr = A_raptor;
   *x_raptor_ptr = x_raptor;
   *b_raptor_ptr = b_raptor;

}
