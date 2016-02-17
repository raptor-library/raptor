// Lines 11-99 of this code were taken from MFEM example, ex1p.cpp (lines 74-162)

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void mfem_laplace(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, const char* mesh_file, int num_elements, int order, bool visualization)
{
    int myid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log((1.0*num_elements)/mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
      fec = new H1_FECollection(order, dim);
   else if (pmesh->GetNodes())
      fec = pmesh->GetNodes()->OwnFEC();
   else
      fec = new H1_FECollection(order = 1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl;

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 9. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential. After serial and
   //    parallel assembly we extract the corresponding parallel matrix A.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

   hypre_ParCSRMatrix *A_hypre = A->StealData();
   double* b_hypre = B->GetData();
   double* x_hypre = X->GetData();

   //raptor::ParMatrix *A_raptor = convert(A_hypre, MPI_COMM_WORLD);
   raptor::ParMatrix* A_raptor = convert(A_hypre);
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
   remove_shared_ptrs(A_hypre);
   hypre_ParCSRMatrixDestroy(A_hypre);
   delete A;
  
//   delete fespace;
   if (order > 0)
      delete fec;
   delete pmesh;


   *A_raptor_ptr = A_raptor;
   *x_raptor_ptr = x_raptor;
   *b_raptor_ptr = b_raptor;

}
