// Lines 11-114 of this code were taken from MFEM example, ex3p.cpp (lines 70-173).  The methods E_exact and f_exact were also taken from this file.

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const mfem::Vector &, mfem::Vector &);
void f_exact(const mfem::Vector &, mfem::Vector &);

void mfem_electromagnetic_diffusion(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, const char* mesh_file, int num_elements, int order, bool visualization)
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
   if (dim != 3)
   {
      if (myid == 0)
         cerr << "\nThis example requires a 3D mesh\n" << endl;
      MPI_Finalize();
      return;
   }

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log((1.0*num_elements)/mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Nedelec finite elements, but we can easily switch
   //    to higher-order spaces by changing the value of p.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl;

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(3, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(3, E_exact);
   x.ProjectCoefficient(E);

   // 9. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl + sigma I, by adding the curl-curl and the
   //    mass domain integrators and finally imposing non-homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
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
   *X = 0.0;

   delete a;
   delete sigma;
   delete muinv;
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
   delete A;
   remove_shared_ptrs(A_hypre);
   hypre_ParCSRMatrixDestroy(A_hypre);

   *A_raptor_ptr = A_raptor;
   *x_raptor_ptr = x_raptor;
   *b_raptor_ptr = b_raptor;

}

// A parameter for the exact solution.
const double kappa = M_PI;

void E_exact(const mfem::Vector &x, mfem::Vector &E)
{
   E(0) = sin(kappa * x(1));
   E(1) = sin(kappa * x(2));
   E(2) = sin(kappa * x(0));
}

void f_exact(const mfem::Vector &x, mfem::Vector &f)
{
   f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
   f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
   f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
}
