// Lines 11-110 of this code were taken from MFEM example, ex4p.cpp (lines 78-177).  The methods F_exact and f_exact were also taken from this file.

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const mfem::Vector &, mfem::Vector &);
void f_exact(const mfem::Vector &, mfem::Vector &);

/**************************************************************
 *****   MFEM Hdiv Diffusion
 **************************************************************
 ***** Creates a linear elasticity matrix from MFEM
 *****
 ***** Parameters
 ***** -------------
 ***** A : raptor::ParMatrix**
 *****    Pointer to uninitialized parallel matrix for laplacian to be stored in
 ***** x : ParVector**
 *****    Pointer to uninitialized parallel vector for solution vector
 ***** y : ParVector**
 *****    Pointer to uninitialized parallel vector for rhs vector
 ***** mesh_file : char (const)
 *****    Location of file containing mesh for MFEM to use
 ***** num_elements: int
 *****    Maximum size for refined serial mesh
 ***** order : int (optional)
 *****    Use continuous Lagrange finite elements of this order.
 *****    If order < 1, uses isoparametric/isogeometric space.
 *****    Default = 3
 ***** set_bc : Bool (Optional)
 *****    Whether to set boundary conditions.  Default = True
 ***** comm_mat : MPI_Comm (optional)
 *****    MPI_Communicator for A.  Default = MPI_COMM_WORLD.
 **************************************************************/
void mfem_hdiv_diffusion(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, const char* mesh_file, int num_elements, int order, bool set_bc, MPI_Comm comm_mat)
{
   int myid, num_procs;
   MPI_Comm_rank(comm_mat, &myid);
   MPI_Comm_size(comm_mat, &num_procs);

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume, as well as periodic meshes with the same code.
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
   //    spaces on them (this is needed in the ADS solver below).
   ParMesh *pmesh = new ParMesh(comm_mat, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }
   pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Raviart-Thomas finite elements, but we can easily
   //    switch to higher-order spaces by changing the value of p.
   FiniteElementCollection *fec = new RT_FECollection(order-1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl;

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(dim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient F(dim, F_exact);
   x.ProjectCoefficient(F);

   // 9. Set up the parallel bilinear form corresponding to the H(div) diffusion
   //    operator grad alpha div + beta I, by adding the div-div and the
   //    mass domain integrators and finally imposing non-homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrix A.
   Coefficient *alpha = new ConstantCoefficient(1.0);
   Coefficient *beta  = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));
   a->Assemble();
   if (set_bc && pmesh->bdr_attributes.Size())
   {
      mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a->EliminateEssentialBC(ess_bdr, x, *b);
   }
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();
   *X = 0.0;

   delete a;
   delete alpha;
   delete beta;
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
   remove_shared_ptrs(A_hypre);
   hypre_ParCSRMatrixDestroy(A_hypre);

   *A_raptor_ptr = A_raptor;
   *x_raptor_ptr = x_raptor;
   *b_raptor_ptr = b_raptor;

}

// The exact solution
void F_exact(const mfem::Vector &p, mfem::Vector &F)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0;

   F(0) = cos(M_PI*x)*sin(M_PI*y);
   F(1) = cos(M_PI*y)*sin(M_PI*x);
   if (dim == 3)
      F(2) = 0.0;
}

// The right hand side
void f_exact(const mfem::Vector &p, mfem::Vector &f)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0;

   double temp = 1 + 2*M_PI*M_PI;

   f(0) = temp*cos(M_PI*x)*sin(M_PI*y);
   f(1) = temp*cos(M_PI*y)*sin(M_PI*x);
   if (dim == 3)
      f(2) = 0;
}
