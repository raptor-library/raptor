// Lines 11-110 of this code were taken from MFEM example, ex4p.cpp (lines 78-177).  The methods analytic_solution, analytic_rhs, and SnapNodes were also taken from this file.

#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double analytic_solution(mfem::Vector &x);
double analytic_rhs(mfem::Vector &x);
void SnapNodes(Mesh &mesh);

void mfem_hdiv_diffusion(raptor::ParMatrix** A_raptor_ptr, raptor::ParVector** x_raptor_ptr, raptor::ParVector** b_raptor_ptr, int elem_type, int ref_levels, int order, bool always_snap, MPI_Comm comm_mat)
{
   int myid, num_procs;
   MPI_Comm_rank(comm_mat, &myid);
   MPI_Comm_size(comm_mat, &num_procs);

   // 3. Generate an initial high-order (surface) mesh on the unit sphere. The
   //    Mesh object represents a 2D mesh in 3 spatial dimensions. We first add
   //    the elements and the vertices of the mesh, and then make it high-order
   //    by specifying a finite element space for its nodes.
   int Nvert = 8, Nelem = 6;
   if (elem_type == 0)
   {
      Nvert = 6;
      Nelem = 8;
   }
   Mesh *mesh = new Mesh(2, Nvert, Nelem, 0, 3);

   if (elem_type == 0) // inscribed octahedron
   {
      const double tri_v[6][3] =
         {{ 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
          { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}};
      const int tri_e[8][3] =
         {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
          {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(tri_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddTriangle(tri_e[j], attribute);
      }
      mesh->FinalizeTriMesh(1, 1, true);
   }
   else // inscribed cube
   {
      const double quad_v[8][3] =
         {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
          {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};
      const int quad_e[6][4] =
         {{3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
          {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddQuad(quad_e[j], attribute);
      }
      mesh->FinalizeQuadMesh(1, 1, true);
   }

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);

   // 4. Refine the mesh while snapping nodes to the sphere. Number of parallel
   //    refinements is fixed to 2.
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
         mesh->UniformRefinement();

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap)
         SnapNodes(*mesh);
   }

   ParMesh *pmesh = new ParMesh(comm_mat, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();

         // Snap the nodes of the refined mesh back to sphere surface.
         if (always_snap)
            SnapNodes(*pmesh);
      }
      if (!always_snap || par_ref_levels < 1)
         SnapNodes(*pmesh);
   }

   // 5. Define a finite element space on the mesh. Here we use isoparametric
   //    finite elements -- the same as the mesh nodes.
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, &fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of unknowns: " << size << endl;

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coef (analytic_rhs);
   FunctionCoefficient sol_coef (analytic_solution);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coef));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet).
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();

   // 9. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //    b(.) and the finite element approximation.
   HypreParMatrix * A = a->ParallelAssemble();
   HypreParVector * B = b->ParallelAssemble();
   HypreParVector * X = x.ParallelAverage();

   delete a;
   delete b;

   hypre_ParCSRMatrix *A_hypre = A->StealData();
   double* b_hypre = B->GetData();
   double* x_hypre = X->GetData();

   //raptor::ParMatrix *A_raptor = convert(A_hypre, comm_mat);
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

double analytic_solution(mfem::Vector &x)
{
   double l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return x(0)*x(1)/l2;
}

double analytic_rhs(mfem::Vector &x)
{
   double l2 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
   return 7*x(0)*x(1)/l2;
}

void SnapNodes(Mesh &mesh)
{
   GridFunction &nodes = *mesh.GetNodes();
   mfem::Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));

      node /= node.Norml2();

      for (int d = 0; d < mesh.SpaceDimension(); d++)
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
   }
}
