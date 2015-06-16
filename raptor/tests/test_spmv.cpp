#include <math.h>
#include "Matrix.hpp"
#include "ParMatrix.hpp"
#include "ParVector.hpp"
#include "Diffusion.hpp"

int main ( int argc, char *argv[] )
{  
   int ilower, iupper;
   int local_size, extra;
   double strong_threshold;

   double eps = 1.0;
   double theta = 0.0;

   int nx = 33;
   int ny = 33;
   int nz = 0;

   //Initialize Matrix and Vectors
   ParMatrix A;
   ParVector x;
   ParVector b;
   
   /* Initialize MPI */
   MPI::Init( argc, argv );
   int numProcs = MPI::COMM_WORLD.Get_size( );
   int myid = MPI::COMM_WORLD.Get_rank( );
 
   double *stencil = diffusion_stencil_2d(eps, theta);  
   ParMatrix A = stencil_grid(stencil, nx, ny, nz); 

   // Create the rhs and solution
   b = ParVector(globalNumRows, localNumRows);
   x = ParVector(globalNumRows, localNumRows);
   
   // Set the rhs values to h^2 and the solution to zero
   {
      double *rhs_values, *x_values;
      int    *rows;
      
      rhs_values = calloc(local_size, sizeof(double));
      
      for (i=0; i<local_size; i++)
      {
         rhs_values[i] = h2;
      }
      
      b.SetValues(rhs_values);
      x.SetConstValue(0.0);
   }

   

   // Finalize MPI
   MPI_Finalize();
   
   return(0);
}


