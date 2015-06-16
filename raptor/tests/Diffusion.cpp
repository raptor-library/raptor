#include <math.h>
#include "Matrix.hpp"
#include "ParMatrix.hpp"
#include "ParVector.hpp"

int main ( int argc, char *argv[] )
{  
   int ilower, iupper;
   int local_size, extra;
   double strong_threshold;

   //Initialize Matrix and Vectors
   ParMatrix A;
   ParVector x;
   ParVector b;
   
   /* Initialize MPI */
   MPI::Init( argc, argv );
   int numProcs = MPI::COMM_WORLD.Get_size( );
   int myid = MPI::COMM_WORLD.Get_rank( );
   
   /* Default problem parameters */
   int n = 33;
   double eps = 0.001;
   double theta = 2.0 * M_PI / 16.0;
   
   // Solver Parameters
   strong_threshold = 0.25;
   
   if (myid==0) hypre_printf("Initialized MPI\n");
   
   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;
      
      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-n") == 0 )
         {
            arg_index++;
            n = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-eps") == 0 )
         {
            arg_index++;
            eps = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-theta") == 0 )
         {
            arg_index++;
            theta = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-str") == 0 )
         {
            arg_index++;
            strong_threshold = atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }
      
      if (print_usage)
      {
         if (myid == 0)
         {
            printf("\n");
            printf("Usage: %s [<options>]\n", argv[0]);
            printf("\n");
            printf("  -n <n>              : problem size in each direction (default: 33)\n");
            printf("  -solver <ID>        : solver ID\n");
            printf("                        0  - AMG (default) \n");
            printf("  -eps <n>            : strength of anisotropy");
            printf("  -theta <n>          : angle of diffusion");
            printf("\n");
         }
         MPI_Finalize();
         return (0);
      }
   }
   
   if (myid==0) hypre_printf("Parsed Command Line\n");
   
   
   double C = cos(theta);
   double S = sin(theta);
   double CS = C*S;
   double CC = C*C;
   double SS = S*S;
   
   double val1 =  (-1*eps - 1)*CC + (-1*eps - 1)*SS + ( 3*eps - 3)*CS;
   double val2 =  ( 2*eps - 4)*CC + (-4*eps + 2)*SS;
   double val3 =  (-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS;
   double val4 =  (-4*eps + 2)*CC + ( 2*eps - 4)*SS;
   double val5 =  ( 8*eps + 8)*CC + ( 8*eps + 8)*SS;
   
   /* Preliminaries: want at least one processor per row */
   if (n*n < numProcs) n = sqrt(numProcs) + 1;
   int globalNumRows = n*n; /* global number of rows */
   double h = 1.0/(n+1); /* mesh size*/
   double h2 = h*h;
   
   if (myid==0) hypre_printf("Calculated variables\n");
   
   /* Each processor knows only of its own rows - the range is denoted by ilower
    and upper.  Here we partition the rows. We account for the fact that
    N may not divide evenly by the number of processors. */
   localNumRows = globalNumRows/numProcs;
   extra = globalNumRows - localNumRows*numProcs;
   
   ilower = localNumRows*myid;
   ilower += hypre_min(myid, extra);
   
   iupper = localNumRows*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;
   
   /* How many rows do I have? */
   localNumRows = iupper - ilower + 1;
   
   /* Now go through my local rows and set the matrix entries.
    Each row has at most 9 entries.
    
    Note that here we are setting one row at a time, though
    one could set all the rows together (see the User's Manual).
    */
   {  
      std:vector<int> indptr;
      std:vector<int> cols;
      std:vector<double> data;
      
      int firstcol = ilower%n;
      int lastcol = (ilower+1)%n;
      
      indptr.push(0);
      for (i = ilower; i <= iupper; i++)
      {  
         // The left block : left position
         if ((i-(n+1))>=0 && firstcol)
         {
            cols.push(i-(n+1));
            data.push(val1);
         }
         
         // The left block : middle position
         if ((i-n)>=0)
         {
            cols.push(i-n);
            data.push(val2);
         }
         
         // The left block : right position
         if ((i-(n-1))>=0 && lastcol)
         {
            cols.push(i-(n-1));
            data.push(val3);
         }
         
         // The middle block : left position
         if (firstcol)
         {
            cols.push(i-1);
            data.push(val4);
         }
         
         // Set the diagonal: position i
         cols.push(i);
         data.push(val5);
         
         // The middle block : right position
         if (lastcol)
         {
            cols.push(i+1);
            data.push(val4);
         }
         
         // The right block : left position
         if ((i+(n-1))< N && firstcol)
         {
            cols.push(i+(n-1));
            data.push(val3);
         }
         
         // The right block : middle position
         if ((i+n)< N)
         {
            cols.push(i+n);
            data.push(val2);
         }
         
         // The right block : right position
         if ((i+(n+1))< N && lastcol)
         {
            cols.push(i+(n+1));
            data.push(val1);
         }

         indptr.push(cols.size());
         
         firstcol++;
         lastcol++;
         
         if (firstcol == n)
         firstcol = 0;
         if (lastcol == n)
         lastcol = 0;
      }
   }
   if (myid == 0) hypre_printf("Set values\n");
   
   // Assemble after setting the coefficients
   int globalRowStarts[numProcs+1];
   for (int i = 0; i < numProcs; i++)
   {
       globalRowStarts[i] = i*localNumRows;

       ilower = (globalNumRows/numProcs)*i;
       ilower += hypre_min(i, extra);
   
       iupper = (globalNumRows/numProcs)*(i+1);
       iupper += hypre_min(i+1, extra);
       iupper = iupper - 1;
   
       /* How many rows do I have? */
       globalRowStarts[i+1] = globalRowStarts[i] + iupper - ilower + 1;
   }
   A = ParMatrix(globalNumRows, globalNumRows, indptr, cols, data, globalRowStarts);
   if (myid==0) hypre_printf("Created A\n");
   
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
   if (myid==0) hypre_printf("Created vectors\n");
   
   double final_res_norm;
   int its;
   
   if (myid == 0) hypre_printf("Setting up solver...\n");
   
   // Finalize MPI
   MPI_Finalize();
   
   return(0);
}


