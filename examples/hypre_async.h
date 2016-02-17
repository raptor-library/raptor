#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"

#include <stdlib.h>
#include <time.h>
void clear_cache(int size, double* cache_list)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        cache_list[i] = rand()%10;
    }
}


HYPRE_Int
hypre_ParCSRCommHandleTest( hypre_ParCSRCommHandle *comm_handle, HYPRE_Int *comm_flags, HYPRE_Int *finished, HYPRE_Int *total_finished, HYPRE_Int *last_finished)
{
   hypre_MPI_Status            status;
   HYPRE_Int                   i;
   HYPRE_Int num_tests = hypre_ParCSRCommHandleNumRequests(comm_handle);
   HYPRE_Int tests_complete = 0;

   if ( comm_handle==NULL )
      return hypre_error_flag;

   if (num_tests)
   {
      for (i = 0; i < num_tests; i++)
      {
         if (comm_flags[i])
            continue;

         hypre_MPI_Test(&hypre_ParCSRCommHandleRequest(comm_handle, i), &comm_flags[i], &status);
         if (comm_flags[i])
         {
            finished[tests_complete++] = i;
         }
      }
    
   }

   *total_finished += tests_complete;
   *last_finished = tests_complete;

   if (*total_finished == num_tests)
   {
      hypre_TFree(hypre_ParCSRCommHandleRequests(comm_handle));
      hypre_TFree(comm_handle);
   }



   return hypre_error_flag;
}

HYPRE_Int
hypre_ParCSRMatrixAsyncMatvec( HYPRE_Complex       alpha,
                          hypre_ParCSRMatrix *A,
                          hypre_ParVector    *x,
                          HYPRE_Complex       beta,
                          hypre_ParVector    *y,
                          hypre_CSRMatrix **offd_proc )
{
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);
   hypre_Vector      *x_local  = hypre_ParVectorLocalVector(x);
   hypre_Vector      *y_local  = hypre_ParVectorLocalVector(y);
   HYPRE_Int          num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int          num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   hypre_Vector      *x_tmp;
   HYPRE_Int          x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_Int          y_size = hypre_ParVectorGlobalSize(y);
   HYPRE_Int          num_vectors = hypre_VectorNumVectors(x_local);
   HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int          ierr = 0;
   HYPRE_Int          num_sends, i, j, jv, index, start;

   HYPRE_Int          vecstride = hypre_VectorVectorStride( x_local );
   HYPRE_Int          idxstride = hypre_VectorIndexStride( x_local );

   HYPRE_Complex     *x_tmp_data, *x_buf_data;
   HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  ParMatvec returns ierr = 11 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 12 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 13 if both are true.
    *
    *  Because temporary vectors are often used in ParMatvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   x_tmp = hypre_SeqVectorCreate( num_cols_offd );
   hypre_SeqVectorInitialize(x_tmp);
   x_tmp_data = hypre_VectorData(x_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   x_buf_data = hypre_CTAlloc(HYPRE_Complex, hypre_ParCSRCommPkgSendMapStart
                                     (comm_pkg, num_sends));

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         x_buf_data[index++]
            = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
   /* ... The assert is because the following loop only works for 'column'
      storage of a multivector. This needs to be fixed to work more generally,
      at least for 'row' storage. This in turn, means either change CommPkg so
      num_sends is no.zones*no.vectors (not no.zones) or, less dangerously, put
      a stride in the logic of CommHandleCreate (stride either from a new arg or
      a new variable inside CommPkg).  Or put the num_vector iteration inside
      CommHandleCreate (perhaps a new multivector variant of it).
   */
   comm_handle = hypre_ParCSRCommHandleCreate
         ( 1, comm_pkg, x_buf_data, x_tmp_data );

   hypre_CSRMatrixMatvec( alpha, diag, x_local, beta, y_local);

   HYPRE_Int *finished = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommHandleNumRequests(comm_handle));
   HYPRE_Int total_finished = 0;
   HYPRE_Int new_completions = 0;

   HYPRE_Int num_cols = 0;
   HYPRE_Int *cols = hypre_CTAlloc(HYPRE_Int, num_cols_offd);

   HYPRE_Int proc_start, proc_end, proc_idx;

   HYPRE_Int *comm_flags = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommHandleNumRequests(comm_handle));

   while (total_finished < hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      hypre_ParCSRCommHandleTest(comm_handle, comm_flags, finished, &total_finished, &new_completions);

      num_cols = 0;
      for (i = 0; i < new_completions; i++)
      {
         proc_idx = finished[i];
         if (proc_idx >= num_recvs) continue;

         proc_start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, proc_idx);
         if (proc_idx < hypre_ParCSRCommPkgNumRecvs(comm_pkg) - 1)
            proc_end = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, finished[i+1]);
         else proc_end = num_cols_offd;
         if (hypre_CSRMatrixNumCols(offd_proc[proc_idx])) hypre_CSRMatrixMatvec( alpha, offd_proc[proc_idx], x_tmp, 1.0, y_local);
      }
   }

   hypre_TFree(comm_flags);
   hypre_SeqVectorDestroy(x_tmp);
   x_tmp = NULL;
   hypre_TFree(x_buf_data);

   return ierr;
}


/********************************************
 ***
 *** Create hypre_ParCSRMatrixOffdProc
 *** for asynchronous SpMV
 ***
 *******************************************/
hypre_CSRMatrix** create_offd_proc_array(hypre_ParCSRMatrix *A)
{
      //Initialize Varibles
      hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
      HYPRE_Int local_num_rows = hypre_CSRMatrixNumRows(offd);
      HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
      HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
      HYPRE_Int *offd_j = hypre_CSRMatrixJ(offd);
      HYPRE_Real *offd_data = hypre_CSRMatrixData(offd);
      hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);

      HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

      hypre_CSRMatrix **offd_proc = hypre_CTAlloc(hypre_CSRMatrix*, num_recvs);
      HYPRE_Int **offd_proc_i = hypre_CTAlloc(HYPRE_Int*, num_recvs);
      HYPRE_Int **offd_proc_j = hypre_CTAlloc(HYPRE_Int*, num_recvs);
      HYPRE_Real **offd_proc_data = hypre_CTAlloc(HYPRE_Real*, num_recvs);

      HYPRE_Int *proc_ctr = hypre_CTAlloc(HYPRE_Int, num_recvs);

      HYPRE_Int row_start, row_end, i, j, proc_idx;

      //Calculate nnz associated with each distant proc
      for (i = 0; i < local_num_rows; i++)
      {
        row_start = offd_i[i];
        row_end = offd_i[i+1];
        qsort1(offd_j, offd_data, row_start, row_end-1 );

        proc_idx = 0;
        for (j = row_start; j < row_end; j++)
        {
           while (recv_vec_starts[proc_idx+1] <= offd_j[j] && proc_idx < num_recvs - 1)
              proc_idx++;

           proc_ctr[proc_idx]++;
        }
      }

      //Create offd_proc for each distant proc
      for (i = 0; i < num_recvs; i++)
      {
         offd_proc[i] = hypre_CSRMatrixCreate(local_num_rows, global_num_rows, proc_ctr[i]);
         proc_ctr[i] = 0;

         hypre_CSRMatrixInitialize(offd_proc[i]);
         offd_proc_i[i] = hypre_CSRMatrixI(offd_proc[i]);
         offd_proc_j[i] = hypre_CSRMatrixJ(offd_proc[i]);
         offd_proc_data[i] = hypre_CSRMatrixData(offd_proc[i]);
      }

      //Copy data from offd to appropriate offd_proc matrix
      for (i = 0; i < local_num_rows; i++)
      {
        row_start = offd_i[i];
        row_end = offd_i[i+1];

        proc_idx = 0;
        for (j = 0; j < num_recvs; j++) offd_proc_i[j][i] = proc_ctr[j];

        for (j = row_start; j < row_end; j++)
        {
           while (recv_vec_starts[proc_idx+1] <= offd_j[j] && proc_idx < num_recvs - 1)
              proc_idx++;

           offd_proc_j[proc_idx][proc_ctr[proc_idx]] = offd_j[j];
           offd_proc_data[proc_idx][proc_ctr[proc_idx]++] = offd_data[j];
        }
      }
      for (j = 0; j < num_recvs; j++)
      {
         offd_proc_i[j][local_num_rows] = proc_ctr[j];
         hypre_CSRMatrixSetRownnz(offd_proc[j]);
      }

      //Free variables
      hypre_TFree(proc_ctr);

      //Set struct variable
      //hypre_ParCSRMatrixOffdProc(A) = offd_proc;
      return offd_proc;

      return 0;
}

