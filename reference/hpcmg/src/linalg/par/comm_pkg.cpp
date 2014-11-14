#include "comm_pkg.h"

#include <cmath>

namespace linalg
{
    namespace par
    {
        CommPkg::CommPkg(CSRMatrix *A):
            comm(A->comm())
        {
            // TODO: light constructor and matvec factory method
            sys::int_t j;
            sys::int_t num_cols_offd, num_cols_diag;
            sys::int_t proc_num, num_elmts;
            int num_procs, rank;
            int num_requests;
            MPI_Request *requests = NULL;
            MPI_Status  *status = NULL;
            sys::int_t *proc_mark, *proc_add;
            int *tmp, *recv_buf, *displs, *info;
            sys::int_t offd_col;
            int local_info;
            int ip, vec_start, vec_len;
            sys::int_t first_col_diag, last_col_diag,
                       first_row_index, last_row_index;

            A->local_range(&first_row_index, &last_row_index,
                               &first_col_diag, &last_col_diag);
            sys::int_t *col_map_offd = A->col_map_offd();

            num_cols_diag = A->diag().num_cols();
            num_cols_offd = A->offd().num_cols();
            Partition *col_part = A->col_part();

            MPI_Comm_size(comm, &num_procs);
            MPI_Comm_rank(comm, &rank);

            proc_mark = new sys::int_t[num_procs];
            proc_add = new sys::int_t[num_procs];
            info = new int[num_procs];

            /* ----------------------------------------------------------------------
             * determine which processors to receive from (set proc_mark) and num_recvs,
             * at the end of the loop proc_mark[i] contains the number of elements to be
             * received from Proc. i
             * ---------------------------------------------------------------------*/

            for (int i=0; i < num_procs; i++)
                proc_add[i] = 0;

            proc_num = 0;
            if (num_cols_offd) offd_col = col_map_offd[0];
            num_recvs = 0;
            j = 0;
            for (sys::int_t i=0; i < num_cols_offd; i++) {
                if (num_cols_diag) proc_num = fmin(num_procs - 1,
                                                   offd_col / num_cols_diag);
                while (col_part->low(proc_num) > offd_col)
                    proc_num -= 1;
                while (col_part->low(proc_num+1) - 1 < offd_col)
                    proc_num += 1;
                proc_mark[num_recvs] = proc_num;
                j = i;
                while (col_part->low(proc_num+1) > offd_col) {
                    proc_add[num_recvs]++;
                    if (j < num_cols_offd-1) {
                        j++;
                        offd_col = col_map_offd[j];
                    } else {
                        j++;
                        offd_col = col_part->low(num_procs);
                    }
                }
                num_recvs++;
                if (j < num_cols_offd) i = j-1;
                else i = j;
            }

            local_info = 2*num_recvs;

            MPI_Allgather(&local_info, 1, MPI_INT, info, 1, MPI_INT, comm);

            /* ----------------------------------------------------------------------
             * generate information to be sent: tmp contains for each recv_proc:
             * id of recv_procs, number of elements to be received for this processor,
             * indices of elements (in this order)
             * ---------------------------------------------------------------------*/

            displs = new int[num_procs+1];
            displs[0] = 0;
            for (int i=1; i< num_procs + 1; i++)
                displs[i] = displs[i-1] + info[i-1];
            recv_buf = new int[displs[num_procs]];

            recv_procs = NULL;
            tmp = NULL;
            if (num_recvs)
            {
                recv_procs = new sys::int_t[num_recvs];
                tmp = new int[local_info];
            }
            recv_vec_starts = new sys::int_t[num_recvs+1];

            j = 0;
            if (num_recvs) recv_vec_starts[0] = 0;
            for (sys::int_t i=0; i < num_recvs; i++)
            {
                num_elmts = proc_add[i];
                recv_procs[i] = proc_mark[i];
                recv_vec_starts[i+1] = recv_vec_starts[i] + num_elmts;
                tmp[j++] = proc_mark[i];
                tmp[j++] = num_elmts;
            }

            MPI_Allgatherv(tmp, local_info, MPI_INT, recv_buf, info,
                    displs, MPI_INT, comm);

            /* ----------------------------------------------------------------------
             * determine num_sends and number of elements to be sent
             * ---------------------------------------------------------------------*/


            num_sends = 0;
            num_elmts = 0;
            proc_add[0] = 0;
            for (int i=0; i< num_procs; i++)
            {
                j = displs[i];
                while ((int) j < displs[i+1])
                {
                    if (recv_buf[j++] == rank)
                    {
                        proc_mark[num_sends] = i;
                        num_sends++;
                        proc_add[num_sends] = proc_add[num_sends-1] +
                                              recv_buf[j];
                        break;
                    }
                    j++;
                }
            }

            /* ----------------------------------------------------------------------
             * determine send_procs and actual elements to be send (in send_map_elmts)
             * and send_map_starts whose i-th entry points to the beginning of the
             * elements to be send to proc. i
             * ---------------------------------------------------------------------*/

            send_procs = NULL;
            send_map_elmts = NULL;
            if (num_sends)
            {
                send_procs = new sys::int_t[num_sends];
                send_map_elmts = new sys::int_t[proc_add[num_sends]];
            }
            send_map_starts = new sys::int_t[num_sends+1];
            num_requests = num_recvs + num_sends;
            if (num_requests)
            {
                requests = new MPI_Request[num_requests];
                status = new MPI_Status[num_requests];
            }

            if (num_sends) send_map_starts[0] = 0;
            for (sys::int_t i=0; i < num_sends; i++)
            {
                send_map_starts[i+1] = proc_add[i+1];
                send_procs[i] = proc_mark[i];
            }

            j = 0;
            for (sys::int_t i=0; i < num_sends; i++)
            {
                vec_start = send_map_starts[i];
                vec_len = send_map_starts[i+1] - vec_start;
                ip = send_procs[i];
                MPI_Irecv(&send_map_elmts[vec_start], vec_len, MPI_INT,
                          ip, 0, comm, &requests[j++]);
            }
            for (sys::int_t i=0; i < num_recvs; i++)
            {
                vec_start = recv_vec_starts[i];
                vec_len = recv_vec_starts[i+1] - vec_start;
                ip = recv_procs[i];
                MPI_Isend(&col_map_offd[vec_start], vec_len, MPI_INT,
                          ip, 0, comm, &requests[j++]);
            }

            if (num_requests)
            {
                MPI_Waitall(num_requests, requests, status);
                delete[] requests;
                delete[] status;
            }

            if (num_sends)
            {
                for (sys::int_t i=0; i < send_map_starts[num_sends]; i++)
                    send_map_elmts[i] -= first_col_diag;
            }

            delete[] proc_add;
            delete[] proc_mark;
            delete[] tmp;
            delete[] recv_buf;
            delete[] displs;
            delete[] info;
            delete[] requests;
            delete[] status;
        }
    }
}
