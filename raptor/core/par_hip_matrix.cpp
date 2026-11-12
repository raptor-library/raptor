// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "par_hip_matrix.hpp"
#include "hip_error.hpp"
#include "par_matrix.hpp"
#include "comm_pkg.hpp"

namespace raptor
{
void upload_hip_matrix(ParCSRMatrix& host, ParHipCSRMatrix& device,
        rocsparse_handle handle)
{
    device.global_num_rows = host.global_num_rows;
    device.global_num_cols = host.global_num_cols;
    upload_hip_matrix(static_cast<CSRMatrix&>(*host.on_proc), device.on_proc, handle);
    upload_hip_matrix(static_cast<CSRMatrix&>(*host.off_proc), device.off_proc, handle);

    auto& send = *host.comm->send_data;
    auto& recv = *host.comm->recv_data;
    if (send.size_msgs > 0)
    {
        HIP_CHECK(hipMalloc(&device.d_sendbuf,
                static_cast<std::size_t>(send.size_msgs) * sizeof(double)));
        HIP_CHECK(hipMalloc(&device.d_send_idx,
                static_cast<std::size_t>(send.size_msgs) * sizeof(int)));
        HIP_CHECK(hipMemcpy(device.d_send_idx, send.indices.data(),
                static_cast<std::size_t>(send.size_msgs) * sizeof(int), hipMemcpyHostToDevice));
    }
    else
    {
        device.d_sendbuf = nullptr;
        device.d_send_idx = nullptr;
    }
    if (recv.size_msgs > 0)
        HIP_CHECK(hipMalloc(&device.d_recvbuf,
                static_cast<std::size_t>(recv.size_msgs) * sizeof(double)));
    else
        device.d_recvbuf = nullptr;

    if (send.size_msgs > 0)
        HIP_CHECK(hipHostMalloc(&device.h_sendbuf,
                static_cast<std::size_t>(send.size_msgs) * sizeof(double), hipHostMallocDefault));
    else
        device.h_sendbuf = nullptr;
    if (recv.size_msgs > 0)
        HIP_CHECK(hipHostMalloc(&device.h_recvbuf,
                static_cast<std::size_t>(recv.size_msgs) * sizeof(double), hipHostMallocDefault));
    else
        device.h_recvbuf = nullptr;
    device.comm = host.comm;
}

void free_hip_matrix(ParHipCSRMatrix& device)
{
    free_hip_matrix(device.on_proc);
    free_hip_matrix(device.off_proc);
    if (device.d_sendbuf) HIP_CHECK(hipFree(device.d_sendbuf));
    if (device.d_recvbuf) HIP_CHECK(hipFree(device.d_recvbuf));
    if (device.d_send_idx) HIP_CHECK(hipFree(device.d_send_idx));
    if (device.h_sendbuf) HIP_CHECK(hipHostFree(device.h_sendbuf));
    if (device.h_recvbuf) HIP_CHECK(hipHostFree(device.h_recvbuf));
}
} // namespace raptor
