// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/par_bicgstab.hpp"
#include "krylov/partial_inner.hpp"

using namespace raptor;

/**************************************************************************************
 BiCGStab 
 **************************************************************************************/
void BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
{
    /*           A : ParCSRMatrix for system to solve
     *           x : ParVector solution to solve for
     *           b : ParVector rhs of system to solve
     *         res : vector containing residuals of each iteration
     *         tol : tolerance for convergence
     *    max_iter : maximum number of iterations
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = x.global_n + 5;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    rr_inner = r.inner_product(r_star);
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    iter = 0;
    // Main BiCGStab Loop
    while (true)
    {
        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

	// s_i = r_i - alpha_i * Ap_i
	s.copy(r);
	s.axpy(Ap, -1.0*alpha);

        // omega_i = (As_i, s_i) / (As_i, As_i)
	A->mult(s, As);
	As_inner = As.inner_product(s);
	AsAs_inner = As.inner_product(As);
	omega = As_inner / AsAs_inner;

	// x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
	x.axpy(p, alpha);
	x.axpy(s, omega);

	// r_{i+1} = s_i - omega_i * As_i
	r.copy(s);
	r.axpy(As, -1.0*omega);

	// beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
	next_inner = r.inner_product(r_star);
	beta = (next_inner / rr_inner) * (alpha / omega);

	// p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
	p.scale(beta);
	p.axpy(r, 1.0);
	p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = r.norm(2);
	res.push_back(norm_r);

        if (norm_r < tol)
	{
	    if (rank == 0)
	    {
                printf("%d Iterations required to converge\n", iter);
		printf("2 Norm of Residual: %lg\n\n", norm_r);
	    }
	    return;
        }

	if (iter == max_iter)
	{
            if (rank == 0)
	    {
                printf("Max Iterations Reached.\n");
		printf("2 Norm of Residual: %lg\n\n", norm_r);
	    }
	    return;
	}

        iter++;
    }
}

void SeqInner_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    rr_inner = sequential_inner(r, r_star);
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
	Apr_inner = sequential_inner(Ap, r_star);
        alpha = rr_inner / Apr_inner;

	// s_i = r_i - alpha_i * Ap_i
	s.copy(r);
	s.axpy(Ap, -1.0*alpha);

        // omega_i = (As_i, s_i) / (As_i, As_i)
	A->mult(s, As);
	As_inner = sequential_inner(As, s);
	AsAs_inner = sequential_inner(As, As);
	omega = As_inner / AsAs_inner;

	// x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
	x.axpy(p, alpha);
	x.axpy(s, omega);

	// r_{i+1} = s_i - omega_i * As_i
	r.copy(s);
	r.axpy(As, -1.0*omega);

	// beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
	next_inner = sequential_inner(r, r_star);
	beta = (next_inner / rr_inner) * (alpha / omega);

	// p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
	p.scale(beta);
	p.axpy(r, 1.0);
	p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = r.norm(2);
	res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

<<<<<<< HEAD
/**************************************************************************************
 AMG Preconditioned BiCGStab 
 **************************************************************************************/
void Pre_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol, int max_iter)
{
    /*           A : ParCSRMatrix for system to solve
     *           x : ParVector solution to solve for
     *           b : ParVector rhs of system to solve
     *         res : vector containing residuals of each iteration
     *         tol : tolerance for convergence
     *    max_iter : maximum number of iterations
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;
    ParMultilevel* ml;
    ParVector p_hat;
    ParVector s_hat;
    int amg_iter;

    // Setup AMG hierarchy
    ml->max_levels = 3;
    ml = new ParMultilevel(0.0, CLJP, Classical, SOR);
    ml->setup(A);

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // BEGIN ALGORITHM
    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    // Use true residual inner product to start
    rr_inner = r.inner_product(r_star);
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // p_i = M^-1 p_i
        // Apply preconditioner
        //p_hat.set_const(0.0);
        //iter = ml->solve(p_hat, p);

        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

        // s_i = r_i - alpha_i * Ap_i
        s.copy(r);
        s.axpy(Ap, -1.0*alpha);

        // s_i = M^-1 s_i
        // Apply preconditioner

        // omega_i = (As_i, s_i) / (As_i, As_i)
        A->mult(s, As);
        As_inner = As.inner_product(s);
        AsAs_inner = As.inner_product(As);
        omega = As_inner / AsAs_inner;

        // x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
        x.axpy(p, alpha);
        x.axpy(s, omega);

        // r_{i+1} = s_i - omega_i * As_i
        r.copy(s);
        r.axpy(As, -1.0*omega);

        // beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
        next_inner = r.inner_product(r_star);
        beta = (next_inner / rr_inner) * (alpha / omega);

        // p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
        p.scale(beta);
        p.axpy(r, 1.0);
        p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = r.norm(2);
        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

void SeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    rr_inner = r.inner_product(r_star);
    norm_r = sequential_norm(r, 2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

	// s_i = r_i - alpha_i * Ap_i
	s.copy(r);
	s.axpy(Ap, -1.0*alpha);

        // omega_i = (As_i, s_i) / (As_i, As_i)
	A->mult(s, As);
	As_inner = As.inner_product(s);
	AsAs_inner = As.inner_product(As);
	omega = As_inner / AsAs_inner;

	// x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
	x.axpy(p, alpha);
	x.axpy(s, omega);

	// r_{i+1} = s_i - omega_i * As_i
	r.copy(s);
	r.axpy(As, -1.0*omega);

	// beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
	next_inner = r.inner_product(r_star);
	beta = (next_inner / rr_inner) * (alpha / omega);

	// p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
	p.scale(beta);
	p.axpy(r, 1.0);
	p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sequential_norm(r, 2);
	res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

void SeqInnerSeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol, int max_iter)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    rr_inner = sequential_inner(r, r_star);
    norm_r = sequential_norm(r, 2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = sequential_inner(Ap, r_star);
        alpha = rr_inner / Apr_inner;

	// s_i = r_i - alpha_i * Ap_i
	s.copy(r);
	s.axpy(Ap, -1.0*alpha);

        // omega_i = (As_i, s_i) / (As_i, As_i)
	A->mult(s, As);
	As_inner = sequential_inner(As, s);
	AsAs_inner = sequential_inner(As, As);
	omega = As_inner / AsAs_inner;

	// x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
	x.axpy(p, alpha);
	x.axpy(s, omega);

	// r_{i+1} = s_i - omega_i * As_i
	r.copy(s);
	r.axpy(As, -1.0*omega);

	// beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
	next_inner = sequential_inner(r, r_star);
	beta = (next_inner / rr_inner) * (alpha / omega);

	// p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
	p.scale(beta);
	p.axpy(r, 1.0);
	p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sequential_norm(r, 2);
	res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

/**************************************************************************************
 BiCGStab with some inner products replaced with partial inner product approximations
 **************************************************************************************/
void PI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, MPI_Comm &inner_comm,
                 int &my_color, int &first_root, int &second_root, int part_global, int contig, double tol, int max_iter)
{
    /*           A : ParCSRMatrix for system to solve
     *           x : ParVector solution to solve for
     *           b : ParVector rhs of system to solve
     *         res : vector containing residuals of each iteration
     *  inner_comm : communicator containing procs used in proc's partial inner product 
     *    my_color : 1 if proc in inner_comm, 0 if proc in recv_comm
     *  first_root : root for communicator of first half of vector
     * second_root : root for communicator of second half of vector
     * part_global : number of values used in partial inner product
     *      contig : determines whether to use contiguous or striped processes for partial inner products
     *         tol : tolerance for convergence
     *    max_iter : maximum number of iterations
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Create communicators for partial inner products
    create_partial_inner_comm(inner_comm, x, my_color, first_root, second_root, part_global, contig);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);
<<<<<<< HEAD

    // BEGIN ALGORITHM
    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    // Use true residual inner product to start
    rr_inner = r.inner_product(r_star);
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

        // s_i = r_i - alpha_i * Ap_i
        s.copy(r);
        s.axpy(Ap, -1.0*alpha);

        // omega_i = (As_i, s_i) / (As_i, As_i)
        A->mult(s, As);
        //As_inner = As.inner_product(s);
        //AsAs_inner = As.inner_product(As);
        // Replace single inner product with half inner for testing
        if (iter % 2 == 0) {
            //As_inner = half_inner(inner_comm, As, s, my_color, 0, first_root, second_root, part_global);
            As_inner = half_inner(inner_comm, As, s, my_color, 1, second_root, first_root, part_global);
            AsAs_inner = half_inner(inner_comm, As, As, my_color, 0, first_root, second_root, part_global);
        }
        else {
            //As_inner = half_inner(inner_comm, As, s, my_color, 1, second_root, first_root, part_global);
            As_inner = half_inner(inner_comm, As, s, my_color, 0, first_root, second_root, part_global);
            AsAs_inner = half_inner(inner_comm, As, As, my_color, 1, second_root, first_root, part_global);
        }
        omega = As_inner / AsAs_inner;

        // x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
        x.axpy(p, alpha);
        x.axpy(s, omega);

        // r_{i+1} = s_i - omega_i * As_i
        r.copy(s);
        r.axpy(As, -1.0*omega);

        // beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
        next_inner = r.inner_product(r_star);
        beta = (next_inner / rr_inner) * (alpha / omega);

        // p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
        p.scale(beta);
        p.axpy(r, 1.0);
        p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = r.norm(2);
        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

/**************************************************************************************
 Preconditioned BiCGStab with some inner products replaced with partial inner product approximations
 **************************************************************************************/
void PrePI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, MPI_Comm &inner_comm,
                 int &my_color, int &first_root, int &second_root, int part_global, int contig, double tol, int max_iter)
{
    /*           A : ParCSRMatrix for system to solve
     *           x : ParVector solution to solve for
     *           b : ParVector rhs of system to solve
     *         res : vector containing residuals of each iteration
     *  inner_comm : communicator containing procs used in proc's partial inner product 
     *    my_color : 1 if proc in inner_comm, 0 if proc in recv_comm
     *  first_root : root for communicator of first half of vector
     * second_root : root for communicator of second half of vector
     * part_global : number of values used in partial inner product
     *      contig : determines whether to use contiguous or striped processes for partial inner products
     *         tol : tolerance for convergence
     *    max_iter : maximum number of iterations
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Create communicators for partial inner products
    create_partial_inner_comm(inner_comm, x, my_color, first_root, second_root, part_global, contig);

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;
    ParMultilevel* ml;
    ParVector p_hat;
    ParVector s_hat;
    int amg_iter;

    // Setup AMG hierarchy
    ml->max_levels = 3;
    ml = new ParMultilevel(0.0, CLJP, Classical, SOR);
    ml->setup(A);

    int iter;
    data_t alpha, beta, omega;
    data_t rr_inner, next_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    // Same max iterations definition as pyAMG
    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.global_n)) + 2;
    }

    // Fixed Constructors
    r.resize(b.global_n, b.local_n, b.first_local);
    r_star.resize(b.global_n, b.local_n, b.first_local);
    p.resize(b.global_n, b.local_n, b.first_local);
    Ap.resize(b.global_n, b.local_n, b.first_local);
    As.resize(b.global_n, b.local_n, b.first_local);

    // BEGIN ALGORITHM
    // r0 = b - A * x0
    A->residual(x, b, r);

    // r* = r0
    r_star.copy(r);

    // p0 = r0
    p.copy(r);

    // Use true residual inner product to start
    rr_inner = r.inner_product(r_star);
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {
        // p_i = M^-1 p_i
        // Apply preconditioner
        //iter = ml->solve(p, )

        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

        // s_i = r_i - alpha_i * Ap_i
        s.copy(r);
        s.axpy(Ap, -1.0*alpha);

        // s_i = M^-1 s_i
        // Apply preconditioner

        // omega_i = (As_i, s_i) / (As_i, As_i)
        A->mult(s, As);
        //As_inner = As.inner_product(s);
        //AsAs_inner = As.inner_product(As);
        // Replace single inner product with half inner for testing
        if (iter % 2 == 0) {
            //As_inner = half_inner(inner_comm, As, s, my_color, 0, first_root, second_root, part_global);
            As_inner = half_inner(inner_comm, As, s, my_color, 1, second_root, first_root, part_global);
            AsAs_inner = half_inner(inner_comm, As, As, my_color, 0, first_root, second_root, part_global);
        }
        else {
            //As_inner = half_inner(inner_comm, As, s, my_color, 1, second_root, first_root, part_global);
            As_inner = half_inner(inner_comm, As, s, my_color, 0, first_root, second_root, part_global);
            AsAs_inner = half_inner(inner_comm, As, As, my_color, 1, second_root, first_root, part_global);
        }
        omega = As_inner / AsAs_inner;

        // x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
        x.axpy(p, alpha);
        x.axpy(s, omega);

        // r_{i+1} = s_i - omega_i * As_i
        r.copy(s);
        r.axpy(As, -1.0*omega);

        // beta_i = (r_{i+1}, r_star) / (r_i, r_star) * alpha_i / omega_i
        next_inner = r.inner_product(r_star);
        beta = (next_inner / rr_inner) * (alpha / omega);

        // p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
        p.scale(beta);
        p.axpy(r, 1.0);
        p.axpy(Ap, -1.0*beta*omega);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = r.norm(2);
        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0)
    {
        if (iter == max_iter)
        {
            printf("Max Iterations Reached.\n");
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
        else
        {
            printf("%d Iteration required to converge\n", iter);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}
