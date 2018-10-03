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
    norm_r = r.norm(2);
    res.push_back(norm_r);

    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    iter = 0;

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
            printf("%d Iteration required to converge\n", iter-1);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
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
            printf("%d Iteration required to converge\n", iter-1);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

/**************************************************************************************
 AMG Preconditioned BiCGStab 
 **************************************************************************************/
void Pre_BiCGStab(ParCSRMatrix* A, ParMultilevel *ml, ParVector& x, ParVector& b, aligned_vector<double>& res, double tol,
                  int max_iter)
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
    ParVector p_hat;
    ParVector s_hat;

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
    p_hat.resize(b.global_n, b.local_n, b.first_local);
    s_hat.resize(b.global_n, b.local_n, b.first_local);
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
        p_hat.set_const_value(0.0);
        ml->cycle(p_hat, p);

        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p_hat, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

        // s_i = r_i - alpha_i * Ap_i
        s.copy(r);
        s.axpy(Ap, -1.0*alpha);

        // s_i = M^-1 s_i
        // Apply preconditioner
        s_hat.set_const_value(0.0);
        ml->cycle(s_hat, s);

        // omega_i = (As_i, s_i) / (As_i, As_i)
        A->mult(s_hat, As);
        As_inner = As.inner_product(s);
        AsAs_inner = As.inner_product(As);
        omega = As_inner / AsAs_inner;

        // x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
        x.axpy(p_hat, alpha);
        x.axpy(s_hat, omega);

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
            printf("%d Iteration required to converge\n", iter-1);
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
            printf("%d Iteration required to converge\n", iter-1);
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
            printf("%d Iteration required to converge\n", iter-1);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

/**************************************************************************************
 BiCGStab with some inner products replaced with partial inner product approximations
 **************************************************************************************/
void PI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, MPI_Comm &inner_comm,
                 MPI_Comm &root_comm, double frac, int inner_color, int root_color, int inner_root, int procs_in_group,
                 int part_global, double tol, int max_iter)
{
    /*              A : ParCSRMatrix for system to solve
     *              x : ParVector solution to solve for
     *              b : ParVector rhs of system to solve
     *            res : vector containing residuals of each iteration
     *     inner_comm : communicator containing procs used in proc's partial inner product 
     *      root_comm : communicator containing roots for each inner_comm
     *    inner_color : int pertaining to which inner_comm each proc belongs 
     *     root_color : int - 0 if the proc is a root for an inner_comm
     *     inner_root : rank of proc that is the root for the processes' inner_comm
     * procs_in_group : average number of procs in each inner_comm
     *    part_global : number of values used in partial inner product
     *            tol : tolerance for convergence
     *       max_iter : maximum number of iterations
     */

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Create communicators for partial inner products
    int other_root;
    int my_ind;
    std::vector<int> roots;
    bool am_root;
    create_partial_inner_comm(inner_comm, root_comm, frac, x, inner_color, root_color, inner_root, procs_in_group, part_global);
    //create_partial_inner_comm_v2(inner_comm, root_comm, frac, x, my_ind, roots, am_root);
    if (inner_root == 0) other_root = num_procs/2;
    else other_root = 0;
    // Number of groups to iterate through for partial inner products
    int groups = 1 / frac;

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;

    int iter= 0;
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
    int group = 0;
    data_t As_old_half, AsAs_old_half;
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

        // Replace inner product with half inner for testing
        //As_inner = As.inner_product(s);
        //As_inner = partial_inner(inner_comm, root_comm, s, As, inner_color, group, inner_root, procs_in_group, part_global);
        As_inner = half_inner(inner_comm, s, As);
        //if (iter == 0) As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
        if (iter%2 == 0) As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
        //if (iter == 0) As_old_half = partial_inner_communicate(inner_comm, root_comm, As_inner, my_ind, roots, am_root);
        As_inner += As_old_half;

        // Replace single inner product with half inner for testing
        //AsAs_inner = As.inner_product(As);
        //AsAs_inner = partial_inner(inner_comm, root_comm, As, As, inner_color, (group+1)%groups, inner_root, procs_in_group, part_global);
        AsAs_inner = half_inner(inner_comm, As, As);
        //if (iter == 0) AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
        if (iter%2 == 0) AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
        //if (iter == 0) AsAs_old_half = partial_inner_communicate(inner_comm, root_comm, AsAs_inner, my_ind, roots, am_root);
        AsAs_inner += AsAs_old_half;
        
        //group = (group++) % groups;

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

// COMMUNICATE INNER PRODUCTS - UPDATE OLD OTHER HALF
        //if (iter > 0) {
        if (iter%2 > 0) {
            As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
            AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
            //As_old_half = partial_inner_communicate(inner_comm, root_comm, As_inner, my_ind, roots, am_root);
            //AsAs_old_half = partial_inner_communicate(inner_comm, root_comm, AsAs_inner, my_ind, roots, am_root);
        }

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
            printf("%d Iteration required to converge\n", iter-1);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}

/**************************************************************************************
 Preconditioned BiCGStab with some inner products replaced with partial inner product approximations
 **************************************************************************************/
void PrePI_BiCGStab(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, aligned_vector<double>& res, 
                 aligned_vector<double>& sAs_inner_list, aligned_vector<double>& AsAs_inner_list, MPI_Comm &inner_comm,
                 MPI_Comm &root_comm, double frac, int inner_color, int root_color, int inner_root, int procs_in_group,
                 int part_global, double tol, int max_iter)
{
    /*              A : ParCSRMatrix for system to solve
     *              x : ParVector solution to solve for
     *              b : ParVector rhs of system to solve
     *            res : vector containing residuals of each iteration
     *     inner_comm : communicator containing procs used in proc's partial inner product 
     *      root_comm : communicator containing roots for each inner_comm
     *    inner_color : int pertaining to which inner_comm each proc belongs 
     *     root_color : int - 0 if the proc is a root for an inner_comm
     *     inner_root : rank of proc that is the root for the processes' inner_comm
     * procs_in_group : average number of procs in each inner_comm
     *    part_global : number of values used in partial inner product
     *            tol : tolerance for convergence
     *       max_iter : maximum number of iterations
     */
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Create communicators for partial inner products
    int other_root;
    int my_ind;
    std::vector<int> roots;
    bool am_root;
    create_partial_inner_comm(inner_comm, root_comm, frac, x, inner_color, root_color, inner_root, procs_in_group, part_global);
    //create_partial_inner_comm_v2(inner_comm, root_comm, frac, x, my_ind, roots, am_root);
    if (inner_root == 0) other_root = num_procs/2;
    else other_root = 0;
    // Number of groups to iterate through for partial inner products
    int groups = 1 / frac;

    ParVector r;
    ParVector r_star;
    ParVector s;
    ParVector p;
    ParVector Ap;
    ParVector As;
    ParVector p_hat;
    ParVector s_hat;

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
    p_hat.resize(b.global_n, b.local_n, b.first_local);
    s_hat.resize(b.global_n, b.local_n, b.first_local);
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
    data_t As_old_half, AsAs_old_half;
    while (norm_r > tol && iter < max_iter)
    {
        // p_i = M^-1 p_i
        // Apply preconditioner
        p_hat.set_const_value(0.0);
        ml->cycle(p_hat, p);

        // alpha_i = (r_i, r*) / (Ap_i, r*)
        A->mult(p_hat, Ap);
        Apr_inner = Ap.inner_product(r_star);
        alpha = rr_inner / Apr_inner;

        // s_i = r_i - alpha_i * Ap_i
        s.copy(r);
        s.axpy(Ap, -1.0*alpha);

        // s_i = M^-1 s_i
        // Apply preconditioner
        s_hat.set_const_value(0.0);
        ml->cycle(s_hat, s);

        // omega_i = (As_i, s_i) / (As_i, As_i)
        A->mult(s_hat, As);
        //As_inner = As.inner_product(s);
        As_inner = half_inner(inner_comm, s, As);
        //if (iter == 0) As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
        if (iter%2 == 0) As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
        //if (iter%2 == 0) As_old_half = partial_inner_communicate(inner_comm, root_comm, As_inner, my_ind, roots, am_root);
        //if (iter == 0) As_old_half = partial_inner_communicate(inner_comm, root_comm, As_inner, my_ind, roots, am_root);
        As_inner += As_old_half;
        sAs_inner_list.push_back(As_inner);

        //AsAs_inner = As.inner_product(As);
        AsAs_inner = half_inner(inner_comm, As, As);
        //if (iter == 0) AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
        if (iter%2 == 0) AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
        //if (iter%2 == 0) AsAs_old_half = partial_inner_communicate(inner_comm, root_comm, AsAs_inner, my_ind, roots, am_root);
        //if (iter == 0) AsAs_old_half = partial_inner_communicate(inner_comm, root_comm, AsAs_inner, my_ind, roots, am_root);
        AsAs_inner += AsAs_old_half;
        AsAs_inner_list.push_back(AsAs_inner);

        omega = As_inner / AsAs_inner;

        // x_{i+1} = x_i + alpha_i * p_i + omega_i * s_i
        x.axpy(p_hat, alpha);
        x.axpy(s_hat, omega);

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
        
        //if (iter > 0) {
        if (iter%2) {
            As_old_half = half_inner_communicate(inner_comm, As_inner, inner_root, other_root);
            AsAs_old_half = half_inner_communicate(inner_comm, AsAs_inner, inner_root, other_root);
            //As_old_half = partial_inner_communicate(inner_comm, root_comm, As_inner, my_ind, roots, am_root);
            //AsAs_old_half = partial_inner_communicate(inner_comm, root_comm, AsAs_inner, my_ind, roots, am_root);
        }

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
            printf("%d Iteration required to converge\n", iter-1);
            printf("2 Norm of Residual: %lg\n\n", norm_r);
        }
    }

    return;
}
