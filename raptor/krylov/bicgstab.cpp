// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/bicgstab.hpp"

using namespace raptor;

void BiCGStab(CSRMatrix* A, Vector& x, Vector& b, std::vector<double>& res, double tol, int max_iter)
{
    Vector r;
    Vector rstar;
    Vector p;
    Vector s;
    Vector Ap;
    Vector As;
    
    int iter;
    data_t alpha, beta, omega;
    data_t rrstar_inner, next_rrstar_inner, Apr_inner, As_inner, AsAs_inner;
    double norm_r;

    if (max_iter <= 0)
    {
        max_iter = ((int)(1.3*b.size())) + 5;
    }

    // Fixed Constructors
    r.resize(b.size());
    rstar.resize(b.size());
    s.resize(b.size());
    p.resize(b.size());
    Ap.resize(b.size());
    As.resize(b.size());

    // r0 = b - A * x0
    A->residual(x, b, r);
    // Find initial residual
    norm_r = r.norm(2);
    res.push_back(norm_r);

    // rstar0 = r0
    rstar.copy(r);

    // p0 = r0
    p.copy(r);

    // Find initial (r, rstar)
    rrstar_inner = rstar.inner_product(r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    iter = 0;
    
    // Main BiCGStab Loop
    while (norm_r > tol && iter < max_iter)
    {

        // alpha_i = (r_i, rstar_i) / (A*p_i, pstar_i)
        A->mult(p, Ap);
        Apr_inner = Ap.inner_product(rstar);
        alpha = rrstar_inner / Apr_inner;
	//alpha = rrstar_inner / Ap.inner_product(rstar);
        
        // s_{i} = r_i - alpha_i * Ap_i
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

        // beta_i = (r_{i+1}, rstar) / (r_i, rstar)
        next_rrstar_inner = rstar.inner_product(r);
        beta = (next_rrstar_inner / rrstar_inner) * (alpha / omega);
        // Update next inner product
	rrstar_inner = next_rrstar_inner;

        // p_{i+1} = r_{i+1} + beta_i * (p_i - omega_i * Ap_i)
	p.axpy(Ap, -1.0*omega);
        p.scale(beta);
        p.axpy(r, 1.0);
       
        norm_r = r.norm(2);

        res.push_back(norm_r);

        iter++;
    }

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

    return;
}

