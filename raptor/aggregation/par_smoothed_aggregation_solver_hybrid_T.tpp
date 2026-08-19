// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HYBRID_T_TPP
#define RAPTOR_PAR_SMOOTHED_AGGREGATION_SOLVER_HYBRID_T_TPP

// extension of ParSmoothedAggregationSolver_T
// for hybrid type hierarchy
// fine level ParCSRMatrix + coarse levels ParBSRMatrix
namespace raptor
{
    template <class T, is_bsr_or_csr<T> matrix_type>
    void ParSmoothedAggregationSolver_T<T, matrix_type>::extend_bsr_hierarchy()
    {
        ParCSRMatrix* S;
        ParBSRMatrix* tentative;
        ParBSRMatrix* P = nullptr;
        ParBSRMatrix* AP;
        auto* fine = coarse_levels.back();
        ParBSRMatrix* A = fine->A;

        std::vector<int> states;
        std::vector<int> off_proc_states;
        std::vector<int> aggregates;
        std::vector<double> R;
        int n_aggs = 0;

        // Form strength of connection
        S = A->strength(this->strength_type, this->strong_threshold, false, 
                1, nullptr);

        // Aggregate Nodes
        switch (agg_type)
        {
            case MIS:
                mis2(S, states, off_proc_states, false, this->weights);
                n_aggs = aggregate(A, S, states, off_proc_states, 
                        aggregates);
                break;
            default:
                mis2(S, states, off_proc_states, false, this->weights);
                n_aggs = aggregate(A, S, states, off_proc_states, 
                        aggregates);
                break;
        }

        // Form tentative interpolation
        tentative = fit_candidates(A, n_aggs, aggregates, B, R, 
                num_candidates, false, interp_tol);
        
        switch (prolong_type)
        {
            case JacobiProlongation:
                P = jacobi_prolongation(A, tentative, false, 
                        prolong_weight, prolong_smooth_steps, prolongation_weighting::local);
                break;
            case BlockJacobiProlongation:
                P = jacobi_prolongation(A, tentative, false, 
                        prolong_weight, prolong_smooth_steps, prolongation_weighting::block);
                break;
            case BlockJacobiSpectralProlongation:
                P = jacobi_prolongation(A, tentative, false, 
                        prolong_weight, prolong_smooth_steps, prolongation_weighting::block_spectral);
                break;
            default:
                throw std::invalid_argument("invalid block prolongation option");
        }
        fine->P = P;

        // Form coarse bsr grid operator
        auto* coarse = new ParLevel_T<ParBSRMatrix>();
        AP = A->mult(P);
        ParBSRMatrix* Ac = AP->mult_T(P);
        coarse->A = Ac;
        delete Ac->comm;
        Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map,
                Ac->on_proc_column_map, A->comm->key, A->comm->mpi_comm);

        const int global_scalar_num_rows = total_global_num_rows(*Ac);
        const int local_scalar_num_rows = total_local_num_rows(*Ac);
        coarse->x.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse->b.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse->tmp.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse->P = nullptr;

        coarse_levels.emplace_back(coarse);

        update_B(B, R, num_candidates, total_local_num_rows(*Ac));

        delete AP;
        delete tentative;
        delete S;
    }

    // first level ParCSRMatrix
    // coarse levels ParBSRMatrix
    template <class T, is_bsr_or_csr<T> matrix_type>
    void ParSmoothedAggregationSolver_T<T, matrix_type>::setup_hybrid_helper(ParCSRMatrix*Af)
    {
        if (Af->global_num_rows <= this->max_coarse || (this->max_levels != -1 && this->max_levels <= 1))
        {
            this->setup_helper(Af);
            return;
        }

        if (this->strength_type == Symmetric)
        {
            throw std::invalid_argument("Hybrid T ParBSRMatrix hierarchy doesn't support Symmetric strength yet");
        }

        int rank, num_procs;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

        if (this->track_times)
        {
            this->setup_times = new double[5 * this->max_levels]();
            init_profile();
        }

        // Add original, fine level to hierarchy
        auto* level_0 = new ParLevel_T<ParCSRMatrix>();
        level_0->A = Af->copy();
        level_0->A->sort();
        level_0->A->on_proc->move_diag();
        level_0->x.resize(Af->global_num_rows, Af->local_num_rows);
        level_0->b.resize(Af->global_num_rows, Af->local_num_rows);
        level_0->tmp.resize(Af->global_num_rows, Af->local_num_rows);
        if (this->tap_amg == 0)
        {
            if (!Af->tap_comm && !Af->tap_mat_comm)
            {
                level_0->A->init_tap_communicators();
            }
            else if (!Af->tap_comm) // 3-step NAPComm
            {
                level_0->A->tap_comm = new TAPComm(Af->partition,
                        Af->off_proc_column_map, Af->on_proc_column_map);
            }
            else if (!Af->tap_mat_comm) // 2-step NAPComm
            {
                level_0->A->tap_mat_comm = new TAPComm(Af->partition,
                        Af->off_proc_column_map, Af->on_proc_column_map, false);
            }
        }

        if (this->weights == nullptr)
        {
            this->form_rand_weights(Af->local_num_rows, Af->partition->first_local_row);
        }

        // build csr level 0
        const bool tap_fine = this->tap_amg == 0;
        ParCSRMatrix* A = level_0->A;
        ParCSRMatrix* S;
        ParCSRMatrix* tentative;
        ParCSRMatrix* P = nullptr;
        ParCSRMatrix* AP;

        std::vector<int> states;
        std::vector<int> off_proc_states;
        std::vector<int> aggregates;
        std::vector<double> R;
        int n_aggs = 0;

        // Form strength of connection
        S = A->strength(this->strength_type, this->strong_threshold, tap_fine, 
                1, nullptr);

        // Aggregate Nodes
        switch (this->agg_type)
        {
            case MIS:
                mis2(S, states, off_proc_states, tap_fine, this->weights);
                n_aggs = aggregate(A, S, states, off_proc_states, 
                        aggregates, tap_fine);
                break;
            default:
                mis2(S, states, off_proc_states, tap_fine, this->weights);
                n_aggs = aggregate(A, S, states, off_proc_states, 
                        aggregates, tap_fine);
                break;
        }


        // Form tentative interpolation
        tentative = fit_candidates(A, n_aggs, aggregates, B, R, 
                num_candidates, false, interp_tol);
        
        P = jacobi_prolongation(A, tentative, tap_fine, 
                        prolong_weight, prolong_smooth_steps);
        level_0->P = P;
        this->levels.emplace_back(level_0);

        // Form coarse bsr grid operator
        auto* coarse_level_0 = new ParLevel_T<ParBSRMatrix>();
        AP = A->mult(P, tap_fine);
        ParCSRMatrix* Ac_csr = AP->mult_T(P, tap_fine);
        ParBSRMatrix* Ac =Ac_csr->to_ParBSR(num_candidates, num_candidates);
        coarse_level_0->A = Ac;
        delete Ac->comm;
        Ac->comm = new ParComm(Ac->partition, Ac->off_proc_column_map,
                Ac->on_proc_column_map, Af->comm->key,
                Af->comm->mpi_comm);

        int global_strength_nnz;
        int global_num_aggs;
        RAPtor_MPI_Allreduce(&S->local_nnz, &global_strength_nnz, 1,
                RAPtor_MPI_INT, RAPtor_MPI_SUM, A->comm->mpi_comm);
        RAPtor_MPI_Allreduce(&n_aggs, &global_num_aggs, 1,
                RAPtor_MPI_INT, RAPtor_MPI_SUM, A->comm->mpi_comm);
        if (rank == 0)
        {
            std::printf("CSR hybrid fine diagnostic: strength_rows=%d "
                    "strength_nnz=%d aggregates=%d "
                    "first_coarse_size=%d\n",
                    S->global_num_rows, global_strength_nnz,
                    global_num_aggs, total_global_num_rows(*Ac));
        }

        const int global_scalar_num_rows = total_global_num_rows(*Ac);
        const int local_scalar_num_rows = total_local_num_rows(*Ac);
        coarse_level_0->x.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse_level_0->b.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse_level_0->tmp.resize(global_scalar_num_rows,local_scalar_num_rows);
        coarse_level_0->P = nullptr;

        coarse_levels.emplace_back(coarse_level_0);

        update_B(B, R, num_candidates, total_local_num_rows(*Ac));

        delete AP;
        delete tentative;
        delete S;
        delete Ac_csr;

        if (this->track_times)
        {
            finalize_profile();
            this->setup_times[0] = total_t;
            this->setup_times[1] = collective_t;
            this->setup_times[2] = p2p_t;
            this->setup_times[3] = vec_t;
            this->setup_times[4] = mat_t;
            init_profile();
        }

        int coarse_level = 0;
        // Add coarse +1 levels to hierarchy 
        while (coarse_levels[coarse_level]->A->global_num_rows > this->max_coarse && 
                (this->max_levels == -1 || (int) coarse_levels.size() < this->max_levels-1))
        {
            extend_bsr_hierarchy();

            if (this->track_times)
            {
                finalize_profile();
                this->setup_times[5*(coarse_level+1)] = total_t;
                this->setup_times[5*(coarse_level+1) + 1] = collective_t;
                this->setup_times[5*(coarse_level+1) + 2] = p2p_t;
                this->setup_times[5*(coarse_level+1) + 3] = vec_t;
                this->setup_times[5*(coarse_level+1) + 4] = mat_t;
                init_profile();
            }

            coarse_level++;
        }
        // + 1 fine 0-level
        this->num_levels = coarse_levels.size() + 1;

        // cache block inverse for block jacobi
        for (std::size_t l = 0; l + 1 < coarse_levels.size(); l++)
        {
            coarse_levels[l]->block_diag_inv = compute_block_diag_inv(coarse_levels[l]->A);
        }

        if (Af->local_num_rows) 
        {
            delete[] this->weights;
            this->weights = nullptr;
        }

        // Duplicate coarsest level across all processes that hold any
        // rows of A_c
        this->duplicate_coarse(coarse_levels.back()->A);

        if (this->track_times)
        {
            finalize_profile();
            this->setup_times[5*(this->num_levels-1)] += total_t;
            this->setup_times[5*(this->num_levels-1) + 1] += collective_t;
            this->setup_times[5*(this->num_levels-1) + 2] += p2p_t;
            this->setup_times[5*(this->num_levels-1) + 3] += vec_t;
            this->setup_times[5*(this->num_levels-1) + 4] += mat_t;

            this->solve_times = new double[5 * this->num_levels]();
        }
    } 

    template <class T, is_bsr_or_csr<T> matrix_type>
    void ParSmoothedAggregationSolver_T<T, matrix_type>::bsr_cycle(ParVector& x, ParVector& b, int level)
    {
        const bool tap_level = false;
        if (this->solve_times)
        {
            init_profile();
        }

        int coarse_level = level - 1;
        ParBSRMatrix* A = coarse_levels[coarse_level]->A;
        ParBSRMatrix* P = coarse_levels[coarse_level]->P;
        ParVector& tmp = coarse_levels[coarse_level]->tmp;

        if (level == this->num_levels - 1)
        {
            if (A->local_num_rows)
            {
                int active_rank;
                RAPtor_MPI_Comm_rank(this->coarse_comm, &active_rank);

                std::vector<double> b_data(this->coarse_n);
                RAPtor_MPI_Allgatherv(b.local.data(), b.local_n, RAPtor_MPI_DOUBLE, b_data.data(), 
                        this->coarse_sizes.data(), this->coarse_displs.data(), 
                        RAPtor_MPI_DOUBLE, this->coarse_comm);

                // x = A_coarse_pinv * b_data
                const int first_local = this->coarse_displs[active_rank];

                for (int i = 0; i < x.local_n; i++)
                {
                    const int global_i = first_local + i;
                    double value = 0.0;

                    for (int j = 0; j < this->coarse_n; j++)
                    {
                        value += this->A_coarse_pinv[global_i*this->coarse_n + j] * b_data[j];
                    }

                    x.local[i] = value;
                }

            }

            if (this->solve_times)
            {
                finalize_profile();
                this->solve_times[5*level] += total_t;
                this->solve_times[5*level + 1] += collective_t;
                this->solve_times[5*level + 2] += p2p_t;
                this->solve_times[5*level + 3] += vec_t;
                this->solve_times[5*level + 4] += mat_t;
            }
        }
        else
        {
            coarse_levels[coarse_level+1]->x.set_const_value(0.0);
            
            // Relax
            relax_bsr(*coarse_levels[coarse_level], x, b);

            A->residual(x, b, tmp, tap_level);

            P->mult_T(tmp, coarse_levels[coarse_level+1]->b, tap_level);


            if (this->solve_times)
            {
                finalize_profile();
                this->solve_times[5*level] += total_t;
                this->solve_times[5*level + 1] += collective_t;
                this->solve_times[5*level + 2] += p2p_t;
                this->solve_times[5*level + 3] += vec_t;
                this->solve_times[5*level + 4] += mat_t;
            }
            bsr_cycle(coarse_levels[coarse_level+1]->x, coarse_levels[coarse_level+1]->b, level+1);
            if (this->solve_times)
            {
                init_profile();
            }


            P->mult_append(coarse_levels[coarse_level+1]->x, x, tap_level);

            relax_bsr(*coarse_levels[coarse_level], x, b);

            if (this->solve_times)
            {
                finalize_profile();
                this->solve_times[5*level] += total_t;
                this->solve_times[5*level + 1] += collective_t;
                this->solve_times[5*level + 2] += p2p_t;
                this->solve_times[5*level + 3] += vec_t;
                this->solve_times[5*level + 4] += mat_t;
            }
        }
    }

    template <class T, is_bsr_or_csr<T> matrix_type>
    void ParSmoothedAggregationSolver_T<T, matrix_type>::hybrid_cycle(ParVector& x, ParVector& b)
    {
        if (this->solve_times)
        {
            init_profile();
        }
        auto* fine = this->levels[0];
        auto* coarse = coarse_levels[0];

        const bool tap_fine = this->tap_amg == 0;

        coarse->x.set_const_value(0.0);

        this->relax(0, fine->A, x, b, fine->tmp, tap_fine);

        fine->A->residual(x, b, fine->tmp, tap_fine);

        fine->P->mult_T(fine->tmp,coarse->b,tap_fine);

        if (this->solve_times)
        {
            finalize_profile();
            this->solve_times[0] += total_t;
            this->solve_times[1] += collective_t;
            this->solve_times[2] += p2p_t;
            this->solve_times[3] += vec_t;
            this->solve_times[4] += mat_t;
        }

        bsr_cycle(coarse->x, coarse->b, 1);

        if (this->solve_times)
        {
            init_profile();
        }

        fine->P->mult_append(coarse->x, x, tap_fine);

        this->relax(0, fine->A, x, b, fine->tmp, tap_fine);

        if (this->solve_times)
        {
            finalize_profile();
            this->solve_times[0] += total_t;
            this->solve_times[1] += collective_t;
            this->solve_times[2] += p2p_t;
            this->solve_times[3] += vec_t;
            this->solve_times[4] += mat_t;
        }
    }
}

#endif
