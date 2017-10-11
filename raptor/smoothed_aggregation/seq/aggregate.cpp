// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

Matrix* Matrix::aggregate()
{
    printf("This matrix format is not supported.\n");
    return NULL;
}

CSRMatrix* CSRMatrix::aggregate()
{
    if (n_rows == 0)
    {
        return new CSRMatrix(0, 0);
    }

    // Fill Aggregates Vector with 0 (to n_rows)
    std::vector<int> aggregates(n_rows, 0);
    
    // Clear anything in c_points vector and guess size
    std::vector<int> c_points(0.2*n_rows);

    // Number of aggs + 1
    int next_aggregate = 1;

    int row_start, row_end;
    int col, agg, agg_col;
    int ctr;

    // Pass 1 -- Create initial aggregates
    for (int row = 0; row < n_rows; row++)
    {
        // Check if already aggregated
        if (aggregates[row])
            continue;

        // Loop through neighbors, search for aggregated ones
        bool has_aggregated_neighbors = false;
        bool has_neighbors = false;

        row_start = idx1[row];
        row_end = idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = idx2[j];
            if (row != col)
            {
                has_neighbors = true;
                if (aggregates[col])
                {
                    has_aggregated_neighbors = true;
                    break;
                }
            }
        }

        // If no neighbors, isolated node so don't aggregate
        if (!has_neighbors)
        {
            aggregates[row] = -(n_rows);
        }

        // If no aggregated neighbors, create an aggregate
        else if (!has_aggregated_neighbors)
        {
            aggregates[row] = next_aggregate;
            c_points.push_back(row);
            for (int j = row_start; j < row_end; j++)
            {
                aggregates[idx2[j]] = next_aggregate;
            }
            next_aggregate++;
        }
    }

    // Pass 2 -- add unaggregated nodes to any neighbors
    for (int row = 0; row < n_rows; row++)
    {
        // If already aggregated, go to next row
        if (aggregates[row]) continue;

        row_start = idx1[row];
        row_end = idx1[row+1];

        for (int j = row_start; j < row_end; j++)
        {
            col = idx2[j];
            agg_col = aggregates[col];
            if (agg_col > 0)
            {
                aggregates[row] = -agg_col;
                break;
            }
        }
    }

    // Next Aggregate now equals num aggregates
    next_aggregate--;

    // Pass 3
    int neg_n_rows = -(n_rows);
    for (int row = 0; row < n_rows; row++)
    {
        agg = aggregates[row];

        // Already aggregated
        if (agg != 0)
        {
            // Regular aggregate, subtract so 0 indexed
            if (agg > 0)
            {
                aggregates[row] = agg - 1;
            }

            // Not to be aggregated
            else if (agg == neg_n_rows)
            {
                aggregates[row] = -1;
            }

            // Make regular aggregate (and 0 indexed)
            else
            {
                aggregates[row] = -(agg) - 1;
            }
            continue;
        }

        // Otherwise, not aggregated (so create new aggregate)
        aggregates[row] = next_aggregate;
        c_points[next_aggregate] = row;

        // Add unaggregated neighbors to new aggregate
        row_start = idx1[row];
        row_end = idx1[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = idx2[j];
            if (aggregates[col] == 0)
            {
                aggregates[col] = next_aggregate;
            }
        }
        next_aggregate++;
    }   

    // If no aggregates, return 0 matrix
    if (!next_aggregate)
    {
        CSRMatrix* T = new CSRMatrix(n_rows, 0);
        for (int i = 0; i < n_rows + 1; i++)
        {
            T->idx1[i] = 0;
        }
        return T;
    }

    // Add aggregates to CSRMatrix T, so that each row
    // has at most 1 nonzero, in column == aggregate[row]
    // If aggregate[row] is -1, leave a 0 row
    else
    {
        CSRMatrix* T = new CSRMatrix(n_rows, next_aggregate);
        T->idx2.reserve(T->n_rows);
        T->vals.reserve(T->n_rows);

        T->idx1[0] = 0;
        for (int i = 0; i < T->n_rows; i++)
        {
            col = aggregates[i];
            T->idx2.push_back(col);
            T->vals.push_back(1.0);
            T->idx1[i+1] = T->idx2.size();
        }
        T->nnz = T->idx2.size();

        return T;
    }
}

