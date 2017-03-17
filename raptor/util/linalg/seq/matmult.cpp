#include "core/matrix.hpp"

using namespace raptor;

void CSRMatrix::mult(const CSRMatrix& B, CSRMatrix* C)
{
    std::vector<int> next(n_cols, -1);
    std::vector<double> sums(n_cols, 0);

    C->nnz = 0;
    C->n_rows = n_rows;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_A = idx1[i];
        int row_end_A = idx1[i+1];
        for (int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = idx2[j];
            double val_A = vals[j];
            int row_start_B = B.idx1[col_A];
            int row_end_B = B.idx1[col_A+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.idx2[k];
                sums[col_B]+= val_A*B.vals[k];
                if (next[col_B] == -1)
                {
                    next[col_B] = head;
                    head = col_B;
                    length++;
                }
            }
        }
        for (int j = 0; j < length; j++)
        {
            if (fabs(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C->vals.push_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}

void CSCMatrix::mult(const CSCMatrix& B, CSCMatrix* C)
{
    std::vector<int> next(B.n_rows, -1);
    std::vector<double> sums(B.n_rows, 0);

    C->nnz = 0;
    C->n_rows = n_rows;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int row_BT = 0; row_BT < B.n_cols; row_BT++)
    {
        int head = -2;
        int length = 0;
        int row_start_BT = B.idx1[row_BT];
        int row_end_BT = B.idx1[row_BT+1];
        for (int j = row_start_BT; j < row_end_BT; j++)
        {
            int col_BT = B.idx2[j];
            double val_BT = B.vals[j];
            int row_start_AT = idx1[col_BT];
            int row_end_AT = idx1[col_BT+1];
            for (int k = row_start_AT; k < row_end_AT; k++)
            {
                int col_AT = idx2[k];
                sums[col_AT]+= val_BT*vals[k];
                if (next[col_AT] == -1)
                {
                    next[col_AT] = head;
                    head = col_AT;
                    length++;
                }
            }
        }
        for (int j = 0; j < length; j++)
        {
            if (fabs(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C->vals.push_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->idx1[row_BT+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}

void CSRMatrix::mult(const CSCMatrix& B, CSRMatrix* C)
{
    std::vector<double> row_vals(n_cols, 0);
    std::vector<int> next(n_cols, -1);
    C->nnz = 0;

    C->n_rows = n_rows;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        int head = -2;
        int length = 0;

        // Add row of A to dense vector (row_vals)
        int row_start = idx1[i];
        int row_end = idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col_A = idx2[j];
            row_vals[col_A] = vals[j];
            next[col_A] = head;
            head = col_A;
            length++;
        }

        for (int col_B = 0; col_B < B.n_cols; col_B++)
        {
            double sum = 0;
            int col_start_B = B.idx1[col_B];
            int col_end_B = B.idx1[col_B+1];
            for (int k = col_start_B; k < col_end_B; k++)
            {
                sum += B.vals[k] * row_vals[B.idx2[k]];
            }
            if (fabs(sum) > zero_tol)
            {
                C->idx2.push_back(col_B);
                C->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            row_vals[tmp] = 0;
        }

        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}


void CSRMatrix::mult(const CSCMatrix& B, CSCMatrix* C)
{
    std::vector<double> col_vals(B.n_rows, 0);
    std::vector<int> next(B.n_rows);
    C->nnz = 0;

    C->n_rows = n_rows;
    C->n_cols = B.n_cols;
    C->idx1.resize(B.n_cols + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int col_B = 0; col_B < B.n_cols; col_B++)
    {
        int head = -2;
        int length = 0;

        // Add row of A to dense vector (row_vals)
        int col_start = B.idx1[col_B];
        int col_end = B.idx1[col_B+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row_B = B.idx2[j];
            col_vals[row_B] = B.vals[j];
            next[row_B] = head;
            head = row_B;
            length++;
        }

        for (int row = 0; row < n_rows; row++)
        {
            double sum = 0;
            int row_start = idx1[row];
            int row_end = idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                sum += vals[k] * col_vals[idx2[k]];
            }
            if (fabs(sum) > zero_tol)
            {
                C->idx2.push_back(row);
                C->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }

        C->idx1[col_B+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}

void CSCMatrix::mult_T(const CSCMatrix& B, CSCMatrix* C)
{        
    std::vector<double> col_vals(B.n_rows, 0);
    std::vector<int> next(n_rows, -1);
    C->nnz = 0;

    C->n_rows = n_cols;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;

    for (int i = 0; i < B.n_cols; i++)
    {
        int head = -2; 
        int length = 0;

        // Add row of A to dense vector (col_vals)
        int col_start_B = B.idx1[i];
        int col_end_B = B.idx1[i+1];
        for (int j = col_start_B; j < col_end_B; j++)
        {
            int row_B = B.idx2[j];
            col_vals[row_B] = B.vals[j];
            next[row_B] = head;
            head = row_B;
            length++;
        }

        for (int row_AT = 0; row_AT < n_cols; row_AT++)
        {
            double sum = 0;
            int row_start_AT = idx1[row_AT];
            int row_end_AT = idx1[row_AT+1];
            for (int j = row_start_AT; j < row_end_AT; j++)
            {
                sum += vals[j] * col_vals[idx2[j]];
            }
            if (fabs(sum) > zero_tol)
            {
                C->idx2.push_back(row_AT);
                C->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            col_vals[tmp] = 0;
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}

void CSCMatrix::mult_T(const CSCMatrix& B, CSRMatrix* C)
{        
    std::vector<double> col_vals(n_rows, 0);
    std::vector<int> next(n_rows, -1);
    C->nnz = 0;

    C->n_rows = n_cols;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < n_cols; i++)
    {
        int head = -2;
        int length = 0;

        // Add row of A to dense vector (col_vals)
        int col_start = idx1[i];
        int col_end = idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row_A = idx2[j];
            col_vals[row_A] = vals[j];
            next[row_A] = head;
            head = row_A;
            length++;
        }

        for (int col_B = 0; col_B < B.n_cols; col_B++)
        {
            double sum = 0;
            int col_start_B = B.idx1[col_B];
            int col_end_B = B.idx1[col_B+1];
            for (int k = col_start_B; k < col_end_B; k++)
            {
                sum += B.vals[k] * col_vals[B.idx2[k]];
            }
            if (fabs(sum) > zero_tol)
            {
                C->idx2.push_back(col_B);
                C->vals.push_back(sum);
            }
        }

        for (int j = 0; j < length; j++)
        {
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            col_vals[tmp] = 0;
        }

        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}


void CSCMatrix::mult_T(const CSRMatrix& B, CSRMatrix* C)
{
    std::vector<int> next(n_rows, -1);
    std::vector<double> sums(n_rows, 0);

    C->nnz = 0;
    C->n_rows = n_cols;
    C->n_cols = B.n_cols;
    C->idx1.resize(n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < n_cols; i++)
    {
        int head = -2;
        int length = 0;
        int col_start_A = idx1[i];
        int col_end_A = idx1[i+1];
        for (int j = col_start_A; j < col_end_A; j++)
        {
            int row_A = idx2[j];
            double val_A = vals[j];
            int row_start_B = B.idx1[row_A];
            int row_end_B = B.idx1[row_A+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.idx2[k];
                sums[col_B]+= val_A*B.vals[k];
                if (next[col_B] == -1)
                {
                    next[col_B] = head;
                    head = col_B;
                    length++;
                }
            }
        }
        for (int j = 0; j < length; j++)
        {
            if (fabs(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C->vals.push_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}


/**************************************************************
*****   CSRMatrix-CSRMatrix Multiply (C = A*B)
**************************************************************
***** Multiplies the matrix times a matrix B, and writes the
***** result in matrix C->
*****
***** Parameters
***** -------------
***** B : CSRMatrix*
*****    Matrix by which to multiply the matrix 
***** C : Matrix*
*****    CSRMatrix in which to place solution
**************************************************************/
void Matrix::mult(const CSRMatrix& B, CSRMatrix* C)
{
    if (format() != CSR)
    {
        printf("Matrix should be CSR format\n");
        return;
    }
}

/**************************************************************
*****   CSRMatrix-CSCMatrix Multiply (C = A*B)
**************************************************************
***** Multiplies the matrix times a matrix B, and writes the
***** result in matrix C->
*****
***** Parameters
***** -------------
***** B : CSCMatrix*
*****    Matrix by which to multiply the matrix 
***** C : Matrix*
*****    CSRMatrix in which to place solution
**************************************************************/
void Matrix::mult(const CSCMatrix& B, CSRMatrix* C)
{
    if (format() != CSR)
    {
        printf("Matrix should be CSR format\n");
        return;
    }

}

/**************************************************************
*****   CSRMatrix-CSCMatrix Multiply (C = A*B)
**************************************************************
***** Multiplies the matrix times a matrix B, and writes the
***** result in matrix C->
*****
***** Parameters
***** -------------
***** B : CSCMatrix*
*****    Matrix by which to multiply the matrix 
***** C : Matrix*
*****    CSRMatrix in which to place solution
**************************************************************/
void Matrix::mult(const CSCMatrix& B, CSCMatrix* C)
{
    if (format() != CSR)
    {
        printf("Matrix should be CSR format\n");
        return;
    }

}
void Matrix::mult_T(const CSCMatrix& B, CSCMatrix* C)
{        
    if (format() != CSC)
    {
        printf("Matrix should be CSC format\n");
        return;
    }
}

void Matrix::mult_T(const CSCMatrix& B, CSRMatrix* C)
{        
    if (format() != CSC)
    {
        printf("Matrix should be CSC format\n");
        return;
    }
}


void Matrix::mult_T(const CSRMatrix& B, CSRMatrix* C)
{
    if (format() != CSC)
    {
        printf("Matrix should be CSC format\n");
        return;
    }
}

void Matrix::RAP(const CSCMatrix& P, CSRMatrix* C)
{
    if (format() != CSR)
    {
        printf("Only implemented for CSR matrices!\n");
        return;
    }

    // Store a column of P, to multiply with each row of A
    std::vector<double> col_vals(P.n_rows, 0);
    std::vector<int> next(P.n_rows);

    // Store a column of AP after it is computed
    // to then multiply with each row of PT
    std::vector<double> col_AP(n_rows, 0);
    std::vector<int> next_AP(n_rows);

    // Keep track of number of nonzeros added
    // to each row (for converting to CSR)
    std::vector<int> row_n(P.n_cols, 0);

    // CSC Matrix for temporarily holding result before
    // converted to CSR and added to C
    CSCMatrix Ctmp;
    Ctmp.n_cols = P.n_cols;
    Ctmp.idx1.resize(P.n_cols+1);
    Ctmp.idx2.reserve(nnz);
    Ctmp.vals.reserve(nnz);

    Ctmp.idx1[0] = 0;
    for (int col_P = 0; col_P < P.n_cols; col_P++)
    {
        int head = -2;
        int length = 0;

        int head_AP = -2;
        int length_AP = 0;

        // Add row of A to dense vector (col_vals)
        int col_start_P = P.idx1[col_P];
        int col_end_P = P.idx1[col_P+1];
        for (int j = col_start_P; j < col_end_P; j++)
        {
            int row_P = P.idx2[j];
            col_vals[row_P] = P.vals[j];
            next[row_P] = head;
            head = row_P;
            length++;
        }

        // Multiply each row of A by col_vals
        // Store sums in col_AP
        for (int row = 0; row < n_rows; row++)
        {
            double sum = 0;
            int row_start = idx1[row];
            int row_end = idx1[row+1];
            for (int j = row_start; j < row_end; j++)
            {
                sum += vals[j] * col_vals[idx2[j]];
            }
            if (fabs(sum) > zero_tol)
            {
                col_AP[row] = sum;
                next_AP[row] = head_AP;
                head_AP = row;
                length_AP++;
            }
        }

        // Multiply each row of PT by col_AP 
        // Add nonzero sums to Ctmp
        for (int row_PT = 0; row_PT < P.n_cols; row_PT++)
        {
            double sum = 0;
            int row_start_PT = P.idx1[row_PT];
            int row_end_PT = P.idx1[row_PT+1];
            for (int j = row_start_PT; j < row_end_PT; j++)
            {
                sum += P.vals[j] * col_AP[P.idx2[j]];
            }
            if (fabs(sum) > zero_tol)
            {
                Ctmp.idx2.push_back(row_PT);
                Ctmp.vals.push_back(sum);
                row_n[row_PT]++;
            }
        }

        // Clear AP vectors
        for (int j = 0; j < length_AP; j++)
        {
            col_AP[head_AP] = 0;
            head_AP = next_AP[head_AP];
        }

        // Clear vectors
        for (int j = 0; j < length; j++)
        {
            col_vals[head] = 0;
            head = next[head];
        }
         
        Ctmp.idx1[col_P+1] = Ctmp.idx2.size();
    }


    // Now we have a CSC matrix Ctmp, but we want to 
    // put solution in CSR matrix C
    // Initialize known dimensions of C
    C->nnz = Ctmp.idx2.size();
    C->n_rows = P.n_cols;
    C->n_cols = P.n_cols;
    C->idx1.resize(P.n_cols + 1);
    C->idx2.resize(C->nnz);
    C->vals.resize(C->nnz);

    // Initialize row point idx1
    C->idx1[0] = 0;
    for (int i = 0; i < P.n_cols; i++)
    {
        C->idx1[i+1] = C->idx1[i] + row_n[i];
        row_n[i] = 0;
    }

    // All values and column indices to appropriate positions
    for (int i = 0; i < P.n_cols; i++)
    {
        int col_start = Ctmp.idx1[i];
        int col_end = Ctmp.idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = Ctmp.idx2[j];
            double value = Ctmp.vals[j];

            int idx = C->idx1[row] + row_n[row]++;
            C->idx2[idx] = i;
            C->vals[idx] = value;
        }
    }
}

void Matrix::RAP(const CSCMatrix& P, CSCMatrix* C)
{
    if (format() != CSC)
    {
        printf("Only implemented for CSC matrices!\n");
        return;
    }

    std::vector<int> next(n_cols, -1);
    std::vector<double> sums(n_cols, 0);

    C->nnz = 0;

    // Dimensions from P^(T)AP
    C->n_rows = P.n_cols;
    C->n_cols = P.n_cols;
    C->idx1.resize(C->n_rows + 1);
    C->idx2.clear();
    C->vals.clear();
    C->idx2.reserve(nnz);
    C->vals.reserve(nnz);

    C->idx1[0] = 0;
    for (int col_P = 0; col_P < P.n_cols; col_P++)
    {
        int head = -2;
        int length = 0;
        int col_start_P = P.idx1[col_P];
        int col_end_P = P.idx1[col_P+1];
        for (int i = col_start_P; i < col_end_P; i++)
        {
            int row_P = P.idx2[i];
            double val_P = P.vals[i];
            int col_start_A = idx1[row_P];
            int col_end_A = idx1[row_P+1];
            for (int j = col_start_A; j < col_end_A; j++)
            {
                int row_A = idx2[j];
                sums[row_A] += val_P * vals[j];
                if (next[row_A] == -1)
                {
                    next[row_A] = head;
                    head = row_A;
                    length++;
                }
            }
        }

        // Vals in col i of (AP) are stored in sums
        for (int row_PT = 0; row_PT < P.n_cols; row_PT++)
        {
            int row_start_PT = P.idx1[row_PT];
            int row_end_PT = P.idx1[row_PT+1];
            double sum = 0;
            for (int i = row_start_PT; i < row_end_PT; i++)
            {
                int col_PT = P.idx2[i];
                sum += P.vals[i] * sums[col_PT];
            }
            if (fabs(sum) > zero_tol)
            {
                C->idx2.push_back(row_PT);
                C->vals.push_back(sum);
            }
        }

        //
        for (int j = 0; j < length; j++)
        {
            if (fabs(sums[head]) > zero_tol)
            {
                int tmp = head;
                head = next[head];
                next[tmp] = -1;
                sums[tmp] = 0;
            }
        }
        C->idx1[col_P + 1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();
}


