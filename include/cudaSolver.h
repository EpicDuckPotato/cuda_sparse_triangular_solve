#ifndef __CUDA_SOLVER_H__
#define __CUDA_SOLVER_H__
#include <cusparse_v2.h>

/*
 * CudaSolver: solves the linear system Ax = b, where A is sparse and square
 */
class CudaSolver {
  public:
    /*
     * constructor: allocates and populates device memory with the given LHS and RHS.
     * Accepts the LHS in COO format, but stores it internally in CSR format.
     * ARGUMENTS
     * row_idx: row indices of the LHS
     * col_idx: column indices of LHS
     * vals: call the LHS A. vals[i] = A[row_idx[i], col_idx[i]]
     * rows: rows in LHS
     * nonzeros: number of nonzeros in LHS. row_idx, col_idx, and vals should each be of length nnz
     * b: dense RHS vector
     * spd: true if the LHS is symmetric positive definite
     */
    CudaSolver(int *row_idx, int *col_idx, double *vals, int rows, int nonzeros, double *b, bool spd);

    /*
     * destructor: frees memory, destroys cusparse handle. TODO: by the rule of three, we should
     * write a copy constructor
     */
    ~CudaSolver();

    /*
     * factor: factors the LHS matrix and stores it in device memory. If spd, this performs
     * the Cholesky factorization and stores the lower triangular factor. Otherwise,
     * performs the LU factorization, storing L and U
     */
    void factor();

    /*
     * solve: factors the LHS matrix into two triangular factors, then performs
     * upper and lower triangular solves. if spd was true in the constructor,
     * this function uses the Cholesky factorization. Otherwise it uses the LU
     * factorization
     * ARGUMENTS
     * x: populated with the solution. Should be allocated to contain m doubles, where
     * m is the number of rows in the LHS matrix
     */
    void solve(double *x);

  private:
    int m;
    int nnz;

    // Store the LHS matrix in CSR format
    int *device_row_ptr;
    int *device_col_idx;
    double *device_vals;

    // The Cholesky factor has the same row pointers and column indices. Only the
    // values are different
    double *L_vals;
    bool lpop; // This just indicates whether we need to free L_vals in the destructor

    double *device_b;

    bool use_cholesky;

    cusparseHandle_t cs_handle;
};

#endif
