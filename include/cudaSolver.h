#ifndef __CUDA_SOLVER_H__
#define __CUDA_SOLVER_H__
#include <cusparse_v2.h>

/*
 * CudaSolver: solves the linear system Ax = b, where A is sparse and square. Note that
 * if you pass a lower triangular system with ones on the diagonal of the LHS,
 * and don't call factor(), calling solve() will solve the system
 */
class CudaSolver {
  public:
    /*
     * constructor: allocates and populates device memory with the given LHS and RHS.
     * Accepts the LHS in COO format, but stores it internally in CSR format
     * ARGUMENTS
     * row_idx: row indices of the LHS
     * col_idx: column indices of LHS. Must be sorted
     * vals: call the LHS A. vals[i] = A[row_idx[i], col_idx[i]]
     * m: rows in LHS
     * nnz: number of nonzeros in LHS. row_idx, col_idx, and vals should each be of length nnz
     * b: dense RHS vector
     */
    CudaSolver(int *row_idx, int *col_idx, double *vals, int m, int nnz, double *b);

    /*
     * destructor: frees memory, destroys cusparse handle. TODO: by the rule of three, we should
     * write a copy constructor
     */
    ~CudaSolver();

    /*
     * factor: computes the incomplete LU factorization of the LHS matrix and stores it in place
     * of the LHS matrix
     */
    void factor();

    /*
     * get_factors: get the lower and upper triangular factors. Should be called after factor()
     * ARGUMENTS
     * row_ptr_L: CSR row pointer for L. Should be allocated to contain m + 1 ints
     * col_idx_L: CSR column indices for L. Should be allocated to contain nnz ints
     * vals_L: populated with the values of the lower triangular factor. Should be allocated
     * to contain nnz doubles
     * row_ptr_U: CSR row pointer for U. Should be allocated to contain m + 1 ints
     * col_idx_U: CSR column indices for U. Should be allocated to contain nnz ints
     * vals_U: populated with the values of the upper triangular factor. Should be allocated
     * to contain nnz doubles
     */
    void get_factors(int *row_ptr_L, int *col_idx_L, double *vals_L,
                     int *row_ptr_U, int *col_idx_U, double *vals_U);

    /*
     * solve: should only be called after factor(). Computes the solution to Ax = b
     * ARGUMENTS
     * x: populated with the solution. Should be allocated to contain m doubles, where
     * m is the number of rows in the LHS matrix
     */
    void solve(double *x);

  private:
    /*
     * lowerTriangularSolve: solves Lv = b and puts the result into b
     */
    void lowerTriangularSolve();

    /*
     * upperTriangularSolve: should be called after lowerTriangularSolve.
     * Solves Ux = v and puts the result into b.
     */
    void upperTriangularSolve();

    int m;
    int nnz;

    // Store the LHS matrix in CSR format. When factor() is called,
    // device_vals is populated with the LU factorization
    int *device_row_ptr;
    int *device_col_idx;
    double *device_vals;
    double *device_b;

    int *col_idx;
    double *vals;

    cusparseHandle_t cs_handle;
};

#endif
