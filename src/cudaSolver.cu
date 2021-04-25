#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>

#include "cudaSolver.h"

CudaSolver::CudaSolver(int *row_idx, int *col_idx, double *vals, int rows, int nonzeros, double *b, bool spd) {
  cusparseCreate(&cs_handle);

  cudaMalloc(&device_row_ptr, sizeof(int)*(m + 1));
  cudaMalloc(&device_col_idx, sizeof(int)*nnz);
  cudaMalloc(&device_vals, sizeof(double)*nnz);

  int *device_row_idx;
  cudaMalloc(&device_row_idx, sizeof(int)*nnz);
  cudaMemcpy(device_row_idx, row_idx, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(device_col_idx, col_idx, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(device_vals, vals, sizeof(double)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, sizeof(double)*m, cudaMemcpyHostToDevice);

  // Convert COO to CSR
  cusparseXcoo2csr(cs_handle, device_row_idx, nnz, m, device_row_ptr,
                   CUSPARSE_INDEX_BASE_ZERO);

  cudaFree(device_row_idx);

  use_cholesky = spd;
  m = m;
  nnz = nnz;

  lpop = false;
}

CudaSolver::~CudaSolver() {
  cusparseDestroy(cs_handle);
  cudaFree(device_row_ptr);
  cudaFree(device_col_idx);
  cudaFree(device_vals);
  cudaFree(device_b);

  if (lpop) {
    cudaFree(L_vals);
  }
}

void CudaSolver::factor() {
  // For the sake of getting things working, we'll just handle the spd case for now.
  // Later on, we can case on spd and choose Cholesky vs LU accordingly
  if (!use_cholesky) {
    return;
  }

  // Boilerplate
  csric02Info_t info;
  cusparseCreateCsric02Info(&info);

  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

  int bufferSize;
  cusparseDcsric02_bufferSize(cs_handle, m, nnz, descr, device_vals, device_row_ptr,
                              device_col_idx, info, &bufferSize);

  void *pBuffer;
  cudaMalloc(&pBuffer, bufferSize);

  // Analyze
  cusparseDcsric02_analysis(cs_handle, m, nnz, descr, device_vals,
                            device_row_ptr, device_col_idx, info,
                            CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

  // Factor, and put the Cholesky factor into L_vals
  cusparseDcsric02(cs_handle, m, nnz, descr, L_vals,
                   device_row_ptr, device_col_idx, info,
                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

  cudaFree(pBuffer);

  cusparseDestroyMatDescr(descr);

  cusparseDestroyCsric02Info(info);
  lpop = true;
}

void CudaSolver::solve(double *x) {
  // For the sake of getting things working, we'll just handle the spd case for now.
  // Later on, we can case on spd and choose Cholesky vs LU accordingly
  if (!use_cholesky) {
    return;
  }

  int *levelInd;
  int *levelPtr;
  int *chainPtr;
  int *rRoot;
  int *wRoot;
  int *cRoot;

  // Analysis phase
  /*
  dim3 blockDim(16, 16);
  kernelFindRoots<<<gridDim, blockDim>>>();
  */
}
