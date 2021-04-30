#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "cudaSolver.h"
#include <malloc.h>

#define THREADS_PER_BLOCK 256

struct GlobalConstants {
  int *row_ptr;
  int *col_idx;
  int m;
  int nnz;
};

__constant__ GlobalConstants cuConstSolverParams;

/*
 * kernelFindRoots: parallelizes over rows of the dependency
 * graph and indicates roots
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRoots(int *roots, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m) {
    int rowStart = cuConstSolverParams.row_ptr[row];

    // There's a - 1 because the last element of the row
    // is the diagonal element, which isn't a dependency
    // for solving this row
    int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

    roots[row] = 1;
    for (int i = rowStart; i < rowEnd; ++i) {
      if (depGraph[i]) {
        // Dependency exists
        roots[row] = 0;
        break;
      }
    }
  }
}

/*
 * kernelFindRootsInCandidates: parallelizes over rows of the dependency
 * graph and indicates roots, only looking at rows given by cRoot
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * cRoot: 0-1 array indicating candidates
 * nCand: number of candidates
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsInCandidates(int *roots, char *cRoot, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (cRoot[row] == 1) {
    int rowStart = cuConstSolverParams.row_ptr[row];

    // There's a - 1 because the last element of the row
    // is the diagonal element, which isn't a dependency
    // for solving this row
    int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

    roots[row] = 1;
    for (int i = rowStart; i < rowEnd; ++i) {
      if (depGraph[i]) {
        // Dependency exists
        roots[row] = 0;
        break;
      }
    }
  } else {
    roots[row] = 0;
  }
}

/*
 * kernelAnalyze: populates levelInd, levelPtr, and cRoot.
 * chainPtr determines the properties and number of kernels to be launched in the solve phase.
 * ARGUMENTS
 * cRoot: candidates are indicated with a 1. We set the rows of any current roots to 0
 * levelInd: sorted rows belonging to each level. We add rows on this level to the end
 * levelPtr: starting indices (in levelInd) of each level.
 * We add the starting index of the NEXT level to the end. If level == 0, we also
 * make levelPtr[0] = 0
 * nRoots: populated with number of roots
 * rootScan: inclusive scan of the 0-1 array indicating roots
 * rowsDone: how many rows have we already added to levelInd
 * (i.e. at what index in levelInd should we start adding things)?
 * level: what level is this?
 * depGraph: value array for the dependency graph
 */
__global__ void kernelAnalyze(char *cRoot, int *levelInd, int *levelPtr, int *nRoots,
                              int *rootScan, int rowsDone, int level, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m &&
      ((row == 0 && rootScan[row] == 1) ||
       (row > 0 && rootScan[row] == rootScan[row - 1] + 1))) {
    levelInd[rowsDone + rootScan[row] - 1] = row;
    cRoot[row] = 0;

    // Eliminate dependencies
    for (int i = 0; i < cuConstSolverParams.nnz; ++i) {
      // TODO: there's got to be a better way of doing this loop
      if (cuConstSolverParams.col_idx[i] == row && cuConstSolverParams.row_ptr[row + 1] != i + 1) {
        depGraph[i] = 0;
      }
    }
  }
  if (level == 0) {
    levelPtr[level] = 0;
  }
  levelPtr[level + 1] = rowsDone + rootScan[cuConstSolverParams.m - 1];
  *nRoots = rootScan[cuConstSolverParams.m - 1];
}

/*
 * kernelMultiblock: processes a single level
 * ARGUMENTS
 * start: start of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelMultiblock(int start, int *levelInd, int *levelPtr, double *b, double *val) {
  int startIdx = levelPtr[start];
  int endIdx = levelPtr[start + 1];

  int idx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < endIdx) {
    // Compute element of solution corresponding to row
    int row = levelInd[idx];
    int rowStart = cuConstSolverParams.row_ptr[row];
    int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

    for (int i = rowStart; i < rowEnd; ++i) {
      b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
    }
    b[row] /= val[rowEnd];
  }
}

/*
 * kernelSingleblock: processes a chain
 * ARGUMENTS
 * start: start of chain
 * end: end of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelSingleblock(int start, int end, int *levelInd, int *levelPtr, double *b, double *val) {
  int startIdx;
  int endIdx;

  for (int i = start; i < end; ++i) {
    startIdx = levelPtr[i];
    endIdx = levelPtr[i + 1];

    int idx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < endIdx) {
      // Compute element of solution corresponding to row
      int row = levelInd[idx];
      int rowStart = cuConstSolverParams.row_ptr[row];
      int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

      for (int i = rowStart; i < rowEnd; ++i) {
        b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
      }
      b[row] /= val[rowEnd];
    }
    __syncthreads();
  }
}

CudaSolver::CudaSolver(int *row_idx, int *col_idx, double *vals, int m, int nnz, double *b, bool spd, bool is_lt) : m(m), nnz(nnz), spd(spd), is_lt(is_lt) {
  cusparseCreate(&cs_handle);

  cudaMalloc(&device_row_ptr, sizeof(int)*(m + 1));
  cudaMalloc(&device_col_idx, sizeof(int)*nnz);
  cudaMalloc(&device_vals, sizeof(double)*nnz);
  cudaMalloc(&device_b, sizeof(double)*m);

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

  if (is_lt) {
    cudaMalloc(&L_vals, sizeof(double)*nnz);
    cudaMemcpy(L_vals, vals, sizeof(double)*nnz, cudaMemcpyHostToDevice);
  }
  lpop = is_lt;
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
  if (!spd) {
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
  lowerTriangularSolve();

  if (!is_lt) {
    upperTriangularSolve();
  }
  cudaMemcpy(x, device_b, m*sizeof(double), cudaMemcpyDeviceToHost);
}

void CudaSolver::lowerTriangularSolve() {
  // For the sake of getting things working, we'll just handle the spd case for now.
  // Later on, we can case on spd and choose Cholesky vs LU accordingly
  if (!spd && !is_lt) {
    return;
  }

  // We'll need to access row_ptr and col_idx quite often without modifying them,
  // so store them as global constants
  GlobalConstants params;
  params.row_ptr = device_row_ptr;
  params.col_idx = device_col_idx;
  params.m = m;
  params.nnz = nnz;
  cudaMemcpyToSymbol(cuConstSolverParams, &params, sizeof(GlobalConstants));

  int *levelInd;
  cudaMalloc(&levelInd, m*sizeof(int));

  int *levelPtr;
  cudaMalloc(&levelPtr, (m + 1)*sizeof(int)); // Worst-case scenario, each level contains a single row, and we need a pointer to the end, so m + 1

  // We can have max THREADS_PER_BLOCK rows in a chain, so there can be max
  // (m + THREADS_PER_BLOCK)/THREADS_PER_BLOCK chains (accounting for integer division)
  int *chainPtr = (int*)malloc(sizeof(int)*(m + THREADS_PER_BLOCK)/THREADS_PER_BLOCK + 1);

  int *rRoot;
  cudaMalloc(&rRoot, m*sizeof(int)); // The maximum number of roots is the number of rows
  cudaMemset(rRoot, 0, m*sizeof(int));

  int *wRoot;
  cudaMalloc(&wRoot, m*sizeof(int));
  cudaMemset(wRoot, 0, m*sizeof(int));

  char *cRoot;
  cudaMalloc(&cRoot, m*sizeof(char));
  cudaMemset(cRoot, 1, m*sizeof(char)); // Everything's a candidate at first

  int *nRoots;
  cudaMalloc(&nRoots, sizeof(int));

  // Sparse binary matrix with the same row pointers and column indices
  // as the LHS. If a row contains all zeros, the corresponding row of the solution
  // has no dependencies, and is therefore a root
  char *depGraph;
  cudaMalloc(&depGraph, nnz*sizeof(char));
  cudaMemset(depGraph, 1, nnz*sizeof(char));

  // ANALYSIS PHASE

  // Finding roots parallelizes over rows, so we have 1D blocks
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((m + blockDim.x - 1) / blockDim.x);


  // Get 0-1 array of roots
  kernelFindRoots<<<gridDim, blockDim>>>(rRoot, depGraph);
  cudaDeviceSynchronize();
  printf("found roots\n");
  thrust::inclusive_scan(thrust::device_pointer_cast(rRoot),
                         thrust::device_pointer_cast(rRoot) + m,
                         thrust::device_pointer_cast(rRoot));
  printf("scanned roots\n");

  int nRoots_host = 0;
  int level = 0;
  int rowsDone = 0;
  int rowsInChain = 0;
  int chainIdx = 0;
  chainPtr[chainIdx] = level;

  // Upon exiting, chainIdx contains the number of chains
  while (true) {
    kernelAnalyze<<<gridDim, blockDim>>>(cRoot, levelInd, levelPtr,
                                         nRoots, rRoot, rowsDone, level, depGraph);
    cudaDeviceSynchronize();
    printf("analyzed\n");

    cudaMemcpy(&nRoots_host, nRoots, sizeof(int), cudaMemcpyDeviceToHost);
    if (nRoots_host == 0) {
      chainPtr[++chainIdx] = level;
      break;
    }

    ++level;

    if (rowsInChain + nRoots_host > THREADS_PER_BLOCK) {
      // We've filled up the current chain
      chainPtr[++chainIdx] = level;
      rowsInChain = nRoots_host;
    }

    rowsInChain += nRoots_host;
    rowsDone += nRoots_host;

    // Get 0-1 array of roots
    kernelFindRootsInCandidates<<<gridDim, blockDim>>>(rRoot, cRoot, depGraph);
    cudaDeviceSynchronize();
    printf("found roots in candidates\n");
    thrust::inclusive_scan(thrust::device_pointer_cast(rRoot),
                           thrust::device_pointer_cast(rRoot) + m,
                           thrust::device_pointer_cast(rRoot));
  }

  // Print out roots on each level for testing
  int *levelPtr_host = (int*)malloc(sizeof(int)*(m + 1));
  int *levelInd_host = (int*)malloc(sizeof(int)*m);
  cudaMemcpy(levelPtr_host, levelPtr, sizeof(int)*(m + 1), cudaMemcpyDeviceToHost);
  cudaMemcpy(levelInd_host, levelInd, sizeof(int)*m, cudaMemcpyDeviceToHost);
  printf("levelPtr: ");
  for (int i = 0; i < level + 1; ++i) {
    printf("%d ", levelPtr_host[i]);
  }
  printf("\n");
  printf("levelInd: ");
  for (int i = 0; i < m; ++i) {
    printf("%d ", levelInd_host[i]);
  }
  printf("\n");
  printf("chainPtr: ");
  for (int i = 0; i < chainIdx + 1; ++i) {
    printf("%d ", chainPtr[i]);
  }
  printf("\n");

  // SOLVE PHASE

  int start;
  int end;

  dim3 gridDimOneBlock(1);
  // Iterate over chains
  for (int i = 0; i < chainIdx; ++i) {
    start = chainPtr[i];
    end = chainPtr[i+1];

    // Process a chain
    if (end - start > 1) {
      kernelSingleblock<<<gridDimOneBlock, blockDim>>>(start, end, levelInd, levelPtr, device_b, device_vals);
    }
    // Process a single level
    else {
      kernelMultiblock<<<gridDim, blockDim>>>(start, levelInd, levelPtr, device_b, device_vals);
    }
  }

  double *x = (double*)malloc(m*sizeof(double));
  cudaMemcpy(x, device_b, m*sizeof(double), cudaMemcpyDeviceToHost);
  printf("solution:\n");
  for (int i = 0; i < m; ++i) {
    printf("%f\n", x[i]);
  }
  printf("\n");
  free(x);

  cudaFree(levelInd);
  cudaFree(levelPtr);
  free(chainPtr);
  cudaFree(rRoot);
  cudaFree(wRoot);
  cudaFree(cRoot);
  cudaFree(nRoots);
  cudaFree(depGraph);
}

void CudaSolver::upperTriangularSolve() {
  // lol
}
