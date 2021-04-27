#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "cudaSolver.h"
#include <malloc.h>

struct GlobalConstants {
  int *row_ptr;
  int *col_idx;
  int m;
  int nnz;
};

__constant__ GlobalConstants cuConstSolverParams;

/*
 * kernelFindRootsP1: parallelizes over rows of the dependency
 * graph and indicates roots
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsP1(int *roots, char *depGraph) {
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
 * kernelFindRootsP2: should be called after kernelFindRootsP1
 * ARGUMENTS
 * wRoot: populated with rows of roots
 * nRoots: populated with number of roots
 * rootScan: inclusive scan of the roots array from kernelFindRootsP1
 */
__global__ void kernelFindRootsP2(int *wRoot, int *nRoots, int *rootScan) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m &&
      ((row == 0 && rootScan[row] == 1) ||
       (row > 0 && rootScan[row] == rootScan[row - 1] + 1))) {
    wRoot[rootScan[row] - 1] = row;
  }
  *nRoots = rootScan[cuConstSolverParams.m - 1];
}

/*
 * kernelFindRootsInCandidatesP1: parallelizes over rows of the dependency
 * graph and indicates roots, only looking at rows given by cRoots
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * cRoots: set of rows that could be roots
 * nCand: number of candidates
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsInCandidatesP1(int *roots, int *cRoots, int *nCand, char *depGraph) {
  int candIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (candIdx < *nCand) {
    int row = cRoots[candIdx];
    int rowStart = cuConstSolverParams.row_ptr[row];

    // There's a - 1 because the last element of the row
    // is the diagonal element, which isn't a dependency
    // for solving this row
    int rowEnd = cuConstSolverParams.row_ptr[cRoots[row] + 1] - 1;

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
 * kernelAnalyze: populates levelInd, levelPtr, chainPtr, and cRoots.
 * chainPtr determines the properties and number of kernels to be launched in the solve phase.
 * ARGUMENTS
 * roots: populated with rows of roots
 * nRoots: number of roots
 * cRoots: set of rows that could be roots
 * nCand: number of candidates
 * levelInd: list of sorted rows belonging to every level
 * levelPtr: list of ending index (in levelInd) of each level
 * chainPtr: list of ending index (in levelPtr) of each chain
 * (Note: Naumov has levelPtr & chainPtr be the list of starting indices of each level/chain
 *   + an extra element to indicate the end of the last level/chain)
 * levelIndSize: size of levelInd
 * levelPtrSize: size of levelPtr
 * chainPtrSize: size of chainPtr
 * depGraph: value array for the dependency graph
 */
__global__ void kernelAnalyze(int *roots, int *nRoots, int *cRoots, int *nCand, int *levelInd, int *levelPtr, int *chainPtr, int *levelIndSize, int *levelPtrSize, int *chainPtrSize, char *depGraph) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *nRoots) {
    // TODO: Not sure if this indexing is correct
    int root = roots[idx];
    int colStart = cuConstSolverParams.col_idx[root];
    int colEnd = cuConstSolverParams.col_idx[root + 1];

    for (int i = colStart; i < colEnd; ++i) {
      if (depGraph[i]) {
        // Dependency exists, set to 0 and add to cRoots
        depGraph[i] = 0;
        cRoots[*nCand] = i;
        *nCand += 1;
      }
    }

    // TODO: Populate levelInd, is this sorted? when to increase levelIndSize?
    levelInd[*levelIndSize + idx] = root;

    // Populate levelPtr, only do this once
    if (idx == *nRoots - 1) {
      levelPtr[*levelPtrSize] = *levelIndSize + *nRoots - 1;
      *levelPtrSize += 1;
    }

    // TODO: how to populate chainPtr? how to determine size of chain?
  }
}

CudaSolver::CudaSolver(int *row_idx, int *col_idx, double *vals, int m, int nnz, double *b, bool spd, bool is_lt) : m(m), nnz(nnz), spd(spd), is_lt(is_lt) {
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
  int *levelPtr;
  int *chainPtr;
  int *rRoot;
  int *wRoot;
  int *cRoot;
  int *scratch;
  int *nRoots;
  int *nCand;
  int *levelIndSize;
  int *levelPtrSize;
  int *chainPtrSize;

  // The maximum number of roots is the number of rows
  cudaMalloc(&rRoot, m*sizeof(int));
  cudaMalloc(&wRoot, m*sizeof(int));
  cudaMalloc(&cRoot, m*sizeof(int));
  cudaMalloc(&scratch, m*sizeof(int));
  cudaMalloc(&nRoots, sizeof(int));
  cudaMalloc(&nCand, sizeof(int));
  cudaMalloc(&levelInd, m*sizeof(int));
  cudaMalloc(&levelPtr, m*sizeof(int));
  cudaMalloc(&chainPtr, m*sizeof(int));
  cudaMalloc(&levelIndSize, sizeof(int));
  cudaMalloc(&levelPtrSize, sizeof(int));
  cudaMalloc(&chainPtrSize, sizeof(int));

  // Sparse binary matrix with the same row pointers and column indices
  // as the LHS. If a row contains all zeros, the corresponding row of the solution
  // has no dependencies, and is therefore a root
  char *depGraph;
  cudaMalloc(&depGraph, nnz*sizeof(char));
  cudaMemset(depGraph, 1, nnz*sizeof(char));

  // ANALYSIS PHASE

  // Finding roots parallelizes over rows, so we have 1D blocks
  dim3 blockDim(256);
  dim3 gridDim((m + blockDim.x - 1) / blockDim.x);

  // TODO: Naumov only used one kernel for this. What am I doing wrong?
  printf("Finding roots p1\n");
  kernelFindRootsP1<<<gridDim, blockDim>>>(scratch, depGraph);
  cudaDeviceSynchronize();
  printf("Scanning\n");
  thrust::inclusive_scan(thrust::device_pointer_cast(scratch),
                         thrust::device_pointer_cast(scratch) + m,
                         thrust::device_pointer_cast(scratch));
  cudaDeviceSynchronize();
  printf("Finding roots p2\n");
  kernelFindRootsP2<<<gridDim, blockDim>>>(wRoot, nRoots, scratch);
  cudaDeviceSynchronize();

  int nCand_host = 0;

  // Only for debugging
  int *wRoot_host = (int*)malloc(m*sizeof(int));
  int nRoots_host = 0;
  while (true) {
    // TODO: replaced rRoot with wRoot, seems like rRoot unnecessary?
    //kernelAnalyze<<<gridDim, blockDim>>>(wRoot, nRoots, cRoot, nCand, levelInd, levelPtr, chainPtr, levelIndSize, levelPtrSize, chainPtrSize, depGraph);
    //
    cudaMemcpy(&nRoots_host, nRoots, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(wRoot_host, wRoot, m*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nRoots_host; ++i) {
      printf("%d ", wRoot[i]);
    }
    printf("\n");

    cudaMemcpy(&nCand_host, nCand, sizeof(int), cudaMemcpyDeviceToHost);
    if (nCand_host == 0) {
      break;
    }

    // TODO: again, Naumov did this with one kernel
    kernelFindRootsInCandidatesP1<<<gridDim, blockDim>>>(scratch, cRoot, nCand, depGraph);
    thrust::inclusive_scan(thrust::device_pointer_cast(scratch),
                           thrust::device_pointer_cast(scratch) + m,
                           thrust::device_pointer_cast(scratch));
    kernelFindRootsP2<<<gridDim, blockDim>>>(wRoot, nRoots, scratch);
  }

  free(wRoot_host);

  // SOLVE PHASE
  // TODO: solve phase


  cudaFree(rRoot);
  cudaFree(wRoot);
  cudaFree(cRoot);
  cudaFree(scratch);
  cudaFree(nRoots);
  cudaFree(nCand);
  cudaFree(depGraph);
  cudaFree(levelInd);
  cudaFree(levelPtr);
  cudaFree(chainPtr);
  cudaFree(levelIndSize);
  cudaFree(levelPtrSize);
  cudaFree(chainPtrSize);
}

void CudaSolver::upperTriangularSolve() {
  // lol
}
