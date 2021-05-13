#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "cudaSolver.h"
#include <malloc.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 256

struct GlobalConstants {
  int *row_ptr;
  int *col_idx;
  int m;
  int nnz;

  // Only for dependency graph
  int *col_ptr;
  int *row_idx;
};

__constant__ GlobalConstants cuConstSolverParams;

// LOWER TRIANGULAR KERNELS. ASSUME DIAGONAL CONTAINS ONES

/*
 * kernelFindRootsL: parallelizes over rows of the dependency
 * graph and indicates roots (for lower triangular)
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsL(int *roots, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m) {
    roots[row] = 1;
    for (int i = cuConstSolverParams.row_ptr[row];
         cuConstSolverParams.col_idx[i] < row && i < cuConstSolverParams.row_ptr[row + 1]; ++i) {
      if (depGraph[i]) {
        // Dependency exists
        roots[row] = 0;
        break;
      }
    }
  }
}

/*
 * kernelFindRootsInCandidatesL: parallelizes over rows of the dependency
 * graph and indicates roots, only looking at rows given by cRoot (for lower triangular)
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * cRoot: 0-1 array indicating candidates
 * nCand: number of candidates
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsInCandidatesL(int *roots, char *cRoot, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m) {
    if (cRoot[row] == 1) {
      roots[row] = 1;
      for (int i = cuConstSolverParams.row_ptr[row];
           i < cuConstSolverParams.row_ptr[row + 1] &&
           cuConstSolverParams.col_idx[i] < row; ++i) {
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
}

/*
 * kernelFindRootsU: parallelizes over rows of the dependency
 * graph and indicates roots (for upper triangular)
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsU(int *roots, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m) {
    int rowStart = cuConstSolverParams.row_ptr[row];
    int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

    roots[row] = 1;
    for (int i = rowEnd; cuConstSolverParams.col_idx[i] > row && i >= rowStart; --i) {
      if (depGraph[i]) {
        // Dependency exists
        roots[row] = 0;
        break;
      }
    }
  }
}

/*
 * kernelFindRootsInCandidatesU: parallelizes over rows of the dependency
 * graph and indicates roots, only looking at rows given by cRoot (for upper triangular)
 * ARGUMENTS
 * roots: roots[i] is populated with 1 if row i is a root, and zero otherwise
 * cRoot: 0-1 array indicating candidates
 * nCand: number of candidates
 * depGraph: value array for the dependency graph
 */
__global__ void kernelFindRootsInCandidatesU(int *roots, char *cRoot, char *depGraph) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < cuConstSolverParams.m) {
    if (cRoot[row] == 1) {
      int rowStart = cuConstSolverParams.row_ptr[row];
      int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

      roots[row] = 1;
      for (int i = rowEnd;
           i >= rowStart && cuConstSolverParams.col_idx[i] > row; --i) {
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
    for (int i = cuConstSolverParams.col_ptr[row];
         i < cuConstSolverParams.col_ptr[row + 1]; ++i) {
      int crow = cuConstSolverParams.row_idx[i];
      for (int j = cuConstSolverParams.row_ptr[crow];
           j < cuConstSolverParams.row_ptr[crow + 1]; ++j) {
        if (cuConstSolverParams.col_idx[j] == row) {
          depGraph[j] = 0;
        }
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
 * kernelMultiblockL: processes a single level for lower triangular
 * ARGUMENTS
 * start: start of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelMultiblockL(int start, int *levelInd, int *levelPtr, double *b, double *val) {
  int startIdx = levelPtr[start];
  int endIdx = levelPtr[start + 1];

  int idx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < endIdx) {
    // Compute element of solution corresponding to row
    int row = levelInd[idx];
    for (int i = cuConstSolverParams.row_ptr[row];
         i < cuConstSolverParams.row_ptr[row + 1] && cuConstSolverParams.col_idx[i] < row; ++i) {
      b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
    }
  }
}

/*
 * kernelSingleblockL: processes a chain for lower triangular
 * ARGUMENTS
 * start: start of chain
 * end: end of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelSingleblockL(int start, int end, int *levelInd, int *levelPtr, double *b, double *val) {
  int startIdx;
  int endIdx;

  for (int i = start; i < end; ++i) {
    startIdx = levelPtr[i];
    endIdx = levelPtr[i + 1];

    int idx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < endIdx) {
      // Compute element of solution corresponding to row
      int row = levelInd[idx];
      for (int i = cuConstSolverParams.row_ptr[row];
           i < cuConstSolverParams.row_ptr[row + 1] && cuConstSolverParams.col_idx[i] < row; ++i) {
        b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
      }
    }
    __syncthreads();
  }
}

/*
 * kernelMultiblockU: processes a single level for upper triangular
 * ARGUMENTS
 * start: start of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelMultiblockU(int start, int *levelInd, int *levelPtr, double *b, double *val) {
  int startIdx = levelPtr[start];
  int endIdx = levelPtr[start + 1];

  int idx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < endIdx) {
    // Compute element of solution corresponding to row
    int row = levelInd[idx];
    int rowStart = cuConstSolverParams.row_ptr[row];
    int rowEnd = cuConstSolverParams.row_ptr[row + 1] - 1;

    int i = rowEnd;
    for (; i >= rowStart && cuConstSolverParams.col_idx[i] > row; --i) {
      b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
    }
    b[row] /= val[i];
  }
}

/*
 * kernelSingleblockU: processes a chain for upper triangular
 * ARGUMENTS
 * start: start of chain
 * end: end of chain
 * levelInd: sorted rows belonging to each level
 * levelPtr: starting indices (in levelInd) of each level
 * b: b matrix, populated with solution
 * val: L matrix values
 */
__global__ void kernelSingleblockU(int start, int end, int *levelInd, int *levelPtr, double *b, double *val) {
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

      int i = rowEnd;
      for (; i >= rowStart && cuConstSolverParams.col_idx[i] > row; --i) {
        b[row] -= val[i] * b[cuConstSolverParams.col_idx[i]];
      }
      b[row] /= val[i];
    }
    __syncthreads();
  }
}

CudaSolver::CudaSolver(int *row_idx, int *col_idx, double *vals, int m, int nnz, double *b) : m(m), nnz(nnz) {
  cusparseCreate(&cs_handle);

  cudaMalloc(&device_row_ptr, sizeof(int)*(m + 1));
  cudaMalloc(&device_col_ptr, sizeof(int)*(m + 1));
  cudaMalloc(&device_row_idx, sizeof(int)*nnz);
  cudaMalloc(&device_col_idx, sizeof(double)*nnz);
  cudaMalloc(&device_vals, sizeof(double)*nnz);
  cudaMemcpy(device_row_idx, row_idx, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(device_col_idx, col_idx, sizeof(int)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(device_vals, vals, sizeof(double)*nnz, cudaMemcpyHostToDevice);

  thrust::device_vector<int> device_row_idx2(device_row_idx, device_row_idx + nnz);
  thrust::device_vector<int> device_row_idx3(device_row_idx3);

  // Convert COO to CSC (for efficient iteration through dependency graph)
  cusparseXcoo2csr(cs_handle, device_col_idx, nnz, m, device_col_ptr,
                   CUSPARSE_INDEX_BASE_ZERO);

  // We assume col_idx is sorted because that's how the matrices from matrix market are.
  thrust::stable_sort_by_key(thrust::device,
                             device_row_idx2.begin(),
                             device_row_idx2.end(),
                             thrust::device_pointer_cast(device_col_idx));

  thrust::stable_sort_by_key(thrust::device,
                             device_row_idx3.begin(),
                             device_row_idx3.end(),
                             thrust::device_pointer_cast(device_vals));

  // Convert COO to CSR
  cusparseXcoo2csr(cs_handle, thrust::raw_pointer_cast(device_row_idx2.data()), nnz, m, device_row_ptr,
                   CUSPARSE_INDEX_BASE_ZERO);

  cudaMalloc(&device_b, sizeof(double)*m);
  cudaMemcpy(device_b, b, sizeof(double)*m, cudaMemcpyHostToDevice);
}

CudaSolver::~CudaSolver() {
  cusparseDestroy(cs_handle);
  cudaFree(device_row_ptr);
  cudaFree(device_col_idx);
  cudaFree(device_col_ptr);
  cudaFree(device_row_idx);
  cudaFree(device_vals);
  cudaFree(device_b);
}

void CudaSolver::factor() {
  // Boilerplate
  csrilu02Info_t info = 0;
  cusparseCreateCsrilu02Info(&info);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

  int bufferSize;
  cusparseDcsrilu02_bufferSize(cs_handle, m, nnz, descr, device_vals, device_row_ptr,
                               device_col_idx, info, &bufferSize);

  void *pBuffer;
  cudaMalloc(&pBuffer, bufferSize);

  // Analyze
  cusparseDcsrilu02_analysis(cs_handle, m, nnz, descr, device_vals,
                             device_row_ptr, device_col_idx, info,
                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

  // Factor, and put the factors into device_vals
  cusparseDcsrilu02(cs_handle, m, nnz, descr, device_vals,
                    device_row_ptr, device_col_idx, info,
                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

  cudaFree(pBuffer);

  cusparseDestroyMatDescr(descr);

  cusparseDestroyCsrilu02Info(info);
}

void CudaSolver::get_factors(int *row_ptr_L, int *col_idx_L, double *vals_L,
                             int *row_ptr_U, int *col_idx_U, double *vals_U) {
  int *full_row_ptr = (int*)malloc(sizeof(int)*(m + 1));
  int *col_idx = (int*)malloc(sizeof(int)*nnz);
  double *vals = (double*)malloc(sizeof(double)*nnz);
  cudaMemcpy(full_row_ptr, device_row_ptr, (m + 1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_idx, device_col_idx, nnz*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(vals, device_vals, nnz*sizeof(double), cudaMemcpyDeviceToHost);

  int iL = 0;
  int iU = 0;
  int row = 0;

  for (row = 0; row < m; ++row) {
    // Lower
    row_ptr_L[row] = iL;
    int j = full_row_ptr[row];
    for (; col_idx[j] < row && j < full_row_ptr[row + 1]; ++j) {
      col_idx_L[iL] = col_idx[j];
      vals_L[iL] = vals[j];
      ++iL;
    }

    col_idx_L[iL] = row;
    vals_L[iL] = 1;
    ++iL;

    // Upper
    row_ptr_U[row] = iU;
    for (; j < full_row_ptr[row + 1]; ++j) {
      col_idx_U[iU] = col_idx[j];
      vals_U[iU] = vals[j];
      ++iU;
    }
  }
  row_ptr_L[row] = iL;
  row_ptr_U[row] = iU;

  free(full_row_ptr);
  free(col_idx);
  free(vals);
}

void CudaSolver::solve(double *x) {
  printf("lower\n");
  triangularSolve(true);
  printf("upper\n");
  triangularSolve(false);
  cudaMemcpy(x, device_b, m*sizeof(double), cudaMemcpyDeviceToHost);
}

void CudaSolver::triangularSolve(bool isLower) {
  // We'll need to access row_ptr and col_idx quite often without modifying them,
  // so store them as global constants
  GlobalConstants params;
  params.row_ptr = device_row_ptr;
  params.col_idx = device_col_idx;
  params.m = m;
  params.nnz = nnz;
  params.col_ptr = device_col_ptr;
  params.row_idx = device_row_idx;
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
  if (isLower) {
    kernelFindRootsL<<<gridDim, blockDim>>>(rRoot, depGraph);
  } else {
    kernelFindRootsU<<<gridDim, blockDim>>>(rRoot, depGraph);
  }
  cudaDeviceSynchronize();

  thrust::inclusive_scan(thrust::device_pointer_cast(rRoot),
                         thrust::device_pointer_cast(rRoot) + m,
                         thrust::device_pointer_cast(rRoot));

  int nRoots_host = 0;
  int level = 0;
  int rowsDone = 0;
  int rowsInChain = 0;
  int chainIdx = 0;
  chainPtr[chainIdx] = level;

  //struct timeval t1, t2;

  // Upon exiting, chainIdx contains the number of chains
  while (true) {
    //gettimeofday(&t1, 0);
    kernelAnalyze<<<gridDim, blockDim>>>(cRoot, levelInd, levelPtr,
                                         nRoots, rRoot, rowsDone, level, depGraph);
    cudaDeviceSynchronize();

    /*
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to analyze roots:  %3.1f ms \n", time);
    */

    cudaMemcpy(&nRoots_host, nRoots, sizeof(int), cudaMemcpyDeviceToHost);
    if (nRoots_host == 0) {
      chainPtr[++chainIdx] = level;
      // Now the last element of chainPtr contains the number of levels
      break;
    }

    ++level;

    if (rowsInChain + nRoots_host > THREADS_PER_BLOCK) {
      // Adding this new level of roots to the current chain
      // would cause us to overflow the current chain. Add a new
      // chain starting at this level
      chainPtr[++chainIdx] = level;
      rowsInChain = 0;
    }

    rowsInChain += nRoots_host;
    rowsDone += nRoots_host;

    // Get 0-1 array of roots
    if (isLower) {
      kernelFindRootsInCandidatesL<<<gridDim, blockDim>>>(rRoot, cRoot, depGraph);
    } else {
      kernelFindRootsInCandidatesU<<<gridDim, blockDim>>>(rRoot, cRoot, depGraph);
    }
    cudaDeviceSynchronize();

    thrust::inclusive_scan(thrust::device_pointer_cast(rRoot),
                           thrust::device_pointer_cast(rRoot) + m,
                           thrust::device_pointer_cast(rRoot));
  }

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
      if (isLower) {
        kernelSingleblockL<<<gridDimOneBlock, blockDim>>>(start, end, levelInd, levelPtr, device_b, device_vals);
      } else {
        kernelSingleblockU<<<gridDimOneBlock, blockDim>>>(start, end, levelInd, levelPtr, device_b, device_vals);
      }
    }
    // Process a single level
    else {
      if (isLower) {
        kernelMultiblockL<<<gridDim, blockDim>>>(start, levelInd, levelPtr, device_b, device_vals);
      } else {
        kernelMultiblockU<<<gridDim, blockDim>>>(start, levelInd, levelPtr, device_b, device_vals);
      }
    }
  }

  printf("Freeing levelInd\n");
  cudaFree(levelInd);
  printf("Freeing levelPtr\n");
  cudaFree(levelPtr);
  printf("Freeing chainPtr\n");
  free(chainPtr);
  printf("Freeing rRoot\n");
  cudaFree(rRoot);
  printf("Freeing cRoot\n");
  cudaFree(cRoot);
  printf("Freeing nRoots\n");
  cudaFree(nRoots);
  printf("Freeing depGraph\n");
  cudaFree(depGraph);
}
