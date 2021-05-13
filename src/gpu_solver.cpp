#include <iostream>
#include <fstream>
#include "mm_io.h"
#include <malloc.h>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include "cudaSolver.h"
#include "csparse.h"
#include <chrono>

extern "C" {
  #include "csparse.h"
  #include "mm_io.h"
}

using namespace std;
using namespace std::chrono;

int main (int argc, char *argv[])
{
  if (argc < 2) {
    printf("Input a filename for the LHS\n");
  }

  char *LHS_file = argv[1];

  // Read LHS matrix (A)
  cout << "Reading LHS" << endl;
  int ret_code;
  MM_typecode matcode;
  FILE *fp;
  if ((fp = fopen(LHS_file, "r")) == NULL)
    exit(1);
  if (mm_read_banner(fp, &matcode) != 0)
  {
    printf("Could not process Matrix Market banner for LHS.\n");
    exit(1);
  }

  int m;
  int n;
  int nz;
  if (mm_read_mtx_crd_size(fp, &m, &n, &nz) != 0)
  {
    printf("Could not process Matrix Market size information for LHS.\n");
    exit(1);
  }

  // Read triplets (COO)
  int *row_idx = (int*)malloc(nz*sizeof(int));
  int *col_idx = (int*)malloc(nz*sizeof(int));
  double *vals = (double*)malloc(nz*sizeof(double));

  for (int i = 0; i < nz; i++)
  {
    fscanf(fp, "%d %d %lg\n", &(row_idx[i]), &(col_idx[i]), &(vals[i]));
    --row_idx[i];
    --col_idx[i];
  }

  // Create an RHS vector (b)
  double *b = (double*)malloc(sizeof(double)*m);
  for (int row = 0; row < m; row++)
  {
    b[row] = 1;
  }

  // Allocate memory for solution
  double *x = (double*)malloc(sizeof(double)*m);

  CudaSolver solver(row_idx, col_idx, vals, m, nz, b);
  solver.factor();
  solver.solve(x);

  ofstream myfile;
  myfile.open("solution.txt");
  for (int row = 0; row < m; ++row) {
    myfile << x[row] << endl;
  }
  myfile.close();

  // Get L factor and check solution against 
  
  int *row_ptr_L = (int*)malloc(sizeof(int)*(m + 1));
  int *col_idx_L = (int*)malloc(sizeof(int)*nz);
  double *vals_L = (double*)malloc(sizeof(double)*nz);
  int *row_ptr_U = (int*)malloc(sizeof(int)*(m + 1));
  int *col_idx_U = (int*)malloc(sizeof(int)*nz);
  double *vals_U = (double*)malloc(sizeof(double)*nz);
  solver.get_factors(row_ptr_L, col_idx_L, vals_L,
                     row_ptr_U, col_idx_U, vals_U);

  auto start = high_resolution_clock::now();
  cs Lt;
  Lt.nzmax = nz;
  Lt.m = m;
  Lt.n = m;
  Lt.p = row_ptr_L;
  Lt.i = col_idx_L;
  Lt.x = vals_L;
  Lt.nz = nz;
  cs_utsolve(&Lt, b);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Csparse upper time: " << duration.count() << endl;

  start = high_resolution_clock::now();
  cs Ut;
  Ut.nzmax = nz;
  Ut.m = m;
  Ut.n = m;
  Ut.p = row_ptr_U;
  Ut.i = col_idx_U;
  Ut.x = vals_U;
  Ut.nz = nz;
  cs_ltsolve(&Ut, b);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout << "Csparse lower time: " << duration.count() << endl;
  

  myfile.open("gt_solution.txt");
  for (int row = 0; row < m; ++row) {
    myfile << b[row] << endl;
  }

  myfile.close();
  free(row_ptr_L);
  free(col_idx_L);
  free(vals_L);
  free(row_ptr_U);
  free(col_idx_U);
  free(vals_U);

  free(row_idx);
  free(col_idx);
  free(vals);
  free(x);

  return 0;
}
