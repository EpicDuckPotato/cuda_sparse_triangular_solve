#include <iostream>
#include <fstream>
#include "mm_io.h"
#include <malloc.h>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include "cudaSolver.h"

extern "C" {
  #include "csparse.h"
  #include "mm_io.h"
}

using namespace std;

int main (int argc, char *argv[])
{
  if (argc < 4) {
    printf("Input a filename for the LHS, a filename for the RHS, and spd or nspd to indicate whether the matrix is or isn't symmetric positive definite.\n");
  }

  char *LHS_file = argv[1];
  char *RHS_file = argv[2];
  char *spd_str = argv[3];

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

  // Read RHS vector (b)
  double *b = (double*)malloc(sizeof(double)*m);
  if ((fp = fopen(RHS_file, "r")) == NULL)
    exit(1);
  for (int line = 0; line < 6; ++line) {
    fscanf(fp, "%*[^\n]\n");
  }
  // Size info
  int bm, bn;
  fscanf(fp, "%d %d\n", &bm, &bn);
  assert(bn == 1);

  for (int row = 0; row < bm; row++)
  {
    fscanf(fp, "%lg\n", &b[row]);
  }

  bool spd = !strncmp(spd_str, "spd", 3);

  // Allocate memory for solution
  double *x = (double*)malloc(sizeof(double)*n);

  CudaSolver solver(row_idx, col_idx, vals, m, nz, b, spd, false);
  solver.factor();
  solver.solve(x);

  free(row_idx);
  free(col_idx);
  free(vals);
  free(x);

  return 0;
}
