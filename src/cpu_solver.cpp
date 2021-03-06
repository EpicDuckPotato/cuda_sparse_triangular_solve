#include <iostream>
#include <fstream>
#include "mm_io.h"
#include "csparse.h"
#include <malloc.h>
#include <cstdlib>

extern "C" {
  // Get declaration for f(int i, char c, float x)
  #include "csparse.h"
  #include "mm_io.h"
}

using namespace std;

// Solves Ax = b
int main (int argc, char *argv[])
{
  if (argc < 3) {
    printf("Input a filename for the LHS and the RHS\n");
  }

  char *LHS_file = argv[1];
  char *RHS_file = argv[2];

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

  cs T;
  if (mm_read_mtx_crd_size(fp, &T.m, &T.n, &T.nz) != 0)
  {
    printf("Could not process Matrix Market size information for LHS.\n");
    exit(1);
  }

  T.i = (int*)malloc(T.nz*sizeof(int));
  T.p = (int*)malloc(T.nz*sizeof(int));
  T.x = (double*)malloc(T.nz*sizeof(double));

  // Read triplets, convert to csc
  for (int i = 0; i < T.nz; i++)
  {
    fscanf(fp, "%d %d %lg\n", &(T.i[i]), &(T.p[i]), &(T.x[i]));
    --T.i[i];
    --T.p[i];
  }
  cs *A = cs_triplet(&T);
  cout << "Converted LHS to CSC" << endl;

  // Get Cholesky factor L, where
  css *S = cs_schol(A, 0);
  cout << "Performed symbolic Cholesky factorization" << endl;
  csn *Ln = cs_chol(A, S);
  cout << "Performed numeric Cholesky factorization" << endl;
  cs *L = Ln->L;

  // Get CSC representation of Cholesky factor L, where A = P'LL'P
  int *col_ptr = L->p;
  int *row_idx = L->i;
  double *vals = L->x;

  // Read RHS vector (b)
  double *b = (double*)malloc(sizeof(double)*(A->m));
  if ((fp = fopen(RHS_file, "r")) == NULL)
    exit(1);
  for (int line = 0; line < 6; ++line) {
    fscanf(fp, "%*[^\n]\n");
  }
  // Size info
  int bm, bn;
  fscanf(fp, "%d %d\n", &bm, &bn);

  for (int row = 0; row < A->m; row++)
  {
    fscanf(fp, "%lg\n", &b[row]);
  }

  cout << "Multiplying by permutation matrix" << endl;
  double *x = (double*)malloc(sizeof(double)*(A->m));
  cs_ipvec(A->m, S->Pinv, b, x);

  cout << "Solving lower triangular system" << endl;

  // Solve Lv = b (lower triangular solve), putting v into x.
  // row corresponds to the row of x that we're solving for.
  // We divide by L[row, row], then multiply x[row] by L[:, row],
  // and subtract from subsequent entries in x
  for (int row = 0; row < A->m; ++row) {
    // Indexing might be confusing here. Recall that
    // a row of x is associated with a column of L
    int col_start = col_ptr[row];
    int col_end = col_ptr[row + 1];

    // x[row] = b[row]/L[row, row]
    x[row] /= vals[col_start];

    // Iterate down the column, starting right after the diagonal
    for (int i = col_start + 1; i < col_end; ++i) {
      // b[row_idx[i]] -= x[row]*L[i, row]
      x[row_idx[i]] -= x[row]*vals[i];
    }
  }

  cout << "Solving upper triangular system" << endl;

  // Solve LTx = v (upper triangular solve). The trick here
  // is that we can just treat L as a CSR matrix now, so
  // we don't need to compute the transpose explicitly.
  int *row_ptr = col_ptr;
  int *col_idx = row_idx;
  for (int row = A->m - 1; row >= 0; --row) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int i = row_start + 1; i < row_end; ++i) {
      x[row] -= x[i]*vals[i];
    }

    x[row] /= vals[row_start];
  }

  cout << "Multiplying by permutation matrix" << endl;
  cs_pvec(A->m, S->Pinv, x, b);

  cout << "Writing solution" << endl;

  ofstream myfile;
  myfile.open("solution.txt");
  for (int row = 0; row < A->m; ++row) {
    myfile << b[row] << endl;
  }

  myfile.close();

  // Solve system with csparse and see if it matches
  if ((fp = fopen(argv[2], "r")) == NULL)
    exit(1);
  for (int line = 0; line < 6; ++line) {
    fscanf(fp, "%*[^\n]\n");
  }
  // Size info
  fscanf(fp, "%d %d\n", &bm, &bn);

  for (int row = 0; row < A->m; row++)
  {
    fscanf(fp, "%lg\n", &b[row]);
  }
  cs_cholsol(A, b, 0);

  cout << "Writing ground truth solution" << endl;

  myfile;
  myfile.open("gt_solution.txt");
  for (int row = 0; row < A->m; ++row) {
    myfile << b[row] << endl;
  }

  myfile.close();

  cs_free(A);
  cs_free(S);
  cs_free(L);
  free(x);

  return 0;
}
