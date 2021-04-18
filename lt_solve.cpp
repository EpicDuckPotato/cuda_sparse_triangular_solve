#include <iostream>
#include <fstream>

extern "C"
{
#include "cholmod.h"
}

using namespace std;

int main (int argc, char *argv[])
{
  if (argc < 3) {
    printf("Input a filename for the LHS and the RHS\n");
  }

  char *matFile = argv[1];
  FILE* fp = fopen(matFile, "r");
  assert(fp != NULL);

  cholmod_sparse *A;
  cholmod_factor *L;

  cholmod_common* c = (cholmod_common*)malloc(sizeof(cholmod_common));
  c->supernodal = CHOLMOD_SIMPLICIAL;
  c->dtype = CHOLMOD_DOUBLE;
  c->itype = CHOLMOD_INT;
  cholmod_l_start(c); /* start CHOLMOD */

  A = cholmod_l_read_sparse(fp, c); /* read in a matrix */
  L = cholmod_l_analyze(A, c); /* analyze */

  cholmod_l_factorize(A, L, c); /* factorize */

  int *col_ptr = (int*)(L->p);
  int *row_idx = (int*)(L->i);
  double *vals = (double*)(L->x);

  // Read RHS
  int ret_code;
  MM_typecode matcode;
  if ((fp = fopen(argv[2], "r")) == NULL) 
    exit(1);
  if (mm_read_banner(fp, &matcode) != 0)
  {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  double *x = malloc(sizeof(double)*(A->nrow));
  for (int row = 0; row < A->nrows; row++)
  {
    fscanf(fp, "%lg\n", &x[row]);
  }

  // Solve Lx = b
  double *x = malloc(sizeof(double)*(A->nrow));
  for (int row = 0; row < A->nrow; ++row) {
    int col_start = col_ptr[row];
    int col_end = col_ptr[row + 1];
    // Iterate over column
    for (int i = col_start; i < col_end; ++i) {
      // x[i] -= x[row]*L[i, row]
      if (row_idx[i] <= row) {
        // Don't consider earlier rows than current row
        continue;
      }
      x[row_idx[i]] -= x[row]*vals[i];
    }
  }

  ofstream myfile;
  myfile.open("solution.txt");
  for (int row = 0; row < A->nrow; ++row) {
    myfile << x[row] << endl;
  }

  myfile.close();
  
  return 0;
}
