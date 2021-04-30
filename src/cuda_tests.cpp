#include <iostream>
#include <fstream>
#include <cstdlib>
#include "cudaSolver.h"
#include <malloc.h>
#include <vector>

using namespace std;

int main (int argc, char *argv[])
{
  int m = 4;
  int nnz = 4;
  vector<int> row_idx(nnz);
  vector<int> col_idx(nnz);
  vector<double> vals(nnz);
  vector<double> b(m);
  vector<double> x(m);

  for (int i = 0; i < nnz; ++i) {
    row_idx[i] = i;
    col_idx[i] = i;
    vals[i] = 1;
    b[i] = 1;
  }

  cout << "Testing 4x4 identity" << endl;
  CudaSolver solver(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0], false, true);
  solver.solve(&x[0]);

  nnz = 6;
  row_idx.insert(row_idx.begin() + 3, 3);
  col_idx.insert(col_idx.begin() + 3, 0);
  vals.insert(vals.begin() + 3, 1);

  row_idx.insert(row_idx.begin() + 2, 2);
  col_idx.insert(col_idx.begin() + 2, 1);
  vals.insert(vals.begin() + 3, 1);

  for (int i = 0; i < 6; ++i) {
    cout << row_idx[i] << " " << col_idx[i] << " " << vals[i] << endl;
  }

  cout << "Testing slightly more complex" << endl;
  CudaSolver solver2(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0], false, true);
  solver2.solve(&x[0]);
  return 0;
}
