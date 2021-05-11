#include <iostream>
#include <fstream>
#include <cstdlib>
#include "cudaSolver.h"
#include <malloc.h>
#include <vector>

using namespace std;

void print_csr_matrix(int *row_ptr, int *col_idx, double *vals, int m) {
  for (int row = 0; row < m; ++row) {
    int rowstart = row_ptr[row];
    int rowend = row_ptr[row + 1];
    int col = 0;
    for (int i = rowstart; i < rowend; ++i) {
      while (col < col_idx[i]) {
        cout << "0 ";
        ++col;
      }
      cout << vals[i] << " ";
      ++col;
    }
    cout << endl;
  }
}

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
  CudaSolver solver(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0]);
  solver.factor();
  solver.solve(&x[0]);

  vector<int> row_ptr_L(m + 1);
  vector<int> col_idx_L(nnz);
  vector<double> vals_L(nnz);
  vector<int> row_ptr_U(m + 1);
  vector<int> col_idx_U(nnz);
  vector<double> vals_U(nnz);
  solver.get_factors(row_ptr_L.data(), col_idx_L.data(), vals_L.data(),
                     row_ptr_U.data(), col_idx_U.data(), vals_U.data());

  cout << "L factor:" << endl;
  print_csr_matrix(row_ptr_L.data(), col_idx_L.data(), vals_L.data(), m);

  cout << "U factor:" << endl;
  print_csr_matrix(row_ptr_U.data(), col_idx_U.data(), vals_U.data(), m);

  cout << "solution:" << endl;
  for (int i = 0; i < m; ++i) {
    cout << x[i] << endl;
  }

  nnz = 6;
  row_idx.insert(row_idx.begin() + 1, 3);
  col_idx.insert(col_idx.begin() + 1, 0);
  vals.insert(vals.begin() + 1, 1);

  cout << "inserting" << endl;
  row_idx.insert(row_idx.begin() + 3, 2);
  col_idx.insert(col_idx.begin() + 3, 1);
  vals.insert(vals.begin() + 3, 1);

  cout << "Testing slightly more complex" << endl;
  CudaSolver solver2(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0]);
  solver2.factor();
  solver2.solve(&x[0]);

  cout << "getting solver 2 factors" << endl;
  row_ptr_L.resize(m);
  col_idx_L.resize(nnz);
  vals_L.resize(nnz);
  row_ptr_U.resize(m);
  col_idx_U.resize(nnz);
  vals_U.resize(nnz);
  solver2.get_factors(row_ptr_L.data(), col_idx_L.data(), vals_L.data(),
                      row_ptr_U.data(), col_idx_U.data(), vals_U.data());
  cout << "L factor:" << endl;
  print_csr_matrix(row_ptr_L.data(), col_idx_L.data(), vals_L.data(), m);

  cout << "U factor:" << endl;
  print_csr_matrix(row_ptr_U.data(), col_idx_U.data(), vals_U.data(), m);

  cout << "solution:" << endl;
  for (int i = 0; i < m; ++i) {
    cout << x[i] << endl;
  }

  cout << "Testing something that's not all ones" << endl;
  b[2] = 10;
  b[3] = 3;
  CudaSolver solver3(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0]);
  solver3.factor();
  solver3.solve(&x[0]);

  solver3.get_factors(row_ptr_L.data(), col_idx_L.data(), vals_L.data(),
                      row_ptr_U.data(), col_idx_U.data(), vals_U.data());
  cout << "L factor:" << endl;
  print_csr_matrix(row_ptr_L.data(), col_idx_L.data(), vals_L.data(), m);

  cout << "U factor:" << endl;
  print_csr_matrix(row_ptr_U.data(), col_idx_U.data(), vals_U.data(), m);

  cout << "solution:" << endl;
  for (int i = 0; i < m; ++i) {
    cout << x[i] << endl;
  }

  cout << "Testing something else that's not all ones" << endl;
  vals[3] = 3;
  CudaSolver solver4(&row_idx[0], &col_idx[0], &vals[0], m, nnz, &b[0]);
  solver4.factor();
  solver4.solve(&x[0]);

  solver4.get_factors(row_ptr_L.data(), col_idx_L.data(), vals_L.data(),
                      row_ptr_U.data(), col_idx_U.data(), vals_U.data());
  cout << "L factor:" << endl;
  print_csr_matrix(row_ptr_L.data(), col_idx_L.data(), vals_L.data(), m);

  cout << "U factor:" << endl;
  print_csr_matrix(row_ptr_U.data(), col_idx_U.data(), vals_U.data(), m);

  cout << "solution:" << endl;
  for (int i = 0; i < m; ++i) {
    cout << x[i] << endl;
  }

  return 0;
}
