#ifndef SPMV_H
#define SPMV_H

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <list>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

/* ----------- Matrix Format ----------- */

struct csrmatrix
{
	int n, nnz;
	std::vector<int> ptrow;
	std::vector<int> indcol;
	std::vector<double> coef;
};

struct bcsr4x4_matrix
{
	int nrows;
	int nblocks;
	std::vector<int> ptrow;
	std::vector<int> indcol;
	std::vector<double> coef;
};

/* ----------- Matrix Handler ----------- */

void generate_CSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp,
                  int nrow, int nnz, int* irow, int* jcol, double* val);

void generate_BCSR4(std::list<std::pair<int, std::array<double, 16>> >* block_rows,
                    int nrow, int nnz, const int* irow, const int* jcol, const double* val,
                    bcsr4x4_matrix& A);

void COO2CSR(csrmatrix& a, int nrow, int nnz, int* irow, int* jcol, double* val);

/* ----------- Error Checking ----------- */

double norm2(const std::vector<double>& x);

double rel_error(const std::vector<double>& ref, const std::vector<double>& test);

/* ----------- SPMV CSR Kernels ----------- */

// CSR format
void SpMV_CSR(double* y, double* x, csrmatrix& A);
void SpMV_CSR_OPT(double* y, double* x, csrmatrix& A);
void SpMV_CSR_FMA(double* y, double* x, csrmatrix& A);
void SpMV_CSR_AVX2(double* y, double* x, csrmatrix& A);

// BCSR 4x4 format
void SpMV_BCSR(double* y, const double* x, const bcsr4x4_matrix& A);
void SpMV_BCSR_OPT(double* y, const double* x, const bcsr4x4_matrix& A);
void SpMV_BCSR_FMA(double* y, const double* x, const bcsr4x4_matrix& A);
void SpMV_BCSR_AVX2(double* y, const double* x, const bcsr4x4_matrix& A);

void flush_cache();

#endif
