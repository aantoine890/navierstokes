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


/* ----------- Matrix Handler ----------- */

void generate_CSR(std::list<int> *ind_cols_tmp, std::list<double> *val_tmp, int nrow, int nnz, int *irow, int *jcol, double *val)
{
	for (int i = 0; i < nnz; i++)
	{
		const int ii = irow[i];
		const int jj = jcol[i];
		if (ind_cols_tmp[ii].empty())
		{
			ind_cols_tmp[ii].push_back(jj);
			val_tmp[ii].push_back(val[i]);
		}
		else
		{
			if (ind_cols_tmp[ii].back() < jj)
			{
				ind_cols_tmp[ii].push_back(jj);
				val_tmp[ii].push_back(val[i]);
			}
			else
			{
				std::list<double>::iterator iv = val_tmp[ii].begin();
				std::list<int>::iterator it = ind_cols_tmp[ii].begin();
				for (; it != ind_cols_tmp[ii].end(); ++it, ++iv)
				{
					if (*it == jj)
					{
						break;
					}
					if (*it > jj)
					{
						ind_cols_tmp[ii].insert(it, jj);
						val_tmp[ii].insert(iv, val[i]);
						break;
					}
				}
			}
		}
	}
}

void generate_BCSR4(std::list<std::pair<int, std::array<double, 16>>> *block_rows, int nrow, int nnz, const int *irow, const int *jcol, const double *val, bcsr4x4_matrix &A)
{
	const int nblocks = nrow / 4;
	for (int k = 0; k < nnz; ++k)
	{
		int i = irow[k], j = jcol[k];
		double v = val[k];
		int bi = i / 4, bj = j / 4;
		int ii = i % 4, jj = j % 4;
		bool found = false;
		for (auto &pair : block_rows[bi])
		{
			if (pair.first == bj)
			{
				pair.second[4 * ii + jj] = v;
				found = true;
				break;
			}
		}
		if (!found)
		{
			std::array<double, 16> new_block = {};
			new_block[4 * ii + jj] = v;
			block_rows[bi].emplace_back(bj, new_block);
		}
	}
	A.nblocks = nblocks;
	A.ptrow.resize(nblocks + 1, 0);
	for (int bi = 0; bi < nblocks; ++bi)
	{
		A.ptrow[bi + 1] = A.ptrow[bi] + block_rows[bi].size();
		for (const auto &pair : block_rows[bi])
		{
			A.indcol.push_back(pair.first);
			A.coef.push_back(pair.second);
		}
	}
	A.nrows = nblocks;
}

void COO2CSR(csrmatrix &a, int nrow, int nnz, int *irow, int *jcol, double *val)
{
	a.n = nrow;
	a.nnz = nnz;
	a.ptrow.resize(nrow + 1);
	a.indcol.resize(nnz);
	a.coef.resize(nnz);
	std::vector<std::list<int>> ind_cols_tmp(nrow);
	std::vector<std::list<double>> val_tmp(nrow);

	// without adding diagonal nor symmetrize for PARDISO mtype = 11
	generate_CSR(&ind_cols_tmp[0], &val_tmp[0],
		     nrow, nnz,
		     &irow[0], &jcol[0], &val[0]);
	{
		int k = 0;
		a.ptrow[0] = 0;
		for (int i = 0; i < nrow; i++)
		{
			std::list<int>::iterator jt = ind_cols_tmp[i].begin();
			std::list<double>::iterator jv = val_tmp[i].begin();
			for (; jt != ind_cols_tmp[i].end(); ++jt, ++jv)
			{
				a.indcol[k] = (*jt);
				a.coef[k] = (*jv);
				k++;
			}
			a.ptrow[i + 1] = k;
		}
	}
}

/* ----------- Error Checking ----------- */

double norm2(const std::vector<double> &x) {
	double s = 0.0;
	for (double xi : x)
		s += xi * xi;
	return std::sqrt(s);
}

double rel_error(const std::vector<double> &ref, const std::vector<double> &test) {
	double s = 0.0;
	for (size_t i = 0; i < ref.size(); ++i)
		s += (ref[i] - test[i]) * (ref[i] - test[i]);
	return std::sqrt(s) / norm2(ref);
}

/* ----------- Interleave operation ----------- */

void orthogonalize(int nrow, const std::vector<double>& b, const std::vector<double>& x1, std::vector<double>& x3, double alpha = 1e-8) {
    volatile double beta = std::inner_product(b.begin(), b.end(), x1.begin(), 0.0);
    for (int i = 0; i < nrow; ++i) {
        x3[i] = x1[i] - alpha * beta * b[i];
    }
}

/* ----------- GenerateLayer ----------- */

void Generate1stlayer(std::vector<int> &ptrowend1, csrmatrix &A)
{
	int nrow = A.n;
	std::vector<int> mask(nrow, 0);
	ptrowend1.resize(A.nnz);
	for (int i = 0; i < nrow; i++)
	{
		for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++)
		{
			int j = A.indcol[ia];
			if (mask[j])
			{
				ptrowend1[ia] = A.ptrow[j];
			}
			else
			{
				ptrowend1[ia] = A.ptrow[j + 1];
				mask[j] = 1;
			}
		}
	}
}

/* ----------- SPMV CSR Kernels ----------- */

/* Pure sequential version without any optimization */
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpMV_CSR(double *y, double *x, csrmatrix &a)
{
	int nrow = a.n;
	const double zero(0.0);
	for (int i = 0; i < nrow; i++)
	{
		y[i] = zero;
		for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++)
		{
			int j = a.indcol[ia];
			y[i] += a.coef[ia] * x[j];
		}
	}
}

/* Sequential version using fma optimization (only) from the compiler */
__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpMV_CSR_OPT(double *y, double *x, csrmatrix &a)
{
	int nrow = a.n;
	const double zero(0.0);
	for (int i = 0; i < nrow; i++)
	{
		y[i] = zero;
		for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++)
		{
			int j = a.indcol[ia];
			y[i] += a.coef[ia] * x[j];
		}
	}
}

/* Sequential version using explicit written fma */
__attribute__((target("no-avx2")))
void SpMV_CSR_FMA(double *y, double *x, csrmatrix &a)
{
	int nrow = a.n;
	const double zero(0.0);

	for (int i = 0; i < nrow; i++)
	{
		y[i] = zero;
		for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++)
		{
			int j = a.indcol[ia];
			y[i] = __builtin_fma(a.coef[ia], x[j], y[i]);
		}
	}
}

/* Sequential version using AVX2 intrisinct */
__attribute__((target("avx2")))
void SpMV_CSR_AVX2(double *y, double *x, csrmatrix &a)
{
	int nrow = a.n;
	const double zero(0.0);

	for (int i = 0; i < nrow; i++)
	{
		y[i] = zero;
		__m256d sum = _mm256_setzero_pd();

		for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia += 4)
		{
			__m256d coef = _mm256_loadu_pd(&a.coef[ia]);
			__m256d x_vals = _mm256_set_pd(
			    x[a.indcol[ia + 3]],
			    x[a.indcol[ia + 2]],
			    x[a.indcol[ia + 1]],
			    x[a.indcol[ia]]);
			sum = _mm256_fmadd_pd(coef, x_vals, sum);
		}

		double temp[4];
		_mm256_storeu_pd(temp, sum);
		y[i] = temp[0] + temp[1] + temp[2] + temp[3];
	}
}

/* ----------- SPMV BCSR Kernels ----------- */

/* Pure sequential version using block 4x4 structure without any optimization */
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpMV_BCSR(double* y, const double* x, const bcsr4x4_matrix& A)
{
	int nblock_rows = A.nrows;
	const double zero(0.0);

	for (int bi = 0; bi < nblock_rows; bi++)
	{
		for (int i = 0; i < 4; i++)
		{
			y[4 * bi + i] = zero;
		}

		for (int ia = A.ptrow[bi]; ia < A.ptrow[bi + 1]; ia++)
		{
			int bj = A.indcol[ia];
			const double* blk = A.coef[ia].data();

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					y[4 * bi + i] += blk[4 * i + j] * x[4 * bj + j];
				}
			}
		}
	}
}

/* Sequential version using block 4x4 structure and fma optimization (only) from the compiler */
__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpMV_BCSR_OPT(double* y, const double* x, const bcsr4x4_matrix& A)
{
	int nblock_rows = A.nrows;
	const double zero(0.0);

	for (int bi = 0; bi < nblock_rows; bi++)
	{
		for (int i = 0; i < 4; i++)
		{
			y[4 * bi + i] = zero;
		}

		for (int ia = A.ptrow[bi]; ia < A.ptrow[bi + 1]; ia++)
		{
			int bj = A.indcol[ia];
			const double* blk = A.coef[ia].data();

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					y[4 * bi + i] += blk[4 * i + j] * x[4 * bj + j];
				}
			}
		}
	}
}

/* Sequential version using block 4x4 structure and explicit written fma */
__attribute__((target("no-avx2")))
void SpMV_BCSR_FMA(double* y, const double* x, const bcsr4x4_matrix& A)
{
	int nblock_rows = A.nrows;
	const double zero(0.0);

	for (int bi = 0; bi < nblock_rows; bi++)
	{
		for (int i = 0; i < 4; i++)
		{
			y[4 * bi + i] = zero;
		}

		for (int ia = A.ptrow[bi]; ia < A.ptrow[bi + 1]; ia++)
		{
			int bj = A.indcol[ia];
			const double* blk = A.coef[ia].data();

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					y[4 * bi + i] = __builtin_fma(blk[4 * i + j], x[4 * bj + j], y[4 * bi + i]);
				}
			}
		}
	}
}

/* Sequential version using block 4x4 structure and AVX2 intrisinct */
__attribute__((target("avx2,fma")))
void SpMV_BCSR_AVX2(double *y, const double *x, const bcsr4x4_matrix &A)
{
    const int nblock_rows = A.nrows;
    std::memset(y, 0, 4 * nblock_rows * sizeof(double));

    for (int bi = 0; bi < nblock_rows; ++bi)
    {
        __m256d acc = _mm256_setzero_pd();

        for (int ia = A.ptrow[bi]; ia < A.ptrow[bi + 1]; ++ia)
        {
            const int bj = A.indcol[ia];
            const double *blk = A.coef[ia].data();

            // Colonne j = 0
            __m256d a0 = _mm256_set_pd(blk[12], blk[8], blk[4], blk[0]);
            __m256d x0 = _mm256_set1_pd(x[4 * bj + 0]);
            acc = _mm256_fmadd_pd(a0, x0, acc);

            // Colonne j = 1
            __m256d a1 = _mm256_set_pd(blk[13], blk[9], blk[5], blk[1]);
            __m256d x1 = _mm256_set1_pd(x[4 * bj + 1]);
            acc = _mm256_fmadd_pd(a1, x1, acc);

            // Colonne j = 2
            __m256d a2 = _mm256_set_pd(blk[14], blk[10], blk[6], blk[2]);
            __m256d x2 = _mm256_set1_pd(x[4 * bj + 2]);
            acc = _mm256_fmadd_pd(a2, x2, acc);

            // Colonne j = 3
            __m256d a3 = _mm256_set_pd(blk[15], blk[11], blk[7], blk[3]);
            __m256d x3 = _mm256_set1_pd(x[4 * bj + 3]);
            acc = _mm256_fmadd_pd(a3, x3, acc);
        }

        _mm256_storeu_pd(&y[4 * bi], acc);
    }
}

/* ----------- SPM2V CSR Kernels ----------- */

/* Fused version without any optimization */
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V_CSR(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
  int nrow = A.n;
  for (int i = 0; i < nrow; i++) {
    z[i] = y[i] = 0.0;
  }
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	y[j] += A.coef[jb] * x[k];
      } // loop : jb
      z[i] += A.coef[ia] * y[j];
    } // loop : ia
  }   // loop : i
}

/* Fused version using fma optimization (only) from the compiler */
__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpM2V_CSR_OPT(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

/* Fused version using explicit written fma */
__attribute__((target("no-avx2")))
void SpM2V_CSR_FMA(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

__attribute__((target("avx2,fma")))
/* Fused version using AVX2 intrisinct */
void SpM2V_CSR_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

/* ----------- SPM2V BCSR Kernels ----------- */

/* Fused version using block 4x4 structure without any optimization */
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V_BCSR(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

/* Fused version using block 4x4 structure and fma optimization (only) from the compiler */
__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpM2V_BCSR_OPT(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

/* Fused version using block 4x4 structure and explicit written fma  */
__attribute__((target("no-avx2")))
void SpM2V_BCSR_FMA(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

/* Fused version using block 4x4 structure and AVX2 intrisinct  */
__attribute__((target("no-avx2")))
void SpM2V_BCSR_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{

}

int main(int argc, char **argv)
{

	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 
	/*                                                               FILE PARSING                                                                               */
	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */

	char fname[256];
	char buf[1024];
	int nrow, nnz;

	FILE *fp;
	int itmp, jtmp, ktmp;

	strcpy(fname, argv[1]);

	if ((fp = fopen(fname, "r")) == NULL)
	{
		fprintf(stderr, "fail to open %s\n", fname);
		return 1;
	}
	fgets(buf, 256, fp);

	while (1)
	{
		fgets(buf, 256, fp);
		if (buf[0] != '%')
		{
			sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
			nrow = itmp;
			nnz = ktmp;
			break;
		}
	}
	std::vector<int> irow(nnz);
	std::vector<int> jcol(nnz);
	std::vector<double> val(nnz);
	{
		int ii = 0;
		int itmp, jtmp;
		float vtmp;
		for (int i = 0; i < nnz; i++)
		{
			fscanf(fp, "%d\t%d\t%f", &itmp, &jtmp, &vtmp);
			irow[ii] = itmp - 1; // zero based
			jcol[ii] = jtmp - 1; // zero based
			val[ii] = (double)vtmp;
			ii++;
		}
		fprintf(stderr, "%d\n", ii);
		nnz = ii;
	}

	fclose(fp);

    /* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 
	/*                                                               INITIALIZATION                                                                             */
	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */

	csrmatrix a;
	COO2CSR(a, nrow, nnz, &irow[0], &jcol[0], &val[0]);

	bcsr4x4_matrix a_bcsr;
	std::vector<std::list<std::pair<int, std::array<double, 16>>>> block_rows((nrow + 3) / 4);
	generate_BCSR4(&block_rows[0], nrow, nnz, &irow[0], &jcol[0], &val[0], a_bcsr);

	std::vector<int> ptrowend1;
  	Generate1stlayer(ptrowend1, a);

	// CSR SpMV vectors
    std::vector<double> b(nrow, 0.0), c(nrow, 0.0), d(nrow, 0.0), e(nrow, 0.0);
    std::vector<double> x(nrow, 1.0), x1(nrow, 0.0), x2(nrow, 0.0), x3(nrow, 0.0);
    std::vector<double> y(nrow, 1.0), y1(nrow, 0.0), y2(nrow, 0.0), y3(nrow, 0.0);
	std::vector<double> w(nrow, 1.0), w1(nrow, 0.0), w2(nrow, 0.0), w3(nrow, 0.0);
    std::vector<double> z(nrow, 1.0), z1(nrow, 0.0), z2(nrow, 0.0), z3(nrow, 0.0);

	// BCSR SpMV vectors
	std::vector<double> bb(nrow, 0.0), cc(nrow, 0.0), dd(nrow, 0.0), ee(nrow, 0.0);
    std::vector<double> xx(nrow, 1.0), xx1(nrow, 0.0), xx2(nrow, 0.0), xx3(nrow, 0.0);
    std::vector<double> yy(nrow, 1.0), yy1(nrow, 0.0), yy2(nrow, 0.0), yy3(nrow, 0.0);
	std::vector<double> ww(nrow, 1.0), ww1(nrow, 0.0), ww2(nrow, 0.0), ww3(nrow, 0.0);
    std::vector<double> zz(nrow, 1.0), zz1(nrow, 0.0), zz2(nrow, 0.0), zz3(nrow, 0.0);

	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 
	/*                                                               BENCHMARK 2x SpMV                                                                          */
	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 

    // Warm-up
    SpMV_CSR(&b[0], &x[0], a);

    // SpMV CSR
    auto start1 = std::chrono::high_resolution_clock::now();
    SpMV_CSR(&x1[0], &b[0], a);
	auto end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, b, x1, x3);

	auto start2 = std::chrono::high_resolution_clock::now();
    SpMV_CSR(&x2[0], &x3[0], a);
    auto end2 = std::chrono::high_resolution_clock::now();

	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_csr_seq = duration1 + duration2;
    

    printf("1. 2x SpMV CSR :        %8ld μs |    ref   |        ref\n", duration_spmv_csr_seq);

	// Warm-up
    SpMV_CSR_OPT(&c[0], &y[0], a);

	// SpMV CSR OPT
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_OPT(&y1[0], &c[0], a);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, c, y1, y3);
	
	start2 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_OPT(&y2[0], &y3[0], a);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_csr_seq_opt = duration1 + duration2;

	printf("2. 2x SpMV CSR OPT :    %8ld μs |   %.2fx  | rel err = %.3e\n", duration_spmv_csr_seq_opt, (double)duration_spmv_csr_seq/duration_spmv_csr_seq_opt, rel_error(x2,y2));

	// Warm-up
    SpMV_CSR_FMA(&d[0], &w[0], a);

	// SpMV CSR FMA
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_FMA(&w1[0], &d[0], a);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, d, w1, w3);

	start2 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_FMA(&w2[0], &w3[0], a);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_csr_seq_fma = duration1 + duration2;

	printf("3. 2x SpMV CSR FMA :    %8ld μs |   %.2fx  | rel err = %.3e\n", duration_spmv_csr_seq_fma, (double)duration_spmv_csr_seq/duration_spmv_csr_seq_fma, rel_error(x2,w2));
	
	// Warm-up
    SpMV_CSR_AVX2(&e[0], &z[0], a);

	// SpMV CSR AVX2
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_AVX2(&z1[0], &e[0], a);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, e, z1, z3);

	start2 = std::chrono::high_resolution_clock::now();
    SpMV_CSR_AVX2(&z2[0], &z3[0], a);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_csr_seq_avx2 = duration1 + duration2;

	printf("4. 2x SpMV CSR AVX2 :   %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spmv_csr_seq_avx2, (double)duration_spmv_csr_seq/duration_spmv_csr_seq_avx2, rel_error(x2,z2));

	// Warm-up
    SpMV_BCSR(&bb[0], &xx[0], a_bcsr);

	// SpMV BCSR
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR(&xx1[0], &bb[0], a_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, bb, xx1, xx3);
	
	start2 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR(&xx2[0], &xx3[0], a_bcsr);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_bcsr_seq = duration1 + duration2;

	printf("5. 2x SpMV BCSR :       %8ld μs |   %.2fx  | rel err = %.3e\n", duration_spmv_bcsr_seq, (double)duration_spmv_csr_seq/duration_spmv_bcsr_seq, rel_error(x2,xx2));

	// Warm-up
    SpMV_BCSR_OPT(&cc[0], &yy[0], a_bcsr);

	// SpMV BCSR OPT
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_OPT(&yy1[0], &cc[0], a_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, cc, yy1, yy3);
	
	start2 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_OPT(&yy2[0], &yy3[0], a_bcsr);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_bcsr_seq_opt = duration1 + duration2;

	printf("6. 2x SpMV BCSR OPT :   %8ld μs |   %.2fx  | rel err = %.3e\n", duration_spmv_bcsr_seq_opt, (double)duration_spmv_csr_seq/duration_spmv_bcsr_seq_opt, rel_error(x2,yy2));

	// Warm-up
    SpMV_BCSR_FMA(&dd[0], &ww[0], a_bcsr);

	// SpMV BCSR FMA 
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_FMA(&ww1[0], &dd[0], a_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, dd, ww1, ww3);
	
	start2 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_FMA(&ww2[0], &ww3[0], a_bcsr);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_bcsr_seq_fma = duration1 + duration2;

	printf("7. 2x SpMV BCSR FMA :   %8ld μs |   %.2fx  | rel err = %.3e\n", duration_spmv_bcsr_seq_fma, (double)duration_spmv_csr_seq/duration_spmv_bcsr_seq_fma, rel_error(x2,ww2));

	// Warm-up
    SpMV_BCSR_AVX2(&ee[0], &zz[0], a_bcsr);

	// SpMV BCSR AVX2
    start1 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_AVX2(&zz1[0], &ee[0], a_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

	orthogonalize(nrow, ee, zz1, zz3);
	
	start2 = std::chrono::high_resolution_clock::now();
    SpMV_BCSR_AVX2(&zz2[0], &zz3[0], a_bcsr);
    end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
	duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	auto duration_spmv_bcsr_seq_avx2 = duration1 + duration2;

	printf("8. 2x SpMV BCSR AVX2 :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spmv_bcsr_seq_avx2, (double)duration_spmv_csr_seq/duration_spmv_bcsr_seq_avx2, rel_error(x2,zz2));

	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 
	/*                                                               BENCHMARK SpM2V                                                                            */
	/* -------------------------------------------------------------------------------------------------------------------------------------------------------- */ 

	// // SpM2V CSR
	// auto start = std::chrono::high_resolution_clock::now();

	// auto end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_csr =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("9. SpM2V CSR :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_csr, (double)duration_spmv_csr_seq/duration_spm2V_csr, rel_error(x2,));

	// // SpM2V CSR OPT
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_csr_opt =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("10. SpM2V CSR OPT :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_csr_opt, (double)duration_spmv_csr_seq/duration_spm2V_csr_opt, rel_error(x2,));

	// // SpM2V CSR FMA
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_csr_fma =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("11. SpM2V CSR FMA :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_csr_fma, (double)duration_spmv_csr_seq/duration_spm2V_csr_fma, rel_error(x2,));

	// // SpM2V CSR AVX2
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_csr_avx2 =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("12. SpM2V CSR AVX2 :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_csr_avx2, (double)duration_spmv_csr_seq/duration_spm2V_csr_avx2, rel_error(x2,));

	// // SpM2V BCSR
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_bcsr =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("13. SpM2V BCSR :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_bcsr, (double)duration_spmv_csr_seq/duration_spm2V_bcsr, rel_error(x2,));

	// // SpM2V BCSR OPT
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_bcsr_opt =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("14. SpM2V BCSR OPT :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_bcsr_opt, (double)duration_spmv_csr_seq/duration_spm2V_bcsr_opt, rel_error(x2,));

	// // SpM2V BCSR FMA
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_bcsr_fma =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("15. SpM2V BCSR FMA :  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_bcsr_fma, (double)duration_spmv_csr_seq/duration_spm2V_bcsr_fma, rel_error(x2,));

	// // SpM2V BCSR AVX2
	// start = std::chrono::high_resolution_clock::now();

	// end = std::chrono::high_resolution_clock::now();
	// auto duration_spm2V_bcsr_avx2 =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	// printf("16. SpM2V BCSR AVX2:  %8ld μs |  %.2fx  | rel err = %.3e\n", duration_spm2V_bcsr_avx2, (double)duration_spmv_csr_seq/duration_spm2V_bcsr_avx2, rel_error(x2,));



	return 0;
}