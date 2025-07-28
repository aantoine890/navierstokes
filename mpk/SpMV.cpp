#include "SpMV.h"

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
			const double* blk = &A.coef[16 * ia];

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
			const double* blk = &A.coef[16 * ia];

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
			const double* blk = &A.coef[16 * ia];

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
            const double *blk = &A.coef[16 * ia];

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

// int main(int argc, char **argv) {
// 	if (argc < 2)
// 	{
// 		fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
// 		return 1;
// 	}

// 	/* ------------------------------------------------------------------ */
// 	/*                      FILE PARSING (Matrix Market)                  */
// 	/* ------------------------------------------------------------------ */

// 	char buf[1024];
// 	int nrow, nnz;
// 	FILE *fp;
// 	int itmp, jtmp, ktmp;

// 	if ((fp = fopen(argv[1], "r")) == NULL)
// 	{
// 		fprintf(stderr, "fail to open %s\n", argv[1]);
// 		return 1;
// 	}

// 	// skip comments
// 	fgets(buf, 1024, fp);
// 	while (1)
// 	{
// 		fgets(buf, 1024, fp);
// 		if (buf[0] != '%')
// 		{
// 			sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
// 			nrow = itmp;
// 			nnz = ktmp;
// 			break;
// 		}
// 	}


// 	std::vector<int> irow(nnz), jcol(nnz);
// 	std::vector<double> val(nnz);

// 	for (int i = 0; i < nnz; i++)
// 	{
// 		float vtmp;
// 		fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
// 		irow[i] = itmp - 1; // 0-based
// 		jcol[i] = jtmp - 1;
// 		val[i] = (double)vtmp;
// 	}
// 	fclose(fp);

//     printf("Matrix loaded: %d rows, %d nonzeros\n", nrow, nnz);

// 	/* ------------------------------------------------------------------ */
// 	/*                       INITIALIZATION                               */
// 	/* ------------------------------------------------------------------ */

// 	csrmatrix A;
// 	COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

//     bcsr4x4_matrix A_bcsr;
// 	std::vector<std::list<std::pair<int, std::array<double, 16>>>> block_rows((nrow + 3) / 4);
// 	generate_BCSR4(&block_rows[0], nrow, nnz, &irow[0], &jcol[0], &val[0], A_bcsr);

// 	std::vector<double> x(nrow, 1.0);     
// 	std::vector<double> y(nrow, 0.0);
//     std::vector<double> base(nrow, 0.0);

// 	/* ------------------------------------------------------------------ */
// 	/*                         SpMV benchmarking                          */
// 	/* ------------------------------------------------------------------ */

//     // SpMV CSR

// 	// flush_cache();

//     auto start = std::chrono::high_resolution_clock::now();
// 	SpMV_CSR(y.data(), x.data(), A);
// 	auto end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_csr = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     base = y;

//     printf("SpMV CSR : %8ld μs | ref | ref\n", duration_spmv_csr);

//     // SpMV CSR OPT
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();

//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_CSR_OPT(y.data(), x.data(), A);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_csr_opt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV CSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_opt, (double)duration_spmv_csr/duration_spmv_csr_opt,rel_error(base,y));

//     // SpMV CSR FMA
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();
    
//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_CSR_FMA(y.data(), x.data(), A);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_csr_fma = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV CSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_fma, (double)duration_spmv_csr/duration_spmv_csr_fma,rel_error(base,y));

//     // SpMV CSR AVX2
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();
    
//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_CSR_AVX2(y.data(), x.data(), A);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_csr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV CSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_avx2, (double)duration_spmv_csr/duration_spmv_csr_avx2,rel_error(base,y));

//     // SpMV BCSR
// 	std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();

//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_BCSR(y.data(), x.data(), A_bcsr);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_bcsr = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV BCSR : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr, (double)duration_spmv_csr/duration_spmv_bcsr,rel_error(base,y));

//     // SpMV CSR OPT
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();

//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_BCSR_OPT(y.data(), x.data(), A_bcsr);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_bcsr_opt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV BCSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_opt, (double)duration_spmv_csr/duration_spmv_bcsr_opt,rel_error(base,y));

//     // SpMV BCSR FMA
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

//     flush_cache();

//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_BCSR_FMA(y.data(), x.data(), A_bcsr);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_bcsr_fma = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV BCSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_fma, (double)duration_spmv_csr/duration_spmv_bcsr_fma,rel_error(base,y));

//     // SpMV BCSR AVX2
//     std::fill(x.begin(), x.end(), 1.0);
//     std::fill(y.begin(), y.end(), 0.0);

// 	flush_cache();    

//     start = std::chrono::high_resolution_clock::now();
// 	SpMV_BCSR_AVX2(y.data(), x.data(), A_bcsr);
// 	end = std::chrono::high_resolution_clock::now();

// 	long duration_spmv_bcsr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

//     printf("SpMV BCSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_avx2, (double)duration_spmv_csr/duration_spmv_bcsr_avx2,rel_error(base,y));


// }