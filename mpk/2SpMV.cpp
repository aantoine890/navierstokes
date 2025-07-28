#include "SpMV.h"

void orthogonalize(int nrow, const std::vector<double>& x, std::vector<double>& y, double alpha = 1e-8)
{
    volatile double beta = 0.0;
    for (int i = 0; i < nrow; ++i)
        beta += x[i] * y[i];

    for (int i = 0; i < nrow; ++i)
        y[i] -= alpha * beta * x[i];
}

void orthonormalize_against_basis(int nrow, std::vector<std::vector<double>>& basis, std::vector<double>& y) {
    for (const auto& x : basis) {
        double dot = 0.0;
        for (int i = 0; i < nrow; ++i)
            dot += y[i] * x[i];

        for (int i = 0; i < nrow; ++i)
            y[i] -= dot * x[i];
    }

    double norm = 0.0;
    for (int i = 0; i < nrow; ++i)
        norm += y[i] * y[i];
    norm = std::sqrt(norm);

}

void reset_vectors(std::vector<double>& x,
                   std::vector<double>& y,
                   std::vector<double>& w,
                   std::vector<double>& z,
                   double val_x = 1.0, double val_y = 0.0,
                   double val_w = 1.0, double val_z = 0.0) {
    std::fill(x.begin(), x.end(), val_x);
    std::fill(y.begin(), y.end(), val_y);
    std::fill(w.begin(), w.end(), val_w);
    std::fill(z.begin(), z.end(), val_z);
}


int main(int argc, char **argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
		return 1;
	}

	/* ------------------------------------------------------------------ */
	/*                      FILE PARSING (Matrix Market)                  */
	/* ------------------------------------------------------------------ */

	char buf[1024];
	int nrow, nnz;
	FILE *fp;
	int itmp, jtmp, ktmp;

	if ((fp = fopen(argv[1], "r")) == NULL)
	{
		fprintf(stderr, "fail to open %s\n", argv[1]);
		return 1;
	}

	// skip comments
	fgets(buf, 1024, fp);
	while (1)
	{
		fgets(buf, 1024, fp);
		if (buf[0] != '%')
		{
			sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
			nrow = itmp;
			nnz = ktmp;
			break;
		}
	}


	std::vector<int> irow(nnz), jcol(nnz);
	std::vector<double> val(nnz);

	for (int i = 0; i < nnz; i++)
	{
		float vtmp;
		fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
		irow[i] = itmp - 1; // 0-based
		jcol[i] = jtmp - 1;
		val[i] = (double)vtmp;
	}
	fclose(fp);

    printf("Matrix loaded: %d rows, %d nonzeros\n", nrow, nnz);

	/* ------------------------------------------------------------------ */
	/*                       INITIALIZATION                               */
	/* ------------------------------------------------------------------ */

	csrmatrix A;
	COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

    bcsr4x4_matrix A_bcsr;
	std::vector<std::list<std::pair<int, std::array<double, 16>>>> block_rows((nrow + 3) / 4);
	generate_BCSR4(&block_rows[0], nrow, nnz, &irow[0], &jcol[0], &val[0], A_bcsr);

    std::vector<double> base(nrow, 0.0);

	// Krylov Basis for orthogonalization
	std::vector<std::vector<double>> krylov_basis;
	for (int i = 0; i < 50; ++i) {
		std::vector<double> v(nrow);
		for (int j = 0; j < nrow; ++j)
			v[j] = std::sin(0.001 * j + i);
		krylov_basis.push_back(std::move(v));
	}

	/* ------------------------------------------------------------------ */
	/*                         SpMV benchmarking                          */
	/* ------------------------------------------------------------------ */

	std::vector<double> x(nrow, 1.0);     
	std::vector<double> y(nrow, 0.0);
	std::vector<double> w(nrow, 1.0);  
	std::vector<double> z(nrow, 0.0);

    // SpMV CSR
    auto start1 = std::chrono::high_resolution_clock::now();
	SpMV_CSR(y.data(), x.data(), A);
	auto end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);

    auto start2 = std::chrono::high_resolution_clock::now();
	SpMV_CSR(z.data(), w.data(), A);
	auto end2 = std::chrono::high_resolution_clock::now();

    long duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    long duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

	long duration_spmv_csr = duration1 + duration2;

    base = z;

	// printf("%.2f\n", (double)duration1/duration2);
    printf("2SpMV CSR : %8ld μs | ref | ref\n", duration_spmv_csr, duration1, duration2);

    // SpMV CSR OPT
	reset_vectors(x, y, w, z);

	flush_cache();

    start1 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_OPT(y.data(), x.data(), A);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);

    start2 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_OPT(z.data(), w.data(), A);
	end2 = std::chrono::high_resolution_clock::now();

    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_csr_opt = duration1 + duration2;

    printf("2SpMV CSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_opt, duration1, duration2, (double)duration_spmv_csr/duration_spmv_csr_opt,rel_error(base,z));

    // SpMV CSR FMA
	reset_vectors(x, y, w, z);

	flush_cache();
    
    start1 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_FMA(y.data(), x.data(), A);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);

    start2 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_OPT(z.data(), w.data(), A);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_csr_fma = duration1 + duration2;

    printf("2SpMV CSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_fma, duration1, duration2, (double)duration_spmv_csr/duration_spmv_csr_fma,rel_error(base,z));

    // SpMV CSR AVX2
	reset_vectors(x, y, w, z);

	flush_cache();
    
    start1 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_AVX2(y.data(), x.data(), A);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);

    start2 = std::chrono::high_resolution_clock::now();
	SpMV_CSR_AVX2(z.data(), w.data(), A);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_csr_avx2 = duration1 + duration2;

    printf("2SpMV CSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_csr_avx2, duration1, duration2, (double)duration_spmv_csr/duration_spmv_csr_avx2,rel_error(base,z));

    // SpMV BCSR
	reset_vectors(x, y, w, z);

	flush_cache();

    start1 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR(y.data(), x.data(), A_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);
    
    start2 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR(z.data(), w.data(), A_bcsr);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_bcsr = duration1 + duration2;

    printf("2SpMV BCSR : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr, duration1, duration2, (double)duration_spmv_csr/duration_spmv_bcsr,rel_error(base,z));

    // SpMV CSR OPT
	reset_vectors(x, y, w, z);

	flush_cache();

    start1 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_OPT(y.data(), x.data(), A_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);
    
    start2 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_OPT(z.data(), w.data(), A_bcsr);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_bcsr_opt = duration1 + duration2;

    printf("2SpMV BCSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_opt, duration1, duration2, (double)duration_spmv_csr/duration_spmv_bcsr_opt,rel_error(base,z));

    // SpMV BCSR FMA
	reset_vectors(x, y, w, z);
    
	flush_cache();

    start1 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_FMA(y.data(), x.data(), A_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);
    
    start2 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_FMA(z.data(), w.data(), A_bcsr);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_bcsr_fma = duration1 + duration2;

    printf("2SpMV BCSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_fma, duration1, duration2, (double)duration_spmv_csr/duration_spmv_bcsr_fma,rel_error(base,z));

    // SpMV BCSR AVX2
	reset_vectors(x, y, w, z);

	flush_cache();
    
    start1 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_AVX2(y.data(), x.data(), A_bcsr);
	end1 = std::chrono::high_resolution_clock::now();

    // orthonormalize_against_basis(nrow, krylov_basis, y);
    
    start2 = std::chrono::high_resolution_clock::now();
	SpMV_BCSR_AVX2(z.data(), w.data(), A_bcsr);
	end2 = std::chrono::high_resolution_clock::now();

	duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
	long duration_spmv_bcsr_avx2 = duration1 + duration2;

    printf("2SpMV BCSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spmv_bcsr_avx2, duration1, duration2, (double)duration_spmv_csr/duration_spmv_bcsr_avx2,rel_error(base,z));


}