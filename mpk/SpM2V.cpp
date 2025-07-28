#include "SpMV.h"

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

void Generate1stlayer_BCSR4(std::vector<int>& ptrowendB, const bcsr4x4_matrix& A)
{
    int nblocks = A.nrows;
    std::vector<int> mask(nblocks, 0);
    ptrowendB.resize(A.indcol.size()); // nb de blocs non nuls

    for (int bi = 0; bi < nblocks; ++bi) {
        for (int m = A.ptrow[bi]; m < A.ptrow[bi + 1]; ++m) {
            int bj = A.indcol[m]; // bloc colonne

            if (mask[bj]) {
                ptrowendB[m] = A.ptrow[bj]; // ligne bj déjà traitée
            } else {
                ptrowendB[m] = A.ptrow[bj + 1]; // ligne bj non traitée
                mask[bj] = 1;
            }
        }
    }
}

void reset_vectors(std::vector<double>& x,
                   std::vector<double>& y,
                   std::vector<double>& z,
                   double val_x = 1.0, double val_y = 0.0, double val_z = 0.0) {
    std::fill(x.begin(), x.end(), val_x);
    std::fill(y.begin(), y.end(), val_y);
    std::fill(z.begin(), z.end(), val_z);
}

/* ----------- SPM2V CSR Kernels ----------- */

/* Fused version without any optimization */
// __attribute__((target("no-sse,no-avx2,no-fma")))
// void SpM2V_CSR(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//   int nrow = A.n;
//   for (int i = 0; i < nrow; i++) {
//     z[i] = y[i] = 0.0;
//   }
//   for (int i = 0; i < nrow; i++) {
//     for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//       int j = A.indcol[ia];
//       for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
//         int k = A.indcol[jb];
//         y[j] += A.coef[jb] * x[k];
//       } // loop : jb
//       z[i] += A.coef[ia] * y[j];
//     } // loop : ia
//   }   // loop : i
// }

__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V_CSR(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    int nrow = A.n;
    
    // Initialisation optimisée
    std::fill(z, z + nrow, 0.0);
    std::fill(y, y + nrow, 0.0);
    
    // Cache des pointeurs pour éviter les déréférencements répétés
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    
    for (int i = 0; i < nrow; i++) {
        int row_start = ptrow[i];
        int row_end = ptrow[i + 1];
        
        for (int ia = row_start; ia < row_end; ia++) {
            int j = indcol[ia];
            double coef_ia = coef[ia];  // Cache du coefficient
            
            int col_start = ptrow[j];
            int col_end = ptrowend1[ia];
            
            for (int jb = col_start; jb < col_end; jb++) {
                int k = indcol[jb];
                y[j] += coef[jb] * x[k];
            }
            
            z[i] += coef_ia * y[j];
        }
    }
}

/* Fused version using fma optimization (only) from the compiler */
// __attribute__((target("fma")))
// __attribute__((target("no-avx2")))
// void SpM2V_CSR_OPT(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//   int nrow = A.n;
//   for (int i = 0; i < nrow; i++) {
//     z[i] = y[i] = 0.0;
//   }
//   for (int i = 0; i < nrow; i++) {
//     for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//       int j = A.indcol[ia];
//       for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
// 	int k = A.indcol[jb];
// 	y[j] += A.coef[jb] * x[k];
//       } // loop : jb
//       z[i] += A.coef[ia] * y[j];
//     } // loop : ia
//   }   // loop : i
// }

__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpM2V_CSR_OPT(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    int nrow = A.n;
    
    // Initialisation optimisée
    std::fill(z, z + nrow, 0.0);
    std::fill(y, y + nrow, 0.0);
    
    // Cache des pointeurs pour éviter les déréférencements répétés
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    
    for (int i = 0; i < nrow; i++) {
        int row_start = ptrow[i];
        int row_end = ptrow[i + 1];
        
        for (int ia = row_start; ia < row_end; ia++) {
            int j = indcol[ia];
            double coef_ia = coef[ia];  // Cache du coefficient
            
            int col_start = ptrow[j];
            int col_end = ptrowend1[ia];
            
            for (int jb = col_start; jb < col_end; jb++) {
                int k = indcol[jb];
                y[j] += coef[jb] * x[k];
            }
            
            z[i] += coef_ia * y[j];
        }
    }
}

/* Fused version using explicit written fma */
// __attribute__((target("no-avx2")))
// void SpM2V_CSR_FMA(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//   int nrow = A.n;
//   for (int i = 0; i < nrow; i++) {
//     z[i] = y[i] = 0.0;
//   }

//   for (int i = 0; i < nrow; i++) {
//     for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//       int j = A.indcol[ia];

//       for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
//         int k = A.indcol[jb];
//         y[j] = __builtin_fma(A.coef[jb], x[k], y[j]);
//       }

//       z[i] = __builtin_fma(A.coef[ia], y[j], z[i]);
//     }
//   }
// }

__attribute__((target("no-avx2")))
void SpM2V_CSR_FMA(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    int nrow = A.n;
    
    // Initialisation optimisée
    std::fill(z, z + nrow, 0.0);
    std::fill(y, y + nrow, 0.0);
    
    // Cache des pointeurs pour éviter les déréférencements répétés
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    
    for (int i = 0; i < nrow; i++) {
        int row_start = ptrow[i];
        int row_end = ptrow[i + 1];
        
        for (int ia = row_start; ia < row_end; ia++) {
            int j = indcol[ia];
            double coef_ia = coef[ia];  // Cache du coefficient
            
            int col_start = ptrow[j];
            int col_end = ptrowend1[ia];
            
            for (int jb = col_start; jb < col_end; jb++) {
                int k = indcol[jb];
                y[j] += __builtin_fma(coef[jb], x[k], y[j]);
            }
            
            z[i] += __builtin_fma(coef_ia, y[j], z[i]);
        }
    }
}


/* Fused version using AVX2 intrisinct */
// __attribute__((target("avx2,fma")))
// void SpM2V_CSR_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//     const int nrow = A.n;

//     for (int i = 0; i < nrow; ++i)
//         z[i] = y[i] = 0.0;

//     for (int i = 0; i < nrow; ++i) {
//         for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ++ia) {
//         int j = A.indcol[ia];

//         int jb_start = A.ptrow[j];
//         int jb_end   = ptrowend1[ia];

//         __m256d yj_vec = _mm256_setzero_pd();

//             for (int jb = jb_start; jb < jb_end; jb += 4) {
//                 // Charger A.coef[jb .. jb+3]
//                 __m256d a_vals = _mm256_set_pd(A.coef[jb + 3],
//                                             A.coef[jb + 2],
//                                             A.coef[jb + 1],
//                                             A.coef[jb + 0]);

//                 // Charger x[k] manuellement
//                 alignas(32) double x_vals_arr[4] = {
//                     x[A.indcol[jb + 0]],
//                     x[A.indcol[jb + 1]],
//                     x[A.indcol[jb + 2]],
//                     x[A.indcol[jb + 3]],
//                 };
//                 __m256d x_vals = _mm256_load_pd(x_vals_arr);  
//                 // FMA : yj_vec += a_vals * x_vals
//                 yj_vec = _mm256_fmadd_pd(a_vals, x_vals, yj_vec);
//             }

//             // Réduction SIMD → scalaire
//             alignas(32) double tmp[4];
//             _mm256_store_pd(tmp, yj_vec);
//             y[j] += tmp[0] + tmp[1] + tmp[2] + tmp[3];

//             // FMA scalaire finale pour z[i]
//             z[i] = __builtin_fma(A.coef[ia], y[j], z[i]);
//         }
//     }
// }

__attribute__((target("avx2,fma")))
void SpM2V_CSR_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    const int nrow = A.n;

    // Initialisation optimisée
    std::fill(z, z + nrow, 0.0);
    std::fill(y, y + nrow, 0.0);

    // Mise en cache des pointeurs
    const int* ptrow   = A.ptrow.data();
    const int* indcol  = A.indcol.data();
    const double* coef = A.coef.data();

    for (int i = 0; i < nrow; ++i) {
        int row_start = ptrow[i];
        int row_end   = ptrow[i + 1];

        for (int ia = row_start; ia < row_end; ++ia) {
            int j = indcol[ia];
            double coef_ia = coef[ia];

            int col_start = ptrow[j];
            int col_end   = ptrowend1[ia];

            __m256d yj_vec = _mm256_setzero_pd();

            int jb;
            for (jb = col_start; jb <= col_end - 4; jb += 4) {
                // Charger A.coef[jb .. jb+3]
                __m256d a_vals = _mm256_loadu_pd(&coef[jb]);

                // Charger x[indcol[jb..jb+3]] dans un tableau temporaire aligné
                alignas(32) double x_vals_arr[4] = {
                    x[indcol[jb + 0]],
                    x[indcol[jb + 1]],
                    x[indcol[jb + 2]],
                    x[indcol[jb + 3]],
                };
                __m256d x_vals = _mm256_load_pd(x_vals_arr);

                // FMA
                yj_vec = _mm256_fmadd_pd(a_vals, x_vals, yj_vec);
            }

            // Réduction SIMD
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, yj_vec);
            y[j] += tmp[0] + tmp[1] + tmp[2] + tmp[3];

            // Produit scalaire final
            z[i] = __builtin_fma(coef_ia, y[j], z[i]);
        }
    }
}

/* ----------- SPM2V BCSR Kernels ----------- */

/* Fused version using block 4x4 structure without any optimization */
// __attribute__((target("no-sse,no-avx2,no-fma")))
// void SpM2V_BCSR(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
// {
//     const int nblocks = A.nrows;

//     // Initialisation des vecteurs
//     for (int i = 0; i < 4 * nblocks; ++i)
//         z[i] = y[i] = 0.0;

//     for (int bi = 0; bi < nblocks; ++bi) {
//         for (int m = A.ptrow[bi]; m < A.ptrow[bi + 1]; ++m) {
//             int bj = A.indcol[m];

//             // --- 1. y[bj] = A[bj, :] * x
//             for (int l = A.ptrow[bj]; l < ptrowendB[m]; ++l) {
//                 int bk = A.indcol[l];
//                 const double* B = &A.coef[16 * l];
//                 const double* xk = &x[4 * bk];
//                 double* yj = &y[4 * bj];

//                 // yj += B * xk
//                 for (int r = 0; r < 4; ++r)
//                     for (int c = 0; c < 4; ++c)
//                         yj[r] += B[4 * r + c] * xk[c];
//             }

//             // --- 2. z[bi] += A[bi, bj] * y[bj]
//             const double* Aij = &A.coef[16 * m];;
//             const double* yj = &y[4 * bj];
//             double* zi = &z[4 * bi];

//             for (int r = 0; r < 4; ++r)
//                 for (int c = 0; c < 4; ++c)
//                     zi[r] += Aij[4 * r + c] * yj[c];
//         }
//     }
// }

__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V_BCSR(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
{
    const int nblocks = A.nrows;
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();

    // Initialisation optimisée
    std::fill(z, z + 4 * nblocks, 0.0);
    std::fill(y, y + 4 * nblocks, 0.0);

    for (int bi = 0; bi < nblocks; ++bi) {
        int row_start = ptrow[bi];
        int row_end   = ptrow[bi + 1];

        for (int m = row_start; m < row_end; ++m) {
            int bj = indcol[m];
            double* yj = &y[4 * bj];

            // 1. y[bj] += A[bj, bk] * x[bk]
            int col_start = ptrow[bj];
            int col_end   = ptrowendB[m];

            for (int l = col_start; l < col_end; ++l) {
                int bk = indcol[l];
                const double* B  = &coef[16 * l];
                const double* xk = &x[4 * bk];
                

                // yj += B * xk
                for (int r = 0; r < 4; ++r) {
                    // double sum = yj[r];
                    // for (int c = 0; c < 4; ++c)
                    //     sum += B[4 * r + c] * xk[c];
                    // yj[r] = sum;
                    for (int c = 0; c < 4; ++c)
                        yj[r] += B[4 * r + c] * xk[c];
                }
            }

            // 2. z[bi] += A[bi, bj] * y[bj]
            const double* Aij = &coef[16 * m];
            // const double* yj  = &y[4 * bj];
            double* zi        = &z[4 * bi];

            for (int r = 0; r < 4; ++r) {
                // double sum = zi[r];
                // for (int c = 0; c < 4; ++c)
                //     sum += Aij[4 * r + c] * yj[c];
                // zi[r] = sum;
                for (int c = 0; c < 4; ++c)
                    zi[r] += Aij[4 * r + c] * yj[c];
            }
        }
    }
}

/* Fused version using block 4x4 structure and fma optimization (only) from the compiler */
// __attribute__((target("fma")))
// __attribute__((target("no-avx2")))
// void SpM2V_BCSR_OPT(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
// {
//     const int nblocks = A.nrows;

//     // Initialisation des vecteurs
//     for (int i = 0; i < 4 * nblocks; ++i)
//         z[i] = y[i] = 0.0;

//     for (int bi = 0; bi < nblocks; ++bi) {
//         for (int m = A.ptrow[bi]; m < A.ptrow[bi + 1]; ++m) {
//             int bj = A.indcol[m];

//             // --- 1. y[bj] = A[bj, :] * x
//             for (int l = A.ptrow[bj]; l < ptrowendB[m]; ++l) {
//                 int bk = A.indcol[l];
//                 const double* B = &A.coef[16 * l];
//                 const double* xk = &x[4 * bk];
//                 double* yj = &y[4 * bj];

//                 // yj += B * xk
//                 for (int r = 0; r < 4; ++r)
//                     for (int c = 0; c < 4; ++c)
//                         yj[r] += B[4 * r + c] * xk[c];
//             }

//             // --- 2. z[bi] += A[bi, bj] * y[bj]
//             const double* Aij = &A.coef[16 * m];
//             const double* yj = &y[4 * bj];
//             double* zi = &z[4 * bi];

//             for (int r = 0; r < 4; ++r)
//                 for (int c = 0; c < 4; ++c)
//                     zi[r] += Aij[4 * r + c] * yj[c];
//         }
//     }
// }

__attribute__((target("fma")))
__attribute__((target("no-avx2")))
void SpM2V_BCSR_OPT(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
{
    const int nblocks = A.nrows;
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();

    // Initialisation optimisée
    std::fill(z, z + 4 * nblocks, 0.0);
    std::fill(y, y + 4 * nblocks, 0.0);

    for (int bi = 0; bi < nblocks; ++bi) {
        int row_start = ptrow[bi];
        int row_end   = ptrow[bi + 1];

        for (int m = row_start; m < row_end; ++m) {
            int bj = indcol[m];

            // 1. y[bj] += A[bj, bk] * x[bk]
            int col_start = ptrow[bj];
            int col_end   = ptrowendB[m];

            for (int l = col_start; l < col_end; ++l) {
                int bk = indcol[l];
                const double* B  = &coef[16 * l];
                const double* xk = &x[4 * bk];
                double* yj       = &y[4 * bj];

                // yj += B * xk
                for (int r = 0; r < 4; ++r) {
                    double sum = 0.0;
                    for (int c = 0; c < 4; ++c)
                        sum += B[4 * r + c] * xk[c];
                    yj[r] += sum;
                }
            }

            // 2. z[bi] += A[bi, bj] * y[bj]
            const double* Aij = &coef[16 * m];
            const double* yj  = &y[4 * bj];
            double* zi        = &z[4 * bi];

            for (int r = 0; r < 4; ++r) {
                double sum = 0.0;
                for (int c = 0; c < 4; ++c)
                    sum += Aij[4 * r + c] * yj[c];
                zi[r] += sum;
            }
        }
    }
}


/* Fused version using block 4x4 structure and explicit written fma  */
// __attribute__((target("no-avx2")))
// void SpM2V_BCSR_FMA(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
// {
//     const int nblocks = A.nrows;

//     // Initialisation des vecteurs
//     for (int i = 0; i < 4 * nblocks; ++i)
//         z[i] = y[i] = 0.0;

//     for (int bi = 0; bi < nblocks; ++bi) {
//         for (int m = A.ptrow[bi]; m < A.ptrow[bi + 1]; ++m) {
//             int bj = A.indcol[m];

//             // --- 1. y[bj] = A[bj, :] * x
//             for (int l = A.ptrow[bj]; l < ptrowendB[m]; ++l) {
//                 int bk = A.indcol[l];
//                 const double* B = &A.coef[16 * l];
//                 const double* xk = &x[4 * bk];
//                 double* yj = &y[4 * bj];

//                 for (int r = 0; r < 4; ++r)
//                     for (int c = 0; c < 4; ++c)
//                         yj[r] = __builtin_fma(B[4 * r + c], xk[c], yj[r]);
//             }

//             // --- 2. z[bi] += A[bi, bj] * y[bj]
//             const double* Aij = &A.coef[16 * m];
//             const double* yj = &y[4 * bj];
//             double* zi = &z[4 * bi];

//             for (int r = 0; r < 4; ++r)
//                 for (int c = 0; c < 4; ++c)
//                     zi[r] = __builtin_fma(Aij[4 * r + c], yj[c], zi[r]);
//         }
//     }
// }

__attribute__((target("no-avx2")))
void SpM2V_BCSR_FMA(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
{
    const int nblocks = A.nrows;

    // Mise en cache des pointeurs
    const int* ptrow   = A.ptrow.data();
    const int* indcol  = A.indcol.data();
    const double* coef = A.coef.data();

    // Initialisation optimisée
    std::fill(z, z + 4 * nblocks, 0.0);
    std::fill(y, y + 4 * nblocks, 0.0);

    for (int bi = 0; bi < nblocks; ++bi) {
        int row_start = ptrow[bi];
        int row_end   = ptrow[bi + 1];

        for (int m = row_start; m < row_end; ++m) {
            int bj = indcol[m];

            // 1. y[bj] += A[bj, bk] * x[bk]
            int col_start = ptrow[bj];
            int col_end   = ptrowendB[m];

            for (int l = col_start; l < col_end; ++l) {
                int bk = indcol[l];
                const double* B  = &coef[16 * l];
                const double* xk = &x[4 * bk];
                double* yj       = &y[4 * bj];

                for (int r = 0; r < 4; ++r) {
                    double acc = yj[r];
                    for (int c = 0; c < 4; ++c)
                        acc = __builtin_fma(B[4 * r + c], xk[c], acc);
                    yj[r] = acc;
                }
            }

            // 2. z[bi] += A[bi, bj] * y[bj]
            const double* Aij = &coef[16 * m];
            const double* yj  = &y[4 * bj];
            double* zi        = &z[4 * bi];

            for (int r = 0; r < 4; ++r) {
                double acc = zi[r];
                for (int c = 0; c < 4; ++c)
                    acc = __builtin_fma(Aij[4 * r + c], yj[c], acc);
                zi[r] = acc;
            }
        }
    }
}

/* Fused version using block 4x4 structure and AVX2 intrisinct  */
// __attribute__((target("avx2,fma")))
// void SpM2V_BCSR_AVX2(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
// {
//     const int nblocks = A.nrows;
    
//     // Initialisation des vecteurs
//     for (int i = 0; i < 4 * nblocks; ++i)
//         z[i] = y[i] = 0.0;
    
//     for (int bi = 0; bi < nblocks; ++bi) {
//         for (int m = A.ptrow[bi]; m < A.ptrow[bi + 1]; ++m) {
//             int bj = A.indcol[m];
            
//             // --- 1. y[bj] = A[bj, :] * x avec FMA
//             for (int l = A.ptrow[bj]; l < ptrowendB[m]; ++l) {
//                 int bk = A.indcol[l];
//                 const double* B = &A.coef[16 * l];
//                 __m256d xvec = _mm256_loadu_pd(&x[4 * bk]);
                
//                 for (int r = 0; r < 4; ++r) {
//                     __m256d brow = _mm256_loadu_pd(&B[4 * r]);
//                     __m256d zero = _mm256_setzero_pd();
                    
//                     // FMA: prod = brow * xvec + 0
//                     __m256d prod = _mm256_fmadd_pd(brow, xvec, zero);
                    
//                     // Réduction simple
//                     double temp[4];
//                     _mm256_storeu_pd(temp, prod);
//                     y[4 * bj + r] += temp[0] + temp[1] + temp[2] + temp[3];
//                 }
//             }
            
//             // --- 2. z[bi] += A[bi, bj] * y[bj] avec FMA
//             const double* Aij = &A.coef[16 * m];
//             __m256d yvec = _mm256_loadu_pd(&y[4 * bj]);
            
//             for (int r = 0; r < 4; ++r) {
//                 __m256d arow = _mm256_loadu_pd(&Aij[4 * r]);
//                 __m256d zero = _mm256_setzero_pd();
                
//                 // FMA: prod = arow * yvec + 0
//                 __m256d prod = _mm256_fmadd_pd(arow, yvec, zero);
                
//                 // Réduction simple
//                 double temp[4];
//                 _mm256_storeu_pd(temp, prod);
//                 z[4 * bi + r] += temp[0] + temp[1] + temp[2] + temp[3];
//             }
//         }
//     }
// }

__attribute__((target("avx2,fma")))
void SpM2V_BCSR_AVX2(double *z, double *y, double *x, bcsr4x4_matrix &A, std::vector<int> &ptrowendB)
{
    const int nblocks = A.nrows;

    // Initialisation optimisée
    std::fill(z, z + 4 * nblocks, 0.0);
    std::fill(y, y + 4 * nblocks, 0.0);

    // Mise en cache des pointeurs
    const int* ptrow   = A.ptrow.data();
    const int* indcol  = A.indcol.data();
    const double* coef = A.coef.data();

    for (int bi = 0; bi < nblocks; ++bi) {
        int row_start = ptrow[bi];
        int row_end   = ptrow[bi + 1];
        __m256d accz = _mm256_setzero_pd();

        for (int m = row_start; m < row_end; ++m) {
            int bj = indcol[m];

            // --- 1. y[bj] += A[bj, bk] * x[bk]
            int col_start = ptrow[bj];
            int col_end   = ptrowendB[m];
            __m256d accy = _mm256_loadu_pd(&y[4 * bj]);
            
#if 0   // Ancienne version not colonne 11/07/2025
            for (int l = col_start; l < col_end; ++l) {
                int bk = indcol[l];
                const double* B    = &coef[16 * l];
                const double* xk   = &x[4 * bk];
                double* yj         = &y[4 * bj];

                __m256d xvec = _mm256_loadu_pd(xk);

                for (int r = 0; r < 4; ++r) {
                    const double* brow = &B[4 * r];
                    __m256d arow = _mm256_loadu_pd(brow);
                    __m256d prod = _mm256_fmadd_pd(arow, xvec, _mm256_setzero_pd());

                    // Réduction horizontale
                    alignas(32) double tmp[4];
                    _mm256_store_pd(tmp, prod);
                    yj[r] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
                }
            }
            // --- 2. z[bi] += A[bi, bj] * y[bj]
            const double* Aij = &coef[16 * m];
            const double* yj  = &y[4 * bj];
            double* zi        = &z[4 * bi];

            __m256d yvec = _mm256_loadu_pd(yj);
        
            for (int r = 0; r < 4; ++r) {
                const double* arow = &Aij[4 * r];
                __m256d avec = _mm256_loadu_pd(arow);
                __m256d prod = _mm256_fmadd_pd(avec, yvec, _mm256_setzero_pd());

                alignas(32) double tmp[4];
                _mm256_store_pd(tmp, prod);
                zi[r] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
#else       
            for (int l = col_start; l < col_end; ++l) {
                int bk = indcol[l];
                const double *blk = &A.coef[16 * l];
                // Colonne j = 0
                __m256d a0 = _mm256_set_pd(blk[12], blk[8], blk[4], blk[0]);
                __m256d x0 = _mm256_set1_pd(x[4 * bk + 0]);
                accy = _mm256_fmadd_pd(a0, x0, accy);

                // Colonne j = 1
                __m256d a1 = _mm256_set_pd(blk[13], blk[9], blk[5], blk[1]);
                __m256d x1 = _mm256_set1_pd(x[4 * bk + 1]);
                accy = _mm256_fmadd_pd(a1, x1, accy);

                // Colonne j = 2
                __m256d a2 = _mm256_set_pd(blk[14], blk[10], blk[6], blk[2]);
                __m256d x2 = _mm256_set1_pd(x[4 * bk + 2]);
                accy = _mm256_fmadd_pd(a2, x2, accy);

                // Colonne j = 3
                __m256d a3 = _mm256_set_pd(blk[15], blk[11], blk[7], blk[3]);
                __m256d x3 = _mm256_set1_pd(x[4 * bk + 3]);
                accy = _mm256_fmadd_pd(a3, x3, accy);
            }

            _mm256_storeu_pd(&y[4 * bj], accy);

            

            {   
                
                const double *blk = &A.coef[16 * m];
                // Colonne j = 0
                __m256d a0 = _mm256_set_pd(blk[12], blk[8], blk[4], blk[0]);
                __m256d x0 = _mm256_set1_pd(y[4 * bj + 0]);
                accz = _mm256_fmadd_pd(a0, x0, accz);

                // Colonne j = 1
                __m256d a1 = _mm256_set_pd(blk[13], blk[9], blk[5], blk[1]);
                __m256d x1 = _mm256_set1_pd(y[4 * bj + 1]);
                accz = _mm256_fmadd_pd(a1, x1, accz);

                // Colonne j = 2
                __m256d a2 = _mm256_set_pd(blk[14], blk[10], blk[6], blk[2]);
                __m256d x2 = _mm256_set1_pd(y[4 * bj + 2]);
                accz = _mm256_fmadd_pd(a2, x2, accz);

                // Colonne j = 3
                __m256d a3 = _mm256_set_pd(blk[15], blk[11], blk[7], blk[3]);
                __m256d x3 = _mm256_set1_pd(y[4 * bj + 3]);
                accz = _mm256_fmadd_pd(a3, x3, accz);

                
            }

            

#endif


            
        }
        _mm256_storeu_pd(&z[4 * bi], accz);
    }
}


int main(int argc, char **argv) {
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

    std::vector<int> ptrowend;
    Generate1stlayer(ptrowend, A);

    std::vector<int> ptrowendB;
    Generate1stlayer_BCSR4(ptrowendB, A_bcsr);

	/* ------------------------------------------------------------------ */
	/*                         SpMV benchmarking                          */
	/* ------------------------------------------------------------------ */

    std::vector<double> x(nrow, 1.0);     
	std::vector<double> y(nrow, 0.0);
    std::vector<double> z(nrow, 0.0);


    // SpM2V CSR
    auto start = std::chrono::high_resolution_clock::now();
	SpM2V_CSR(z.data(), y.data(), x.data(), A, ptrowend);
	auto end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_csr = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    base = z;

    printf("SpM2V CSR : %8ld μs | ref | ref\n", duration_spm2v_csr);

    // SpMV CSR OPT
    reset_vectors(x,y,z);

    flush_cache();

    start = std::chrono::high_resolution_clock::now();
	SpM2V_CSR_OPT(z.data(), y.data(), x.data(), A, ptrowend);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_csr_opt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V CSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_csr_opt, (double)duration_spm2v_csr/duration_spm2v_csr_opt,rel_error(base,z));

    // SpMV CSR FMA
    reset_vectors(x,y,z);

    flush_cache();
    
    start = std::chrono::high_resolution_clock::now();
	SpM2V_CSR_OPT(z.data(), y.data(), x.data(), A, ptrowend);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_csr_fma = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V CSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_csr_fma, (double)duration_spm2v_csr/duration_spm2v_csr_fma,rel_error(base,z));

    // SpMV CSR AVX2
    reset_vectors(x,y,z);

    flush_cache();
    
    start = std::chrono::high_resolution_clock::now();
	SpM2V_CSR_AVX2(z.data(), y.data(), x.data(), A, ptrowend);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_csr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V CSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_csr_avx2, (double)duration_spm2v_csr/duration_spm2v_csr_avx2,rel_error(base,z));

    // SpMV BCSR
    reset_vectors(x,y,z);

    flush_cache();

    start = std::chrono::high_resolution_clock::now();
	SpM2V_BCSR(z.data(), y.data(), x.data(), A_bcsr, ptrowendB);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_bcsr = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V BCSR : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_bcsr, (double)duration_spm2v_csr/duration_spm2v_bcsr,rel_error(base,z));

    // SpMV CSR OPT
    reset_vectors(x,y,z);

    flush_cache();

    start = std::chrono::high_resolution_clock::now();
	SpM2V_BCSR_OPT(z.data(), y.data(), x.data(), A_bcsr, ptrowendB);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_bcsr_opt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V BCSR OPT : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_bcsr_opt, (double)duration_spm2v_csr/duration_spm2v_bcsr_opt,rel_error(base,z));

    // SpMV BCSR FMA
    reset_vectors(x,y,z);

    flush_cache();
    
    start = std::chrono::high_resolution_clock::now();
	SpM2V_BCSR_FMA(z.data(), y.data(), x.data(), A_bcsr, ptrowendB);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_bcsr_fma = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V BCSR FMA : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_bcsr_fma, (double)duration_spm2v_csr/duration_spm2v_bcsr_fma,rel_error(base,z));

    // SpMV BCSR AVX2
    reset_vectors(x,y,z);

    flush_cache();
    
    start = std::chrono::high_resolution_clock::now();
	SpM2V_BCSR_AVX2(z.data(), y.data(), x.data(), A_bcsr, ptrowendB);
	end = std::chrono::high_resolution_clock::now();

	long duration_spm2v_bcsr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("SpM2V BCSR AVX2 : %8ld μs | %.2fx | rel err = %.3e\n", duration_spm2v_bcsr_avx2, (double)duration_spm2v_csr/duration_spm2v_bcsr_avx2,rel_error(base,z));

    return 0;
}