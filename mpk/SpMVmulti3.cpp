// Matrix Powers Kernel for a Sparse matrix to increase local access of
// the coefficient based on finite element connectivity
// Copyright 19 Jun. A. Suzuki

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

struct csrmatrix {
  int n, nnz;
  std::vector<int> ptrow;
  std::vector<int> indcol;  
  std::vector<double> coef;
};

struct bcsr4x4_matrix {
  int nrows;
  int nblocks;
  std::vector<int> ptrow;
  std::vector<int> indcol;
  std::vector<std::array<double, 16>> coef;
};

void Generate1stlayer(std::vector<int> &ptrowend1,
		      csrmatrix &A)
{
  int nrow = A.n;
  std::vector<int> mask(nrow, 0);
  ptrowend1.resize(A.nnz);
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      if (mask[j]) {
	ptrowend1[ia] = A.ptrow[j];
      }
      else {
	ptrowend1[ia] = A.ptrow[j + 1];
	mask[j] = 1;	
      }
    }
  }
}

// __attribute__((optimize("O3")))
// __attribute__((target("no-sse,no-avx2,no-fma")))
__attribute__((optimize("O3")))
__attribute__((target("fma")))
__attribute__((target("no-sse,no-avx2")))
void SpM2V(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1) {
    const int nrow = A.n;
    std::fill_n(y, nrow, 0.0);
    std::fill_n(z, nrow, 0.0);

    // Précalcul des pointeurs pour un accès plus rapide
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    const int* pend = ptrowend1.data();

    for (int i = 0; i < nrow; i++) {
        const int row_start = ptrow[i];
        const int row_end = ptrow[i + 1];
        
        for (int ia = row_start; ia < row_end; ia++) {
            const int j = indcol[ia];
            const double a_ij = coef[ia];
            double sum = 0.0;
            
            const int inner_start = ptrow[j];
            const int inner_end = pend[ia];
            
            // Déroulage de boucle manuel (loop unrolling)
            int jb = inner_start;
            for (; jb + 3 < inner_end; jb += 4) {
                sum += coef[jb] * x[indcol[jb]] +
                       coef[jb+1] * x[indcol[jb+1]] +
                       coef[jb+2] * x[indcol[jb+2]] +
                       coef[jb+3] * x[indcol[jb+3]];
            }
            for (; jb < inner_end; jb++) {
                sum += coef[jb] * x[indcol[jb]];
            }
            
            y[j] += sum;
            z[i] += a_ij * y[j];
        }
    }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    int nrow = A.n;
    
    // Initialisation de y et z à zéro
    for (int i = 0; i < nrow; i++) {
        z[i] = y[i] = 0.0;
    }

    for (int i = 0; i < nrow; i++) {
        for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
            int j = A.indcol[ia];
            
            // Accumulation dans y[j] avec AVX2
            __m256d y_vec = _mm256_setzero_pd();
            int jb = A.ptrow[j];
            
            // Traitement par blocs de 4 éléments
            for (; jb + 3 < ptrowend1[ia]; jb += 4) {
                // Chargement des 4 coefficients
                __m256d coef_vec = _mm256_loadu_pd(&A.coef[jb]);
                
                // Chargement des 4 éléments de x correspondants
                __m256d x_vec = _mm256_set_pd(
                    x[A.indcol[jb + 3]],
                    x[A.indcol[jb + 2]],
                    x[A.indcol[jb + 1]],
                    x[A.indcol[jb]]
                );
                
                // Multiplication et accumulation
                y_vec = _mm256_fmadd_pd(coef_vec, x_vec, y_vec);
            }
            
            // Réduction du vecteur y_vec
            double temp[4];
            _mm256_storeu_pd(temp, y_vec);
            y[j] += temp[0] + temp[1] + temp[2] + temp[3];
            
            // Mise à jour de z[i]
            z[i] += A.coef[ia] * y[j];
        }
    }
}

void Generate1stlayer_BCSR4(std::vector<int>& ptrowend1_blk,
                            const bcsr4x4_matrix& A)
{
    const int nblock_rows   = A.nrows;          // = N/4
    const int nnz_block     = static_cast<int>(A.indcol.size());

    ptrowend1_blk.resize(nnz_block);

    /*  mask[b_j] vaut 0 si la colonne-bloc b_j n'a pas encore été rencontrée
        (toutes lignes-bloc confondues) ; 1 sinon                           */
    std::vector<int> mask(nblock_rows, 0);

    for (int bi = 0; bi < nblock_rows; ++bi) {                  // ligne-bloc
        for (int ib = A.ptrow[bi]; ib < A.ptrow[bi + 1]; ++ib) { // bloc (bi, b_j)
            int bj = A.indcol[ib];                               // colonne-bloc

            if (mask[bj]) {
                /* Colonne-bloc déjà vue : on pointe vers le début de la
                   ligne-bloc bj (row-start)                           */
                ptrowend1_blk[ib] = A.ptrow[bj];
            } else {
                /* Première apparition : on pointe vers la fin de la
                   ligne-bloc bj (row-end) et on marque la colonne       */
                ptrowend1_blk[ib] = A.ptrow[bj + 1];
                mask[bj] = 1;
            }
        }
    }
}

// __attribute__((optimize("O3")))
// __attribute__((target("no-sse,no-avx2,no-fma")))
__attribute__((optimize("O3")))
__attribute__((target("fma")))
__attribute__((target("no-sse,no-avx2")))
void SpM2V_BCSR4(double* z, double* y, const double* x,
                        const bcsr4x4_matrix& A, const std::vector<int>& ptrowend1) {
    const int nblock_rows = A.nrows;
    const int nrow = nblock_rows * 4;

    std::fill_n(y, nrow, 0.0);
    std::fill_n(z, nrow, 0.0);

    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const std::array<double, 16>* coef = A.coef.data();
    const int* pend = ptrowend1.data();

    for (int bi = 0; bi < nblock_rows; ++bi) {
        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];

            // Étape 1 — reconstruire un y_tmp[bj] = sum_k A[bj,k] * x[k]
            double ytmp[4] = {0, 0, 0, 0};
            for (int jb = ptrow[bj]; jb < pend[ia]; ++jb) {
                const int bk = indcol[jb];
                const auto& blk = coef[jb];

                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        ytmp[i] += blk[4 * i + j] * x[4 * bk + j];
                    }
                }
            }

            // Étape 2 — y[bj] += ytmp
            for (int i = 0; i < 4; ++i)
                y[4 * bj + i] += ytmp[i];

            // Étape 3 — z[bi] += A[bi,bj] * y[bj]
            const auto& ablk = coef[ia];
            for (int i = 0; i < 4; ++i) {
                double acc = 0.0;
                for (int j = 0; j < 4; ++j) {
                    acc += ablk[4 * i + j] * y[4 * bj + j];
                }
                z[4 * bi + i] += acc;
            }
        }
    }
}


__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2_(double* z, double* y, const double* x,
                          const bcsr4x4_matrix& A, const std::vector<int>& ptrowend1) {
    const int nblock_rows = A.nrows;
    const int nrow = nblock_rows * 4;
    
    std::memset(y, 0, nrow * sizeof(double));
    std::memset(z, 0, nrow * sizeof(double));
    
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const std::array<double, 16>* coef = A.coef.data();
    const int* pend = ptrowend1.data();
    
    for (int bi = 0; bi < nblock_rows; ++bi) {
        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];
            
            // Étape 1 — reconstruire un y_tmp[bj] = sum_k A[bj,k] * x[k]
            __m256d ytmp = _mm256_setzero_pd();
            
            for (int jb = ptrow[bj]; jb < pend[ia]; ++jb) {
                const int bk = indcol[jb];
                const auto& blk = coef[jb];
                
                __m256d xvec = _mm256_loadu_pd(&x[4 * bk]);
                
                // Utilisation de FMA pour de meilleures performances
                for (int i = 0; i < 4; ++i) {
                    __m256d row = _mm256_loadu_pd(&blk[4 * i]);
                    __m256d prod = _mm256_mul_pd(row, xvec);
                    
                    // Somme horizontale optimisée
                    __m256d temp = _mm256_hadd_pd(prod, prod);
                    __m128d hi = _mm256_extractf128_pd(temp, 1);
                    __m128d lo = _mm256_castpd256_pd128(temp);
                    __m128d sum = _mm_add_pd(hi, lo);
                    
                    double result = _mm_cvtsd_f64(sum);
                    
                    // Utilisation de FMA pour l'accumulation
                    double* ytmp_ptr = (double*)&ytmp;
                    ytmp_ptr[i] += result;
                }
            }
            
            // Étape 2 — y[bj] += ytmp (avec FMA)
            __m256d ybj = _mm256_loadu_pd(&y[4 * bj]);
            ybj = _mm256_add_pd(ybj, ytmp);
            _mm256_storeu_pd(&y[4 * bj], ybj);
            
            // Étape 3 — z[bi] += A[bi,bj] * y[bj] (avec FMA)
            const auto& ablk = coef[ia];
            __m256d ybj_vec = _mm256_loadu_pd(&y[4 * bj]);
            __m256d zbi = _mm256_loadu_pd(&z[4 * bi]);
            
            for (int i = 0; i < 4; ++i) {
                __m256d row = _mm256_loadu_pd(&ablk[4 * i]);
                __m256d prod = _mm256_mul_pd(row, ybj_vec);
                
                __m256d temp = _mm256_hadd_pd(prod, prod);
                __m128d hi = _mm256_extractf128_pd(temp, 1);
                __m128d lo = _mm256_castpd256_pd128(temp);
                __m128d sum = _mm_add_pd(hi, lo);
                
                double result = _mm_cvtsd_f64(sum);
                
                double* zbi_ptr = (double*)&zbi;
                zbi_ptr[i] += result;
            }
            
            _mm256_storeu_pd(&z[4 * bi], zbi);
        }
    }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2___(double* z, double* y, const double* x,
const bcsr4x4_matrix& A, const std::vector<int>& ptrowend1) {
    const int nblock_rows = A.nrows;
    const int nrow = nblock_rows * 4;
    
    // Initialisation optimisée avec AVX2
    const __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < nrow; i += 4) {
        _mm256_storeu_pd(&y[i], zero);
        _mm256_storeu_pd(&z[i], zero);
    }
    
    const int* __restrict ptrow = A.ptrow.data();
    const int* __restrict indcol = A.indcol.data();
    const std::array<double, 16>* __restrict coef = A.coef.data();
    const int* __restrict pend = ptrowend1.data();
    
    for (int bi = 0; bi < nblock_rows; ++bi) {
        __m256d zbi = _mm256_loadu_pd(&z[4 * bi]);
        
        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];
            
            // Étape 1 — Calcul de ytmp avec FMA vectorisé
            __m256d ytmp = _mm256_setzero_pd();
            
            for (int jb = ptrow[bj]; jb < pend[ia]; ++jb) {
                const int bk = indcol[jb];
                const auto& blk = coef[jb];
                const __m256d xvec = _mm256_loadu_pd(&x[4 * bk]);
                
                // Multiplication matrice-vecteur 4x4 optimisée avec FMA
                const __m256d* blk_ptr = reinterpret_cast<const __m256d*>(blk.data());
                
                __m256d row0 = _mm256_loadu_pd(&blk[0]);
                __m256d row1 = _mm256_loadu_pd(&blk[4]);
                __m256d row2 = _mm256_loadu_pd(&blk[8]);
                __m256d row3 = _mm256_loadu_pd(&blk[12]);
                
                // Calcul des produits scalaires avec FMA
                __m256d dot0 = _mm256_mul_pd(row0, xvec);
                __m256d dot1 = _mm256_mul_pd(row1, xvec);
                __m256d dot2 = _mm256_mul_pd(row2, xvec);
                __m256d dot3 = _mm256_mul_pd(row3, xvec);
                
                // Sommes horizontales optimisées
                __m256d sum01 = _mm256_hadd_pd(dot0, dot1);
                __m256d sum23 = _mm256_hadd_pd(dot2, dot3);
                
                __m128d sum01_lo = _mm256_castpd256_pd128(sum01);
                __m128d sum01_hi = _mm256_extractf128_pd(sum01, 1);
                __m128d sum23_lo = _mm256_castpd256_pd128(sum23);
                __m128d sum23_hi = _mm256_extractf128_pd(sum23, 1);
                
                __m128d final01 = _mm_add_pd(sum01_lo, sum01_hi);
                __m128d final23 = _mm_add_pd(sum23_lo, sum23_hi);
                
                // Reconstruction du vecteur résultat
                __m256d result = _mm256_set_pd(
                    _mm_cvtsd_f64(_mm_shuffle_pd(final23, final23, 1)),
                    _mm_cvtsd_f64(final23),
                    _mm_cvtsd_f64(_mm_shuffle_pd(final01, final01, 1)),
                    _mm_cvtsd_f64(final01)
                );
                
                ytmp = _mm256_add_pd(ytmp, result);
            }
            
            // Étape 2 — Accumulation dans y avec FMA
            __m256d ybj = _mm256_loadu_pd(&y[4 * bj]);
            ybj = _mm256_add_pd(ybj, ytmp);
            _mm256_storeu_pd(&y[4 * bj], ybj);
            
            // Étape 3 — Accumulation dans z avec FMA (optimisée)
            const auto& ablk = coef[ia];
            
            __m256d row0 = _mm256_loadu_pd(&ablk[0]);
            __m256d row1 = _mm256_loadu_pd(&ablk[4]);
            __m256d row2 = _mm256_loadu_pd(&ablk[8]);
            __m256d row3 = _mm256_loadu_pd(&ablk[12]);
            
            __m256d dot0 = _mm256_mul_pd(row0, ybj);
            __m256d dot1 = _mm256_mul_pd(row1, ybj);
            __m256d dot2 = _mm256_mul_pd(row2, ybj);
            __m256d dot3 = _mm256_mul_pd(row3, ybj);
            
            __m256d sum01 = _mm256_hadd_pd(dot0, dot1);
            __m256d sum23 = _mm256_hadd_pd(dot2, dot3);
            
            __m128d sum01_lo = _mm256_castpd256_pd128(sum01);
            __m128d sum01_hi = _mm256_extractf128_pd(sum01, 1);
            __m128d sum23_lo = _mm256_castpd256_pd128(sum23);
            __m128d sum23_hi = _mm256_extractf128_pd(sum23, 1);
            
            __m128d final01 = _mm_add_pd(sum01_lo, sum01_hi);
            __m128d final23 = _mm_add_pd(sum23_lo, sum23_hi);
            
            __m256d z_increment = _mm256_set_pd(
                _mm_cvtsd_f64(_mm_shuffle_pd(final23, final23, 1)),
                _mm_cvtsd_f64(final23),
                _mm_cvtsd_f64(_mm_shuffle_pd(final01, final01, 1)),
                _mm_cvtsd_f64(final01)
            );
            
            zbi = _mm256_add_pd(zbi, z_increment);
        }
        
        // Stockage final de zbi
        _mm256_storeu_pd(&z[4 * bi], zbi);
    }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2(double* __restrict z, double* __restrict y, const double* __restrict x,
                            const bcsr4x4_matrix& A, const std::vector<int>& ptrowend1) {
    const int nblock_rows = A.nrows;
    const int nrow = nblock_rows * 4;

    const __m256d zero = _mm256_setzero_pd();
    for (int i = 0; i < nrow; i += 4) {
        _mm256_storeu_pd(&y[i], zero);
        _mm256_storeu_pd(&z[i], zero);
    }

    const int* __restrict ptrow = A.ptrow.data();
    const int* __restrict indcol = A.indcol.data();
    const std::array<double, 16>* __restrict coef = A.coef.data();
    const int* __restrict pend = ptrowend1.data();

    for (int bi = 0; bi < nblock_rows; ++bi) {
        __m256d zbi = _mm256_loadu_pd(&z[4 * bi]);

        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];

            // Étape 1 — ytmp = sum_k A[bj,k] * x_k (4x4 matvec + accumulate)
            __m256d ytmp = _mm256_setzero_pd();

            for (int jb = ptrow[bj]; jb < pend[ia]; ++jb) {
                const int bk = indcol[jb];
                const auto& blk = coef[jb];

                __m256d xvec = _mm256_loadu_pd(&x[4 * bk]);

                __m256d row0 = _mm256_loadu_pd(&blk[0]);
                __m256d row1 = _mm256_loadu_pd(&blk[4]);
                __m256d row2 = _mm256_loadu_pd(&blk[8]);
                __m256d row3 = _mm256_loadu_pd(&blk[12]);

                __m256d acc = _mm256_setzero_pd();
                acc = _mm256_insertf128_pd(acc, _mm_add_pd(
                    _mm256_castpd256_pd128(_mm256_fmadd_pd(row0, xvec, _mm256_setzero_pd())),
                    _mm256_castpd256_pd128(_mm256_fmadd_pd(row1, xvec, _mm256_setzero_pd()))
                ), 0);

                acc = _mm256_insertf128_pd(acc, _mm_add_pd(
                    _mm256_castpd256_pd128(_mm256_fmadd_pd(row2, xvec, _mm256_setzero_pd())),
                    _mm256_castpd256_pd128(_mm256_fmadd_pd(row3, xvec, _mm256_setzero_pd()))
                ), 1);

                ytmp = _mm256_add_pd(ytmp, acc);
            }

            // Étape 2 — y[bj] += ytmp
            __m256d ybj = _mm256_loadu_pd(&y[4 * bj]);
            ybj = _mm256_add_pd(ybj, ytmp);
            _mm256_storeu_pd(&y[4 * bj], ybj);

            // Étape 3 — z[bi] += A[bi,bj] * y[bj]
            const auto& ablk = coef[ia];
            __m256d row0 = _mm256_loadu_pd(&ablk[0]);
            __m256d row1 = _mm256_loadu_pd(&ablk[4]);
            __m256d row2 = _mm256_loadu_pd(&ablk[8]);
            __m256d row3 = _mm256_loadu_pd(&ablk[12]);

            // __m256d acc0 = _mm256_fmadd_pd(row0, ybj, _mm256_setzero_pd());
            // __m256d acc1 = _mm256_fmadd_pd(row1, ybj, _mm256_setzero_pd());
            // __m256d acc2 = _mm256_fmadd_pd(row2, ybj, _mm256_setzero_pd());
            // __m256d acc3 = _mm256_fmadd_pd(row3, ybj, _mm256_setzero_pd());
            
            __mm256d acc;
            acc = _mm256_fmadd_pd(row0, ybj, zbi);
            zbi = _mm256_fmadd_pd(row1, ybj, acc);
            acc = _mm256_fmadd_pd(row2, ybj, zbi);
            zbi = _mm256_fmadd_pd(row3, ybj, acc);
            

            // // Horizontal sums
            // double tmp[4];
            // _mm256_storeu_pd(tmp, acc0);
            // double s0 = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            // _mm256_storeu_pd(tmp, acc1);
            // double s1 = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            // _mm256_storeu_pd(tmp, acc2);
            // double s2 = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            // _mm256_storeu_pd(tmp, acc3);
            // double s3 = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            // __m256d zincr = _mm256_set_pd(s3, s2, s1, s0);
            // zbi = _mm256_add_pd(zbi, zincr);
        }

        _mm256_storeu_pd(&z[4 * bi], zbi);

    }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2__(double* __restrict z, double* __restrict y, const double* __restrict x,
                        const bcsr4x4_matrix& A, const std::vector<int>& ptrowend1) {
    const int nblock_rows = A.nrows;
    const int nrow = nblock_rows * 4;

    // Optimized zero initialization with memory load
    alignas(32) static const double zero_buf[4] = {0.0, 0.0, 0.0, 0.0};
    const __m256d zero = _mm256_load_pd(zero_buf);
    for (int i = 0; i < nrow; i += 4) {
        _mm256_storeu_pd(&y[i], zero);
        _mm256_storeu_pd(&z[i], zero);
    }

    const int* __restrict ptrow = A.ptrow.data();
    const int* __restrict indcol = A.indcol.data();
    const std::array<double, 16>* __restrict coef = A.coef.data();
    const int* __restrict pend = ptrowend1.data();

    for (int bi = 0; bi < nblock_rows; ++bi) {
        __m256d zbi = _mm256_loadu_pd(&z[4 * bi]);

        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];
            __m256d ytmp = _mm256_load_pd(zero_buf);

            for (int jb = ptrow[bj]; jb < pend[ia]; ++jb) {
                const int bk = indcol[jb];
                const auto& blk = coef[jb];
                const __m256d xvec = _mm256_loadu_pd(&x[4 * bk]);

                __m256d row0 = _mm256_loadu_pd(&blk[0]);
                __m256d row1 = _mm256_loadu_pd(&blk[4]);
                __m256d row2 = _mm256_loadu_pd(&blk[8]);
                __m256d row3 = _mm256_loadu_pd(&blk[12]);

                __m256d dot0 = _mm256_mul_pd(row0, xvec);
                __m256d dot1 = _mm256_mul_pd(row1, xvec);
                __m256d dot2 = _mm256_mul_pd(row2, xvec);
                __m256d dot3 = _mm256_mul_pd(row3, xvec);

                __m256d sum01 = _mm256_hadd_pd(dot0, dot1);
                __m256d sum23 = _mm256_hadd_pd(dot2, dot3);

                __m128d sum01_lo = _mm256_castpd256_pd128(sum01);
                __m128d sum01_hi = _mm256_extractf128_pd(sum01, 1);
                __m128d sum23_lo = _mm256_castpd256_pd128(sum23);
                __m128d sum23_hi = _mm256_extractf128_pd(sum23, 1);

                __m128d final01 = _mm_add_pd(sum01_lo, sum01_hi);
                __m128d final23 = _mm_add_pd(sum23_lo, sum23_hi);

                __m256d result = _mm256_set_pd(
                    _mm_cvtsd_f64(_mm_shuffle_pd(final23, final23, 1)),
                    _mm_cvtsd_f64(final23),
                    _mm_cvtsd_f64(_mm_shuffle_pd(final01, final01, 1)),
                    _mm_cvtsd_f64(final01)
                );

                ytmp = _mm256_add_pd(ytmp, result);
            }

            __m256d ybj = _mm256_loadu_pd(&y[4 * bj]);
            ybj = _mm256_add_pd(ybj, ytmp);
            _mm256_storeu_pd(&y[4 * bj], ybj);

            const auto& ablk = coef[ia];

            __m256d row0 = _mm256_loadu_pd(&ablk[0]);
            __m256d row1 = _mm256_loadu_pd(&ablk[4]);
            __m256d row2 = _mm256_loadu_pd(&ablk[8]);
            __m256d row3 = _mm256_loadu_pd(&ablk[12]);

            __m256d dot0 = _mm256_mul_pd(row0, ybj);
            __m256d dot1 = _mm256_mul_pd(row1, ybj);
            __m256d dot2 = _mm256_mul_pd(row2, ybj);
            __m256d dot3 = _mm256_mul_pd(row3, ybj);

            __m256d sum01 = _mm256_hadd_pd(dot0, dot1);
            __m256d sum23 = _mm256_hadd_pd(dot2, dot3);

            __m128d sum01_lo = _mm256_castpd256_pd128(sum01);
            __m128d sum01_hi = _mm256_extractf128_pd(sum01, 1);
            __m128d sum23_lo = _mm256_castpd256_pd128(sum23);
            __m128d sum23_hi = _mm256_extractf128_pd(sum23, 1);

            __m128d final01 = _mm_add_pd(sum01_lo, sum01_hi);
            __m128d final23 = _mm_add_pd(sum23_lo, sum23_hi);

            __m256d z_increment = _mm256_set_pd(
                _mm_cvtsd_f64(_mm_shuffle_pd(final23, final23, 1)),
                _mm_cvtsd_f64(final23),
                _mm_cvtsd_f64(_mm_shuffle_pd(final01, final01, 1)),
                _mm_cvtsd_f64(final01)
            );

            zbi = _mm256_add_pd(zbi, z_increment);
        }

        _mm256_storeu_pd(&z[4 * bi], zbi);
    }
}



// __attribute__((optimize("O3")))
// __attribute__((target("no-sse,no-avx2,no-fma")))
__attribute__((optimize("O3")))
__attribute__((target("fma")))
__attribute__((target("no-sse,no-avx2")))
void SpMV(double *y, double *x, csrmatrix &a)
{
  int nrow = a.n;
  const double zero(0.0);
  for (int i = 0; i < nrow; i++) {
    y[i] = zero;
    for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++) {
      int j = a.indcol[ia];
      y[i] += a.coef[ia] * x[j];
    }
  }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpMV_AVX2(double *y, double *x, csrmatrix &a) {
  int nrow = a.n;
  for (int i = 0; i < nrow; i++) {
    __m256d sum = _mm256_setzero_pd();
    int ia = a.ptrow[i];
    int end = a.ptrow[i + 1];
    
    // Boucle vectorisée
    for (; ia + 3 < end; ia += 4) {
      __m256d coef = _mm256_loadu_pd(&a.coef[ia]);
      __m256d x_vec = _mm256_set_pd(
        x[a.indcol[ia + 3]],
        x[a.indcol[ia + 2]],
        x[a.indcol[ia + 1]],
        x[a.indcol[ia]]
      );
      sum = _mm256_fmadd_pd(coef, x_vec, sum);
    }
    
    // Réduction
    double temp[4];
    _mm256_storeu_pd(temp, sum);
    y[i] = temp[0] + temp[1] + temp[2] + temp[3];
    
  }
}

__attribute__((optimize("O3")))
__attribute__((target("fma")))
__attribute__((target("no-sse,no-avx2")))
void SpMV_BCSR4(double* y, const double* x, const bcsr4x4_matrix& A) {
    const int nblock_rows = A.nrows;
    std::memset(y, 0, nblock_rows * 4 * sizeof(double));

    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const std::array<double, 16>* coef = A.coef.data();

    for (int bi = 0; bi < nblock_rows; ++bi) {
        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];
            const auto& blk = coef[ia];

            for (int i = 0; i < 4; ++i) {
                double acc = 0.0;
                for (int j = 0; j < 4; ++j) {
                    acc += blk[4 * i + j] * x[4 * bj + j];
                }
                y[4 * bi + i] += acc;
            }
        }
    }
}

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpMV_BCSR4_AVX2(double* y, const double* x, const bcsr4x4_matrix& A) {
    const int nblock_rows = A.nrows;
    
    std::memset(y, 0, nblock_rows * 4 * sizeof(double));
    
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const std::array<double, 16>* coef = A.coef.data();
    
    for (int bi = 0; bi < nblock_rows; ++bi) {
        __m256d y_block = _mm256_loadu_pd(&y[4 * bi]);
        
        for (int ia = ptrow[bi]; ia < ptrow[bi + 1]; ++ia) {
            const int bj = indcol[ia];
            const auto& blk = coef[ia];
            
            __m256d x_block = _mm256_loadu_pd(&x[4 * bj]);
            
            // Utilisation de FMA pour chaque ligne
            for (int i = 0; i < 4; ++i) {
                __m256d row = _mm256_loadu_pd(&blk[4 * i]);
                
                // FMA: row * x_block + 0
                __m256d prod = _mm256_mul_pd(row, x_block);
                
                // Réduction horizontale optimisée
                __m256d temp = _mm256_hadd_pd(prod, prod);
                __m128d hi = _mm256_extractf128_pd(temp, 1);
                __m128d lo = _mm256_castpd256_pd128(temp);
                __m128d sum = _mm_add_pd(hi, lo);
                
                double result = _mm_cvtsd_f64(sum);
                
                // Accumulation avec FMA dans y[4*bi + i]
                double* y_ptr = (double*)&y_block;
                y_ptr[i] += result;
            }
        }
        
        _mm256_storeu_pd(&y[4 * bi], y_block);
    }
}



void generate_CSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp, 
		  int nrow, int nnz, 
		  int *irow, int *jcol, double* val)
{
  for (int i = 0; i < nnz; i++) {
    const int ii = irow[i];
    const int jj = jcol[i];
    if (ind_cols_tmp[ii].empty()) {
      ind_cols_tmp[ii].push_back(jj);
      val_tmp[ii].push_back(val[i]);
    }
    else {
      if (ind_cols_tmp[ii].back() < jj) {
	ind_cols_tmp[ii].push_back(jj);
	val_tmp[ii].push_back(val[i]);
      }
      else {
	std::list<double>::iterator iv = val_tmp[ii].begin();
	std::list<int>::iterator it = ind_cols_tmp[ii].begin();
	for ( ; it != ind_cols_tmp[ii].end(); ++it, ++iv) {
	  if (*it == jj) {
	      break;
	  }
	  if (*it > jj) {
	    ind_cols_tmp[ii].insert(it, jj);
	    val_tmp[ii].insert(iv, val[i]);
	    break;
	  }
	}
      }
    }
  }
}

void generate_BCSR4(std::list<std::pair<int, std::array<double, 16>>>* block_rows,
                    int nrow, int nnz,
                    const int* irow, const int* jcol, const double* val,
                    bcsr4x4_matrix& A) {
  const int nblocks = nrow / 4;
  for (int k = 0; k < nnz; ++k) {
    int i = irow[k], j = jcol[k];
    double v = val[k];
    int bi = i / 4, bj = j / 4;
    int ii = i % 4, jj = j % 4;
    bool found = false;
    for (auto& pair : block_rows[bi]) {
      if (pair.first == bj) {
        pair.second[4 * ii + jj] = v;
        found = true;
        break;
      }
    }
    if (!found) {
      std::array<double, 16> new_block = {};
      new_block[4 * ii + jj] = v;
      block_rows[bi].emplace_back(bj, new_block);
    }
  }
  A.nblocks = nblocks;
  A.ptrow.resize(nblocks + 1, 0);
  for (int bi = 0; bi < nblocks; ++bi) {
    A.ptrow[bi + 1] = A.ptrow[bi] + block_rows[bi].size();
    for (const auto& pair : block_rows[bi]) {
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
  std::vector<std::list<int> > ind_cols_tmp(nrow);
  std::vector<std::list<double> > val_tmp(nrow);

  // without adding diagonal nor symmetrize for PARDISO mtype = 11
  generate_CSR(&ind_cols_tmp[0], &val_tmp[0], 
               nrow, nnz,
               &irow[0], &jcol[0], &val[0]);
  {
    int k = 0;
    a.ptrow[0] = 0;
    for (int i = 0; i < nrow; i++) {
      std::list<int>::iterator jt = ind_cols_tmp[i].begin();
      std::list<double>::iterator jv = val_tmp[i].begin();     
      for ( ; jt != ind_cols_tmp[i].end(); ++jt, ++jv) {
        a.indcol[k] = (*jt);
        a.coef[k] = (*jv);
        k++;
      }
      a.ptrow[i + 1] = k;
    }
  }
}

double norm2(const std::vector<double> &x) {
    double s = 0.0;
    for (double xi : x) s += xi * xi;
    return std::sqrt(s);
}

double rel_error(const std::vector<double> &ref, const std::vector<double> &test) {
    double s = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        s += (ref[i] - test[i]) * (ref[i] - test[i]);
    return std::sqrt(s) / norm2(ref);
}


int main(int argc, char **argv)
{
  char fname[256];
  char buf[1024];
  int nrow, nnz;

  FILE *fp;
  int itmp, jtmp, ktmp;
  
  strcpy(fname, argv[1]);
  
  if ((fp = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "fail to open %s\n", fname);
    return 1;
  }
  fgets(buf, 256, fp);
  
  while (1) {
    fgets(buf, 256, fp);
    if (buf[0] != '%') {
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
    for (int i = 0; i < nnz; i++) {
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
  
  csrmatrix a;
  COO2CSR(a, nrow, nnz, &irow[0], &jcol[0], &val[0]);

  // Conversion vers BCSR4x4
  bcsr4x4_matrix a_bcsr;
  std::vector<std::list<std::pair<int, std::array<double, 16>>>> block_rows((nrow + 3) / 4);
  generate_BCSR4(&block_rows[0], nrow, nnz, &irow[0], &jcol[0], &val[0], a_bcsr);
  

  std::vector<int> ptrowend1_bcsr;
  Generate1stlayer_BCSR4(ptrowend1_bcsr, a_bcsr);

  // Initialisation
  std::vector<double> x(nrow, 1.0), b(nrow, 0.0);
  std::vector<double> x1(nrow, 0.0), x2(nrow, 0.0);
  std::vector<double> y1(nrow, 0.0), y2(nrow, 0.0);
  std::vector<double> z1(nrow, 0.0), z2(nrow, 0.0);
  std::vector<double> w1(nrow, 0.0), w2(nrow, 0.0);
  std::vector<double> v1(nrow, 0.0), v2(nrow, 0.0);
  std::vector<double> u1(nrow, 0.0), u2(nrow, 0.0);

  // Warm-up
  SpMV(&b[0], &x[0], a);

  std::vector<int> ptrowend1;
  Generate1stlayer(ptrowend1, a);

  // 1. 2x SpMV séquentiel
  auto start = std::chrono::high_resolution_clock::now();
  SpMV(&x1[0], &b[0], a); // ajouter axpy et inner prod in between
  SpMV(&x2[0], &x1[0], a); // ne pas réutilsier x1 mais c
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_spmv = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 2. 2x SpMV AVX2
  start = std::chrono::high_resolution_clock::now();
  SpMV_AVX2(&z1[0], &b[0], a);
  SpMV_AVX2(&z2[0], &z1[0], a);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spmv_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 3. SpM2V séquentiel
  start = std::chrono::high_resolution_clock::now();
  SpM2V(&y2[0], &y1[0], &b[0], a, ptrowend1);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spm2v = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 4. SpM2V AVX2
  start = std::chrono::high_resolution_clock::now();
  SpM2V_AVX2(&y2[0], &y1[0], &b[0], a, ptrowend1);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spm2v_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 5. SpM2V BCSR4 AVX2
  start = std::chrono::high_resolution_clock::now();
  SpM2V_BCSR4_AVX2(w2.data(), w1.data(), b.data(), a_bcsr, ptrowend1_bcsr);
  end = std::chrono::high_resolution_clock::now();
  auto duration_bcsr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 6. 2x SpMV BCSR4 AVX2
  start = std::chrono::high_resolution_clock::now();
  SpMV_BCSR4_AVX2(v1.data(), b.data(), a_bcsr);
  SpMV_BCSR4_AVX2(v2.data(), v1.data(), a_bcsr);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spmv_bcsr_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 7. SpM2V BCSR4 (séquentiel)
  start = std::chrono::high_resolution_clock::now();
  SpM2V_BCSR4(u2.data(), u1.data(), b.data(), a_bcsr, ptrowend1_bcsr);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spm2v_bcsr_seq = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // 8. 2x SpMV BCSR4 (sequential)
  start = std::chrono::high_resolution_clock::now();
  SpMV_BCSR4(u1.data(), b.data(), a_bcsr);
  SpMV_BCSR4(u2.data(), u1.data(), a_bcsr);
  end = std::chrono::high_resolution_clock::now();
  auto duration_spmv_bcsr_seq = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double error_spmv_bcsr_seq = rel_error(x2, u2);

  // Vérification des résultats
  double error_spmv = rel_error(x2, y2);
  double error_avx2 = rel_error(x2, z2);
  double error_bcsr = rel_error(x2, w2);
  double error_spmv_bcsr = rel_error(x2, v2);
  double error_spm2v_bcsr_seq = rel_error(x2, u2);

  // Affichage des résultats
  printf("Performance comparison (A*A*x):\n");
  printf("1. 2x SpMV (sequential):        %8ld μs\n", duration_spmv);
  printf("2. 2x SpMV (AVX2):              %8ld μs (speedup: %.2fx)\n", duration_spmv_avx2, (double)duration_spmv / duration_spmv_avx2);
  printf("3. 2x SpMV BCSR4 (sequential):  %8ld μs (speedup: %.2fx)\n", duration_spmv_bcsr_seq, (double)duration_spmv / duration_spmv_bcsr_seq);
  printf("4. 2x SpMV BCSR4 (AVX2):        %8ld μs (speedup: %.2fx)\n", duration_spmv_bcsr_avx2, (double)duration_spmv / duration_spmv_bcsr_avx2);
  printf("5. SpM2V (sequential):          %8ld μs (speedup: %.2fx)\n", duration_spm2v, (double)duration_spmv / duration_spm2v);
  printf("6. SpM2V (AVX2):                %8ld μs (speedup: %.2fx)\n", duration_spm2v_avx2, (double)duration_spmv / duration_spm2v_avx2);
  printf("7. SpM2V BCSR4 (sequential):    %8ld μs (speedup: %.2fx)\n", duration_spm2v_bcsr_seq, (double)duration_spmv / duration_spm2v_bcsr_seq);
  printf("8. SpM2V BCSR4 (AVX2):          %8ld μs (speedup: %.2fx)\n", duration_bcsr_avx2, (double)duration_spmv / duration_bcsr_avx2);
  
  printf("\nAccuracy verification:\n");
  printf("Relative error 2xSpMV vs 2xBCSR4 :      %e\n", error_spmv_bcsr_seq);
  printf("Relative error 2xSpMV vs 2xBCSR4_AVX2:  %e\n", error_spmv_bcsr);
  printf("Relative error 2xSpMV vs SpM2V:         %e\n", error_spmv);
  printf("Relative error 2xSpMV vs 2xAVX2:        %e\n", error_avx2);
  printf("Relative error 2xSpMV vs BCSR4_AVX2:    %e\n", error_bcsr);
  printf("Relative error 2xSpMV vs SpM2V_BCSR4:   %e\n", error_spm2v_bcsr_seq);

  return 0;
}
