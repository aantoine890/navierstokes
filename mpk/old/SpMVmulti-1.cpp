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

struct csrmatrix {
  int n, nnz;
  std::vector<int> ptrow;
  std::vector<int> indcol;  
  std::vector<double> coef;
};

struct bcsr4x4_matrix {
    int nrows; // nombre de blocs-lignes (nb_blocs = nrows)
    int nblocks;
    std::vector<int> ptrow;   // taille = nrows + 1
    std::vector<int> indcol;  // indices colonnes des blocs
    std::vector<std::array<double, 16>> coef; // blocs 4×4
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
// void SpM2V(double *z, double *y,
// 	   double *x, csrmatrix &A, std::vector<int> &ptrowend1)
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

// Version optimisée avec plusieurs améliorations de performance
__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V(double *z, double *y,
                     double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
    int nrow = A.n;
    
    // Optimisation 1: Initialisation vectorisée avec memset
    std::memset(z, 0, nrow * sizeof(double));
    std::memset(y, 0, nrow * sizeof(double));
    
    // Optimisation 2: Pré-calcul des pointeurs fréquemment utilisés
    const int* __restrict__ ptrow = A.ptrow.data();
    const int* __restrict__ indcol = A.indcol.data();
    const double* __restrict__ coef = A.coef.data();
    const int* __restrict__ ptrowend1_ptr = ptrowend1.data();
    
    // Optimisation 3: Boucle principale avec optimisations
    for (int i = 0; i < nrow; i++) {
        const int row_start = ptrow[i];
        const int row_end = ptrow[i + 1];
        
        // Optimisation 4: Accumulation locale pour z[i]
        double z_acc = 0.0;
        
        // Déroulement partiel de la boucle externe
        for (int ia = row_start; ia < row_end; ia++) {
            const int j = indcol[ia];
            const double coef_ia = coef[ia];
            
            // Optimisation 5: Accumulation locale pour y[j]
            double y_acc = 0.0;
            
            const int inner_start = ptrow[j];
            const int inner_end = ptrowend1_ptr[ia];
            
            // Optimisation 6: Déroulement de boucle interne (unroll by 4)
            int jb = inner_start;
            for (; jb + 3 < inner_end; jb += 4) {
                y_acc += coef[jb] * x[indcol[jb]] +
                         coef[jb+1] * x[indcol[jb+1]] +
                         coef[jb+2] * x[indcol[jb+2]] +
                         coef[jb+3] * x[indcol[jb+3]];
            }
            
            // Traitement des éléments restants
            for (; jb < inner_end; jb++) {
                y_acc += coef[jb] * x[indcol[jb]];
            }
            
            y[j] += y_acc;
            z_acc += coef_ia * y[j];
        }
        
        z[i] = z_acc;
    }
}


// __attribute__((optimize("O3")))
// __attribute__((target("avx2,fma")))
// void SpM2V_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//     int nrow = A.n;
    
//     // Initialisation vectorisée ultra-rapide
//     for (int i = 0; i < nrow; i += 4) {
//         __m256d zero = _mm256_setzero_pd();
//         _mm256_store_pd(&z[i], zero);
//         _mm256_store_pd(&y[i], zero);
//     }
    
//     // Boucle principale avec optimisations maximales
//     for (int i = 0; i < nrow; i++) {
//         for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//             int j = A.indcol[ia];
//             double coef_ia = A.coef[ia];
            
//             __m256d sum_vec1 = _mm256_setzero_pd();
//             __m256d sum_vec2 = _mm256_setzero_pd();
//             int jb = A.ptrow[j];
//             int end_jb = ptrowend1[ia];
            
//             // Déroulage de boucle 2x pour réduire les dépendances
//             for (; jb + 7 < end_jb; jb += 8) {
//                 // Premier groupe de 4
//                 int k0 = A.indcol[jb];
//                 int k1 = A.indcol[jb + 1];
//                 int k2 = A.indcol[jb + 2];
//                 int k3 = A.indcol[jb + 3];
                
//                 __m256d coef_vec1 = _mm256_load_pd(&A.coef[jb]);
//                 __m256d x_vec1 = _mm256_set_pd(x[k3], x[k2], x[k1], x[k0]);
//                 sum_vec1 = _mm256_fmadd_pd(coef_vec1, x_vec1, sum_vec1);
                
//                 // Deuxième groupe de 4
//                 int k4 = A.indcol[jb + 4];
//                 int k5 = A.indcol[jb + 5];
//                 int k6 = A.indcol[jb + 6];
//                 int k7 = A.indcol[jb + 7];
                
//                 __m256d coef_vec2 = _mm256_load_pd(&A.coef[jb + 4]);
//                 __m256d x_vec2 = _mm256_set_pd(x[k7], x[k6], x[k5], x[k4]);
//                 sum_vec2 = _mm256_fmadd_pd(coef_vec2, x_vec2, sum_vec2);
//             }
            
//             // Combinaison des deux accumulateurs
//             __m256d sum_vec = _mm256_add_pd(sum_vec1, sum_vec2);
            
//             // Traitement des blocs de 4 restants
//             for (; jb + 3 < end_jb; jb += 4) {
//                 int k0 = A.indcol[jb];
//                 int k1 = A.indcol[jb + 1];
//                 int k2 = A.indcol[jb + 2];
//                 int k3 = A.indcol[jb + 3];
                
//                 __m256d coef_vec = _mm256_load_pd(&A.coef[jb]);
//                 __m256d x_vec = _mm256_set_pd(x[k3], x[k2], x[k1], x[k0]);
//                 sum_vec = _mm256_fmadd_pd(coef_vec, x_vec, sum_vec);
//             }
            
//             // Réduction horizontale optimisée
//             __m256d sum_shuffle = _mm256_hadd_pd(sum_vec, sum_vec);
//             __m128d sum_high = _mm256_extractf128_pd(sum_shuffle, 1);
//             __m128d sum_low = _mm256_castpd256_pd128(sum_shuffle);
//             __m128d sum_final = _mm_add_pd(sum_low, sum_high);
//             double partial_sum = _mm_cvtsd_f64(sum_final);
            
//             // Éléments restants
//             for (; jb < end_jb; jb++) {
//                 int k = A.indcol[jb];
//                 partial_sum += A.coef[jb] * x[k];
//             }
            
//             y[j] += partial_sum;
//             z[i] += coef_ia * y[j];
//         }
//     }
// }

// __attribute__((optimize("O3")))
// __attribute__((target("avx2,fma")))
// void SpM2V_AVX2(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1)
// {
//     int nrow = A.n;
//     memset(z, 0, nrow * sizeof(double));
//     memset(y, 0, nrow * sizeof(double));
    
//     for (int i = 0; i < nrow; i++) {
//         double z_acc = 0.0;
//         for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//             int j = A.indcol[ia];
//             double coef_ia = A.coef[ia];
            
//             __m256d y_local = _mm256_setzero_pd();
//             int jb_start = A.ptrow[j];
//             int jb_end = ptrowend1[ia];
            
//             for (int jb = jb_start; jb < jb_end; jb += 4) {
//                 __m256d coef_vec = _mm256_loadu_pd(&A.coef[jb]);
                
//                 // Gather manuel optimisé avec prefetch
//                 _mm_prefetch(&x[A.indcol[jb + 4]], _MM_HINT_T0);
                
//                 // Version optimisée de votre gather manuel
//                 __m256d x_vec = _mm256_set_pd(
//                     x[A.indcol[jb + 3]],
//                     x[A.indcol[jb + 2]], 
//                     x[A.indcol[jb + 1]], 
//                     x[A.indcol[jb]]
//                 );
                
//                 y_local = _mm256_fmadd_pd(coef_vec, x_vec, y_local);
//             }
            
//             // Réduction horizontale optimisée
//             alignas(32) double temp[4];
//             _mm256_store_pd(temp, y_local);
//             double y_val = (temp[0] + temp[1]) + (temp[2] + temp[3]);
            
//             y[j] += y_val;
//             z_acc += coef_ia * y[j];
//         }
//         z[i] = z_acc;
//     }
// }

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2(double *z, double *y, const double *x, const bcsr4x4_matrix &A)
{
    const int nrow = A.nrows * 4;

    // Initialisation
    std::memset(z, 0, nrow * sizeof(double));
    std::memset(y, 0, nrow * sizeof(double));

    const int n_blocks = A.ptrow.size() - 1;

    for (int bi = 0; bi < A.nrows; bi++) {
        const int row_base = 4 * bi;
        __m256d z0 = _mm256_setzero_pd(); // accumulation locale pour z[row_base:row_base+3]

        for (int k = A.ptrow[bi]; k < A.ptrow[bi + 1]; ++k) {
            const int bj = A.indcol[k];
            const int col_base = 4 * bj;
            const std::array<double, 16>& block = A.coef[k];

            // Charger x[col_base + 0:3]
            __m256d x0 = _mm256_loadu_pd(&x[col_base]);

            // Multiplier le bloc 4x4 avec x
            __m256d r0 = _mm256_loadu_pd(&block[0]);
            __m256d r1 = _mm256_loadu_pd(&block[4]);
            __m256d r2 = _mm256_loadu_pd(&block[8]);
            __m256d r3 = _mm256_loadu_pd(&block[12]);

            // fmadd : y[bi] += block * x
            __m256d acc0 = _mm256_setzero_pd();
            acc0 = _mm256_fmadd_pd(r0, x0, acc0);
            double tmp0 = block[0] * x[col_base + 0] + block[1] * x[col_base + 1] + block[2] * x[col_base + 2] + block[3] * x[col_base + 3];
            double tmp1 = block[4] * x[col_base + 0] + block[5] * x[col_base + 1] + block[6] * x[col_base + 2] + block[7] * x[col_base + 3];
            double tmp2 = block[8] * x[col_base + 0] + block[9] * x[col_base + 1] + block[10] * x[col_base + 2] + block[11] * x[col_base + 3];
            double tmp3 = block[12] * x[col_base + 0] + block[13] * x[col_base + 1] + block[14] * x[col_base + 2] + block[15] * x[col_base + 3];

            y[col_base + 0] += tmp0;
            y[col_base + 1] += tmp1;
            y[col_base + 2] += tmp2;
            y[col_base + 3] += tmp3;

            // Accumuler dans z
            z0 = _mm256_fmadd_pd(_mm256_loadu_pd(&block[0]), _mm256_set1_pd(y[col_base + 0]), z0);
            z0 = _mm256_fmadd_pd(_mm256_loadu_pd(&block[4]), _mm256_set1_pd(y[col_base + 1]), z0);
            z0 = _mm256_fmadd_pd(_mm256_loadu_pd(&block[8]), _mm256_set1_pd(y[col_base + 2]), z0);
            z0 = _mm256_fmadd_pd(_mm256_loadu_pd(&block[12]), _mm256_set1_pd(y[col_base + 3]), z0);
        }

        _mm256_storeu_pd(&z[row_base], z0);
    }
}





void Generate2ndlayer(std::vector<std::vector<int> >&ptrowend2,
		      csrmatrix &A,
		      std::vector<int> &ptrowend1)
{
  int nrow = A.n;
  std::vector<int> mask2(nrow, 0);  
  ptrowend2.resize(A.nnz);  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      ptrowend2[ia].resize(ptrowend1[ia] - A.ptrow[j]);
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	if (mask2[k]) {
	  ptrowend2[ia][jjb] = A.ptrow[k];
	}
	else {
	  ptrowend2[ia][jjb] = A.ptrow[k + 1];
	  mask2[k] = 1;
	}
      } // loop : jb
    } // loop : ia
  }  // loop : i
}

void SpM3V(double *w, double *z, double *y,
	   double *x, csrmatrix &A, std::vector<int> &ptrowend1,
	   std::vector<std::vector<int> > &ptrowend2)	   
{
  int nrow = A.n;  
  for (int i = 0; i < nrow; i++) {
    w[i] = z[i] = y[i] = 0.0;
  }  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  y[k] += A.coef[kc] * x[l];
	}
	z[j] += A.coef[jb] * y[k];	
      } // loop : jb
      w[i] += A.coef[ia] * z[j];
    } // loop : ia
  }   // loop : i
}

void Generate3rdlayer(std::vector<std::vector<std::vector<int> > > &ptrowend3,
		      csrmatrix &A,
		      std::vector<int> &ptrowend1,
		      std::vector<std::vector<int> > &ptrowend2)
{
  int nrow = A.n;
  std::vector<int> mask3(nrow, 0);  
  ptrowend3.resize(A.nnz);  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      ptrowend3[ia].resize(ptrowend1[ia] - A.ptrow[j]);
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  ptrowend3[ia][jjb].resize(ptrowend2[ia][jjb] - A.ptrow[k]);
	  int kkc = kc - A.ptrow[k];
	  if (mask3[l]) {
	    ptrowend3[ia][jjb][kkc] = A.ptrow[l];
	  }
	  else {
	    ptrowend3[ia][jjb][kkc] = A.ptrow[l + 1];	  
	    mask3[l] = 1;
	  }
	}
      } // loop : jb
    } // loop : ia
  }  // loop : i
}

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM4V(double *v, double *w, double *z, double *y,
	   double *x, csrmatrix &A, std::vector<int> &ptrowend1,
	   std::vector<std::vector<int> > &ptrowend2,
	   std::vector<std::vector<std::vector<int> > > &ptrowend3)
	   
{
  int nrow = A.n;  
  for (int i = 0; i < nrow; i++) {
    v[i] = w[i] = z[i] = y[i] = 0.0;
  }  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  int kkc = kc - A.ptrow[k];
	  for (int ld = A.ptrow[l]; ld < ptrowend3[ia][jjb][kkc]; ld++) {
	    int m = A.indcol[ld];	
	    y[l] += A.coef[ld] * x[m];
	  }
	  z[k] += A.coef[kc] * y[l];	
	} // loop : jb
	w[j] += A.coef[jb] * z[k];
      }
      v[i] += A.coef[ia] * w[j];      
    } // loop : ia
  }   // loop : i
}

__attribute__((optimize("O3")))
__attribute__((target("fma,avx2")))
void SpM4V_AVX2(double* y4, double* y3, double* y2, double* y1, const double* x,
                const csrmatrix& A,
                const std::vector<int>& ptrowend1,
                const std::vector<std::vector<int>>& ptrowend2,
                const std::vector<std::vector<std::vector<int>>>& ptrowend3)
{
    int n = A.n;
    std::memset(y1, 0, n * sizeof(double));
    std::memset(y2, 0, n * sizeof(double));
    std::memset(y3, 0, n * sizeof(double));
    std::memset(y4, 0, n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ++ia) {
            int j = A.indcol[ia];
            int jb_start = A.ptrow[j];
            int jb_end = ptrowend1[ia];
            for (int jb = jb_start; jb < jb_end; ++jb) {
                int k = A.indcol[jb];
                int jjb = jb - jb_start;
                int kc_start = A.ptrow[k];
                int kc_end = ptrowend2[ia][jjb];
                for (int kc = kc_start; kc < kc_end; ++kc) {
                    int l = A.indcol[kc];
                    int kkc = kc - kc_start;
                    int ld_start = A.ptrow[l];
                    int ld_end = ptrowend3[ia][jjb][kkc];

                    __m256d acc = _mm256_setzero_pd();
                    int ld = ld_start;
                    for (; ld <= ld_end - 4; ld += 4) {
                        int m0 = A.indcol[ld + 0];
                        int m1 = A.indcol[ld + 1];
                        int m2 = A.indcol[ld + 2];
                        int m3 = A.indcol[ld + 3];
                        __m256d xm = _mm256_set_pd(x[m3], x[m2], x[m1], x[m0]);
                        __m256d am = _mm256_loadu_pd(&A.coef[ld]);
                        acc = _mm256_fmadd_pd(am, xm, acc);
                    }
                    double sum = 0.0;
                    __m128d lo = _mm256_castpd256_pd128(acc);
                    __m128d hi = _mm256_extractf128_pd(acc, 1);
                    __m128d sum2 = _mm_add_pd(lo, hi);
                    __m128d sum1 = _mm_hadd_pd(sum2, sum2);
                    sum += ((double*)&sum1)[0];
                    for (; ld < ld_end; ++ld) {
                        int m = A.indcol[ld];
                        sum += A.coef[ld] * x[m];
                    }
                    y1[l] += sum;
                    y2[k] += A.coef[kc] * y1[l];
                }
                y3[j] += A.coef[jb] * y2[k];
            }
            y4[i] += A.coef[ia] * y3[j];
        }
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
                    bcsr4x4_matrix& A)
{
    const int nblocks = nrow / 4;
    for (int k = 0; k < nnz; ++k) {
        int i = irow[k];
        int j = jcol[k];
        double v = val[k];

        int bi = i / 4;   // bloc ligne
        int bj = j / 4;   // bloc colonne
        int ii = i % 4;   // position dans le bloc
        int jj = j % 4;

        // Cherche si le bloc (bi, bj) existe déjà
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

    // Conversion vers vecteurs
    A.nblocks = nblocks;
    A.ptrow.resize(nblocks + 1, 0);

    for (int bi = 0; bi < nblocks; ++bi) {
        A.ptrow[bi + 1] = A.ptrow[bi] + block_rows[bi].size();
        for (const auto& pair : block_rows[bi]) {
            A.indcol.push_back(pair.first);
            A.coef.push_back(pair.second);
        }
    }
}


__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
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
__attribute__((target("fma,avx2")))
void SpMV_AVX2(double *y, const double *x, const csrmatrix &a)
{
  int nrow = a.n;

  for (int i = 0; i < nrow; i++) {
    __m256d acc = _mm256_setzero_pd();  // accumulateur vectoriel 
    int k;
    int start = a.ptrow[i];
    int end   = a.ptrow[i+1];

    // boucle vectorisée 4 par 4
    for (k = start; k <= end - 4; k += 4) {
      // charger 4 valeurs x[j]
      __m256d xval = _mm256_set_pd(
        x[a.indcol[k+3]],
        x[a.indcol[k+2]],
        x[a.indcol[k+1]],
        x[a.indcol[k+0]]
      );

      // charger 4 coefficients a.coef[k]
      __m256d aval = _mm256_loadu_pd(&a.coef[k]);

      // acc += aval * xval
      acc = _mm256_fmadd_pd(aval, xval, acc);
    }

    // réduction horizontale de acc
    double temp[4];
    _mm256_storeu_pd(temp, acc);
    y[i] = temp[0] + temp[1] + temp[2] + temp[3];

    // // reste scalaire
    // for (; k < end; ++k) {
    //   y[i] += a.coef[k] * x[a.indcol[k]];
    // }
  }
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


// int main(int argc, char **argv)
// {
//   char fname[256];
//   char buf[1024];
//   int nrow, nnz;

//   FILE *fp;
//   int itmp, jtmp, ktmp;
  
//   strcpy(fname, argv[1]);
  
//   if ((fp = fopen(fname, "r")) == NULL) {
//     fprintf(stderr, "fail to open %s\n", fname);
//   }
//   fgets(buf, 256, fp);
  
//   while (1) {
//     fgets(buf, 256, fp);
//     if (buf[0] != '%') {
//       sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
//       nrow = itmp;
//       nnz = ktmp;
//       break;
//     }
//   }
//   std::vector<int> irow(nnz);
//   std::vector<int> jcol(nnz);
//   std::vector<double>  val(nnz);
//   {
//     int ii = 0;
//     int itmp, jtmp;
//     float vtmp;
//     for (int i = 0; i < nnz; i++) {
//       fscanf(fp, "%d\t%d\t%f", &itmp, &jtmp, &vtmp);
//       irow[ii] = itmp - 1; // zero based
//       jcol[ii] = jtmp - 1; // zero based
//       val[ii] = (double)vtmp;
//       ii++;
//     }
//     fprintf(stderr, "%d\n", ii);
//     nnz = ii;
//   }
  
//   fclose (fp);
  
//   csrmatrix a;
//   COO2CSR(a, nrow, nnz, &irow[0], &jcol[0], &val[0]);

//   std::vector<double> x(nrow, 1.0), b(nrow);
//   std::vector<double> x1(nrow), x2(nrow), x3(nrow), x4(nrow);
//   std::vector<double> y1(nrow), y2(nrow), y3(nrow), y4(nrow);
  
//   SpMV(&b[0], &x[0], a); // b = A * x
//   SpMV(&x1[0], &b[0], a); // x1 = A * b
//   SpMV(&x2[0], &x1[0], a); // x2 = A * x1 = A * A * b
//   SpMV(&x3[0], &x2[0], a); // x3 = A * x2 = A * A * A * b
//   SpMV(&x4[0], &x3[0], a); // x4 = A x3 = A * A * A * A * b    

//   std::vector<int> ptrowend1(nnz);
//   std::vector<std::vector<int> > ptrowend2(nnz);
//   std::vector<std::vector<std::vector<int> > > ptrowend3(nnz);    

//   Generate1stlayer(ptrowend1, a);
//   SpM2V(&y2[0], &y1[0], &b[0], a, ptrowend1);

//   for (int i = 0; i < nrow; i++) {
//     fprintf(stderr, "%d : %g %g : %g %g \n", i, x1[i], y1[i], x2[i], y2[i]);
//   }

//   Generate2ndlayer(ptrowend2, a, ptrowend1);
// #if 0
//   fprintf(stderr, "%s %d\n", __FILE__, __LINE__);
//   for (int i = 0; i < nrow; i++) {
//     for (int ia  = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++) {
//       int j = a.indcol[ia];
//       fprintf(stderr, "%d %d : %d %d %d\n", i, j, a.ptrow[j], a.ptrow[j + 1], ptrowend1[ia]);
//       for (int jb = a.ptrow[j]; jb < ptrowend1[ia]; jb++) {
// 	int k = a.indcol[jb];
// 	int jjb = jb - a.ptrow[j];
// 	fprintf(stderr, "  %d %d %d : %d %d %d\n", i, j, k, a.ptrow[k], a.ptrow[k + 1], ptrowend2[ia][jjb]);	
//       }
//     }
//   }
// #endif
//   SpM3V(&y3[0], &y2[0], &y1[0], &b[0], a, ptrowend1, ptrowend2);
  
//   for (int i = 0; i < nrow; i++) {
//     fprintf(stderr, ": %d : %g %g : %g %g : %g %g\n",
// 	    i, x1[i], y1[i], x2[i], y2[i], x3[i], y3[i]);
//   }

//   Generate3rdlayer(ptrowend3, a, ptrowend1, ptrowend2);  
//   // Version scalaire
//   auto t0 = std::chrono::high_resolution_clock::now();
//   SpM4V(&y4[0], &y3[0], &y2[0], &y1[0], &b[0],
//         a, ptrowend1, ptrowend2, ptrowend3);
//   auto t1 = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double, std::milli> dur_scalar = t1 - t0;
//   fprintf(stderr, "[SpM4V] time = %.3f ms\n", dur_scalar.count());

//   for (int i = 0; i < nrow; i++) {
//     fprintf(stderr, ":: %d : %g %g : %g %g : %g %g : %g %g\n",
//             i, x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]);
//   }

//   // Version AVX2
//   auto t2 = std::chrono::high_resolution_clock::now();
//   SpM4V_AVX2(&y4[0], &y3[0], &y2[0], &y1[0], &b[0],
//             a, ptrowend1, ptrowend2, ptrowend3);
//   auto t3 = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double, std::milli> dur_avx2 = t3 - t2;
//   fprintf(stderr, "[SpM4V_AVX2] time = %.3f ms\n", dur_avx2.count());

//   for (int i = 0; i < nrow; i++) {
//     fprintf(stderr, "AVX2 :: %d : %g %g : %g %g : %g %g : %g %g\n",
//             i, x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]);
//   }

//   // Speedup
//   double speedup = dur_scalar.count() / dur_avx2.count();
//   fprintf(stderr, "Speedup (scalar / AVX2) = %.2fx\n", speedup);
// }

// int main(int argc, char **argv)
// {
//     if (argc < 2) {
//         fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
//         return 1;
//     }

//     // ------------------------ Lecture de la matrice ------------------------
//     char fname[256]; strcpy(fname, argv[1]);
//     FILE *fp = fopen(fname, "r");
//     if (!fp) { fprintf(stderr, "Failed to open %s\n", fname); return 1; }

//     char buf[1024]; fgets(buf, 256, fp);
//     int nrow, nnz, itmp, jtmp, ktmp;
//     while (fgets(buf, 256, fp)) {
//         if (buf[0] != '%') {
//             sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
//             nrow = itmp; nnz = ktmp;
//             break;
//         }
//     }

//     std::vector<int> irow(nnz), jcol(nnz);
//     std::vector<double> val(nnz);
//     for (int i = 0; i < nnz; i++) {
//         float vtmp;
//         fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
//         irow[i] = itmp - 1;
//         jcol[i] = jtmp - 1;
//         val[i] = static_cast<double>(vtmp);
//     }
//     fclose(fp);

//     csrmatrix A;
//     COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

//     std::vector<double> x(nrow, 1.0), b(nrow);
//     std::vector<double> x1(nrow), x2(nrow);  // ref
//     std::vector<double> y1(nrow), y2(nrow);  // SpM4V
//     std::vector<double> z1(nrow), z2(nrow);  // SpMV AVX2

//     std::vector<int> ptrowend1(nnz);

//     // ----------------------------------------------------------------------
//     // 2×SpMV scalaire (référence)
//     // ----------------------------------------------------------------------
//     auto t_ref2_0 = std::chrono::high_resolution_clock::now();
//     SpMV(x1.data(), b.data(), A);
//     SpMV(x2.data(), x1.data(), A);
//     auto t_ref2_1 = std::chrono::high_resolution_clock::now();
//     double time_ref2 = std::chrono::duration<double, std::milli>(t_ref2_1 - t_ref2_0).count();
//     fprintf(stderr, "[SpMV ×2 (scalar)] time = %.3f ms\n", time_ref2);

//     // ----------------------------------------------------------------------
//     // 2×SpMV AVX2
//     // ----------------------------------------------------------------------
//     auto t_avx2_0 = std::chrono::high_resolution_clock::now();
//     SpMV_AVX2(z1.data(), b.data(), A);
//     SpMV_AVX2(z2.data(), z1.data(), A);
//     auto t_avx2_1 = std::chrono::high_resolution_clock::now();
//     double time_avx2 = std::chrono::duration<double, std::milli>(t_avx2_1 - t_avx2_0).count();
//     double err_avx2 = rel_error(x2, z2);
//     fprintf(stderr, "[SpMV ×2 (AVX2)]   time = %.3f ms | rel. error = %.2e\n", time_avx2, err_avx2);

//     // ----------------------------------------------------------------------
//     // SpM2V scalaire
//     // ----------------------------------------------------------------------
//     auto t_spm2v_0 = std::chrono::high_resolution_clock::now();
//     SpM2V(y2.data(), y1.data(), b.data(), A, ptrowend1);
//     auto t_spm2v_1 = std::chrono::high_resolution_clock::now();
//     double time_spm2v = std::chrono::duration<double, std::milli>(t_spm2v_1 - t_spm2v_0).count();
//     double err_spm2v = rel_error(x2, y2);
//     fprintf(stderr, "[SpM2V (scalar)]   time = %.3f ms | rel. error = %.2e\n", time_spm2v, err_spm2v);

//     // ----------------------------------------------------------------------
//     // SpM2V AVX2
//     // ----------------------------------------------------------------------
//     auto t_spm2v_avx_0 = std::chrono::high_resolution_clock::now();
//     SpM2V_AVX2(y2.data(), y1.data(), b.data(), A, ptrowend1);
//     auto t_spm2v_avx_1 = std::chrono::high_resolution_clock::now();
//     double time_spm2v_avx = std::chrono::duration<double, std::milli>(t_spm2v_avx_1 - t_spm2v_avx_0).count();
//     double err_spm2v_avx = rel_error(x2, y2);
//     fprintf(stderr, "[SpM2V (AVX2)]     time = %.3f ms | rel. error = %.2e\n", time_spm2v_avx, err_spm2v_avx);

//     // ----------------------------------------------------------------------
//     // Résumé
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "\n--- Résumé des performances ---\n");
//     fprintf(stderr, "SpMV ×2 (scalar) : %.3f ms [ref]\n", time_ref2);
//     fprintf(stderr, "SpMV ×2 (AVX2)   : %.3f ms | speedup = %.2fx\n", time_avx2, time_ref2 / time_avx2);
//     fprintf(stderr, "SpM2V   (scalar) : %.3f ms | speedup = %.2fx\n", time_spm2v, time_ref2 / time_spm2v);
//     fprintf(stderr, "SpM2V   (AVX2)   : %.3f ms | speedup = %.2fx\n", time_spm2v_avx, time_ref2 / time_spm2v_avx);
// }

// int main(int argc, char **argv)
// {
//     if (argc < 2) {
//         fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
//         return 1;
//     }

//     // ------------------------ Lecture de la matrice ------------------------
//     char fname[256]; strcpy(fname, argv[1]);
//     FILE *fp = fopen(fname, "r");
//     if (!fp) { fprintf(stderr, "Failed to open %s\n", fname); return 1; }

//     char buf[1024]; fgets(buf, 256, fp);
//     int nrow, nnz, itmp, jtmp, ktmp;
//     while (fgets(buf, 256, fp)) {
//         if (buf[0] != '%') {
//             sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
//             nrow = itmp; nnz = ktmp;
//             break;
//         }
//     }

//     std::vector<int> irow(nnz), jcol(nnz);
//     std::vector<double> val(nnz);
//     for (int i = 0; i < nnz; i++) {
//         float vtmp;
//         fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
//         irow[i] = itmp - 1;
//         jcol[i] = jtmp - 1;
//         val[i] = static_cast<double>(vtmp);
//     }
//     fclose(fp);

//     csrmatrix A;
//     COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

//     std::vector<double> x(nrow, 1.0), b(nrow);
//     std::vector<double> x1(nrow), x2(nrow), x3(nrow), x4(nrow);  // ref
//     std::vector<double> y1(nrow), y2(nrow), y3(nrow), y4(nrow);  // SpM4V
//     std::vector<double> z1(nrow), z2(nrow), z3(nrow), z4(nrow);  // SpMV AVX2

//     // ----------------------------------------------------------------------
//     // Référence : 4×SpMV scalaire
//     // ----------------------------------------------------------------------
//     auto t_ref0 = std::chrono::high_resolution_clock::now();
//     SpMV(b.data(), x.data(), A);
//     SpMV(x1.data(), b.data(), A);
//     SpMV(x2.data(), x1.data(), A);
//     SpMV(x3.data(), x2.data(), A);
//     SpMV(x4.data(), x3.data(), A);
//     auto t_ref1 = std::chrono::high_resolution_clock::now();
//     double time_ref = std::chrono::duration<double, std::milli>(t_ref1 - t_ref0).count();
//     fprintf(stderr, "[SpMV ×4 (scalar)] time = %.3f ms\n", time_ref);

//     // ----------------------------------------------------------------------
//     // 4×SpMV AVX2
//     // ----------------------------------------------------------------------
//     auto t_avx0 = std::chrono::high_resolution_clock::now();
//     SpMV_AVX2(b.data(), x.data(), A);
//     SpMV_AVX2(z1.data(), b.data(), A);
//     SpMV_AVX2(z2.data(), z1.data(), A);
//     SpMV_AVX2(z3.data(), z2.data(), A);
//     SpMV_AVX2(z4.data(), z3.data(), A);
//     auto t_avx1 = std::chrono::high_resolution_clock::now();
//     double time_avx = std::chrono::duration<double, std::milli>(t_avx1 - t_avx0).count();
//     double err_avx = rel_error(x4, z4);
//     fprintf(stderr, "[SpMV ×4 (AVX2)]   time = %.3f ms | rel. error = %.2e\n", time_avx, err_avx);

//     // ----------------------------------------------------------------------
//     // SpM4V scalaire
//     // ----------------------------------------------------------------------
//     std::vector<int> ptrowend1(nnz);
//     std::vector<std::vector<int>> ptrowend2(nnz);
//     std::vector<std::vector<std::vector<int>>> ptrowend3(nnz);
//     Generate1stlayer(ptrowend1, A);
//     Generate2ndlayer(ptrowend2, A, ptrowend1);
//     Generate3rdlayer(ptrowend3, A, ptrowend1, ptrowend2);

//     auto t0 = std::chrono::high_resolution_clock::now();
//     SpM4V(y4.data(), y3.data(), y2.data(), y1.data(), b.data(), A, ptrowend1, ptrowend2, ptrowend3);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double time_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count();
//     double err_scalar = rel_error(x4, y4);
//     fprintf(stderr, "[SpM4V (scalar)]   time = %.3f ms | rel. error = %.2e\n", time_scalar, err_scalar);

//     // ----------------------------------------------------------------------
//     // SpM4V AVX2
//     // ----------------------------------------------------------------------
//     auto t2 = std::chrono::high_resolution_clock::now();
//     SpM4V_AVX2(y4.data(), y3.data(), y2.data(), y1.data(), b.data(), A, ptrowend1, ptrowend2, ptrowend3);
//     auto t3 = std::chrono::high_resolution_clock::now();
//     double time_spm4v_avx = std::chrono::duration<double, std::milli>(t3 - t2).count();
//     double err_spm4v_avx = rel_error(x4, y4);
//     fprintf(stderr, "[SpM4V (AVX2)]     time = %.3f ms | rel. error = %.2e\n", time_spm4v_avx, err_spm4v_avx);

//     // ----------------------------------------------------------------------
//     // Résumé
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "\n--- Résumé des performances ---\n");
//     fprintf(stderr, "SpMV ×4 (scalar) : %.3f ms [ref]\n", time_ref);
//     fprintf(stderr, "SpMV ×4 (AVX2)   : %.3f ms | speedup = %.2fx\n", time_avx, time_ref / time_avx);
//     fprintf(stderr, "SpM4V   (scalar) : %.3f ms | speedup = %.2fx\n", time_scalar, time_ref / time_scalar);
//     fprintf(stderr, "SpM4V   (AVX2)   : %.3f ms | speedup = %.2fx\n", time_spm4v_avx, time_ref / time_spm4v_avx);

//     return 0;
    
// }

// int main(int argc, char **argv)
// {
//     if (argc < 2) {
//         fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
//         return 1;
//     }

//     // ------------------------ Lecture de la matrice ------------------------
//     char fname[256]; 
//     strcpy(fname, argv[1]);
//     FILE *fp = fopen(fname, "r");
//     if (!fp) { 
//         fprintf(stderr, "Failed to open %s\n", fname); 
//         return 1; 
//     }

//     char buf[1024]; 
//     fgets(buf, 256, fp);
//     int nrow, nnz, itmp, jtmp, ktmp;
//     while (fgets(buf, 256, fp)) {
//         if (buf[0] != '%') {
//             sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
//             nrow = itmp; 
//             nnz = ktmp;
//             break;
//         }
//     }

//     std::vector<int> irow(nnz), jcol(nnz);
//     std::vector<double> val(nnz);
//     for (int i = 0; i < nnz; i++) {
//         float vtmp;
//         fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
//         irow[i] = itmp - 1;
//         jcol[i] = jtmp - 1;
//         val[i] = static_cast<double>(vtmp);
//     }
//     fclose(fp);

//     csrmatrix A;
//     COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

//     // Vecteurs de travail
//     std::vector<double> x(nrow, 1.0);
//     std::vector<double> result_4spmv(nrow), result_spm4v(nrow);
//     std::vector<double> temp1(nrow), temp2(nrow), temp3(nrow), temp4(nrow);

//     fprintf(stderr, "=== MATRICE CHARGÉE ===\n");
//     fprintf(stderr, "Dimensions: %d×%d\n", nrow, nrow);
//     fprintf(stderr, "Éléments non-nuls: %d\n", A.nnz);
//     fprintf(stderr, "Densité: %.2f%%\n", (double)A.nnz / (nrow * nrow) * 100);

//     // ----------------------------------------------------------------------
//     // PRÉPARATION des structures pour SpM4V
//     // ----------------------------------------------------------------------
//     std::vector<int> ptrowend1(nnz);
//     std::vector<std::vector<int>> ptrowend2(nnz);
//     std::vector<std::vector<std::vector<int>>> ptrowend3(nnz);
    
//     fprintf(stderr, "\n=== PRÉPARATION SpM4V ===\n");
//     auto prep_start = std::chrono::high_resolution_clock::now();
    
//     Generate1stlayer(ptrowend1, A);
//     fprintf(stderr, "Generate1stlayer: OK\n");
    
//     Generate2ndlayer(ptrowend2, A, ptrowend1);
//     fprintf(stderr, "Generate2ndlayer: OK\n");
    
//     Generate3rdlayer(ptrowend3, A, ptrowend1, ptrowend2);
//     fprintf(stderr, "Generate3rdlayer: OK\n");
    
//     auto prep_end = std::chrono::high_resolution_clock::now();
//     double prep_time = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
//     fprintf(stderr, "Temps de préparation: %.3f ms\n", prep_time);

//     // ----------------------------------------------------------------------
//     // TESTS DE PERFORMANCE
//     // ----------------------------------------------------------------------
//     const int NUM_RUNS = 50;  // Nombre d'exécutions pour moyenner
//     fprintf(stderr, "\n=== TESTS DE PERFORMANCE ===\n");
//     fprintf(stderr, "Nombre d'exécutions par test: %d\n", NUM_RUNS);

//     // ----------------------------------------------------------------------
//     // Test 1: 4×SpMV_AVX2 (méthode de référence)
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "\nTest 1: 4×SpMV_AVX2...\n");
//     double total_time_4spmv = 0.0;
    
//     // Échauffement
//     for (int run = 0; run < 5; run++) {
//         SpMV_AVX2(temp1.data(), x.data(), A);
//         SpMV_AVX2(temp2.data(), temp1.data(), A);
//         SpMV_AVX2(temp3.data(), temp2.data(), A);
//         SpMV_AVX2(temp4.data(), temp3.data(), A);
//     }
    
//     // Mesures
//     for (int run = 0; run < NUM_RUNS; run++) {
//         auto t0 = std::chrono::high_resolution_clock::now();
        
//         SpMV_AVX2(temp1.data(), x.data(), A);      // A×x
//         SpMV_AVX2(temp2.data(), temp1.data(), A);  // A²×x
//         SpMV_AVX2(temp3.data(), temp2.data(), A);  // A³×x
//         SpMV_AVX2(temp4.data(), temp3.data(), A);  // A⁴×x
        
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_time_4spmv += std::chrono::duration<double, std::milli>(t1 - t0).count();
//     }
//     double avg_time_4spmv = total_time_4spmv / NUM_RUNS;
//     result_4spmv = temp4;  // Sauvegarder le résultat de référence

//     // ----------------------------------------------------------------------
//     // Test 2: SpM4V_AVX2
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "Test 2: SpM4V_AVX2...\n");
//     double total_time_spm4v = 0.0;
    
//     // Échauffement
//     for (int run = 0; run < 5; run++) {
//         SpM4V_AVX2(temp4.data(), temp3.data(), temp2.data(), temp1.data(), 
//                    x.data(), A, ptrowend1, ptrowend2, ptrowend3);
//     }
    
//     // Mesures
//     for (int run = 0; run < NUM_RUNS; run++) {
//         auto t0 = std::chrono::high_resolution_clock::now();
        
//         SpM4V_AVX2(temp4.data(), temp3.data(), temp2.data(), temp1.data(), 
//                    x.data(), A, ptrowend1, ptrowend2, ptrowend3);
        
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_time_spm4v += std::chrono::duration<double, std::milli>(t1 - t0).count();
//     }
//     double avg_time_spm4v = total_time_spm4v / NUM_RUNS;
//     result_spm4v = temp4;  // Sauvegarder le résultat

//     // ----------------------------------------------------------------------
//     // Test 3: 4×SpMV scalaire (pour comparaison)
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "Test 3: 4×SpMV scalaire...\n");
//     double total_time_4spmv_scalar = 0.0;
    
//     for (int run = 0; run < NUM_RUNS; run++) {
//         auto t0 = std::chrono::high_resolution_clock::now();
        
//         SpMV(temp1.data(), x.data(), A);
//         SpMV(temp2.data(), temp1.data(), A);
//         SpMV(temp3.data(), temp2.data(), A);
//         SpMV(temp4.data(), temp3.data(), A);
        
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_time_4spmv_scalar += std::chrono::duration<double, std::milli>(t1 - t0).count();
//     }
//     double avg_time_4spmv_scalar = total_time_4spmv_scalar / NUM_RUNS;

//     // ----------------------------------------------------------------------
//     // Test 4: SpM4V scalaire (pour comparaison)
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "Test 4: SpM4V scalaire...\n");
//     double total_time_spm4v_scalar = 0.0;
    
//     for (int run = 0; run < NUM_RUNS; run++) {
//         auto t0 = std::chrono::high_resolution_clock::now();
        
//         SpM4V(temp4.data(), temp3.data(), temp2.data(), temp1.data(), 
//               x.data(), A, ptrowend1, ptrowend2, ptrowend3);
        
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_time_spm4v_scalar += std::chrono::duration<double, std::milli>(t1 - t0).count();
//     }
//     double avg_time_spm4v_scalar = total_time_spm4v_scalar / NUM_RUNS;

//     // ----------------------------------------------------------------------
//     // CALCUL DES MÉTRIQUES
//     // ----------------------------------------------------------------------
    
//     // Nombre d'opérations total (identique pour toutes les méthodes)
//     long long total_ops = 0;
//     for (int i = 0; i < nrow; i++) {
//         int degree = A.ptrow[i+1] - A.ptrow[i];
//         total_ops += degree;
//     }
//     total_ops *= 4 * 2;  // 4 SpMV × 2 ops par élément (mult + add)
    
//     // FLOPS pour chaque méthode
//     double flops_4spmv = total_ops / (avg_time_4spmv / 1000.0);
//     double flops_spm4v = total_ops / (avg_time_spm4v / 1000.0);
//     double flops_4spmv_scalar = total_ops / (avg_time_4spmv_scalar / 1000.0);
//     double flops_spm4v_scalar = total_ops / (avg_time_spm4v_scalar / 1000.0);
    
//     // Speedups
//     double speedup_avx2 = avg_time_4spmv / avg_time_spm4v;
//     double speedup_4spmv_vectorization = avg_time_4spmv_scalar / avg_time_4spmv;
//     double speedup_spm4v_vectorization = avg_time_spm4v_scalar / avg_time_spm4v;
//     double speedup_algorithm = avg_time_4spmv_scalar / avg_time_spm4v_scalar;
    
//     // Vérification de la précision
//     double error = rel_error(result_4spmv, result_spm4v);
    
//     // ----------------------------------------------------------------------
//     // AFFICHAGE DES RÉSULTATS
//     // ----------------------------------------------------------------------
//     fprintf(stderr, "\n");
//     fprintf(stderr, "████████████████████████████████████████████████████████████████\n");
//     fprintf(stderr, "███                   RÉSULTATS FINAUX                       ███\n");
//     fprintf(stderr, "████████████████████████████████████████████████████████████████\n");
//     fprintf(stderr, "\n");
    
//     fprintf(stderr, "=== INFORMATIONS GÉNÉRALES ===\n");
//     fprintf(stderr, "Matrice: %d×%d, nnz=%d\n", nrow, nrow, A.nnz);
//     fprintf(stderr, "Opérations totales: %lld\n", total_ops);
//     fprintf(stderr, "Moyenné sur %d exécutions\n", NUM_RUNS);
//     fprintf(stderr, "Temps de préparation SpM4V: %.3f ms\n", prep_time);
//     fprintf(stderr, "\n");
    
//     fprintf(stderr, "=== TEMPS D'EXÉCUTION ===\n");
//     fprintf(stderr, "4×SpMV (scalaire) : %8.3f ms\n", avg_time_4spmv_scalar);
//     fprintf(stderr, "4×SpMV (AVX2)     : %8.3f ms\n", avg_time_4spmv);
//     fprintf(stderr, "SpM4V  (scalaire) : %8.3f ms\n", avg_time_spm4v_scalar);
//     fprintf(stderr, "SpM4V  (AVX2)     : %8.3f ms\n", avg_time_spm4v);
//     fprintf(stderr, "\n");
    
//     fprintf(stderr, "=== PERFORMANCES (GFLOPS) ===\n");
//     fprintf(stderr, "4×SpMV (scalaire) : %8.2f GFLOPS\n", flops_4spmv_scalar / 1e9);
//     fprintf(stderr, "4×SpMV (AVX2)     : %8.2f GFLOPS\n", flops_4spmv / 1e9);
//     fprintf(stderr, "SpM4V  (scalaire) : %8.2f GFLOPS\n", flops_spm4v_scalar / 1e9);
//     fprintf(stderr, "SpM4V  (AVX2)     : %8.2f GFLOPS\n", flops_spm4v / 1e9);
//     fprintf(stderr, "\n");
    
//     fprintf(stderr, "=== SPEEDUPS ===\n");
//     fprintf(stderr, "AVX2 vs Scalaire:\n");
//     fprintf(stderr, "  4×SpMV: %.2fx\n", speedup_4spmv_vectorization);
//     fprintf(stderr, "  SpM4V : %.2fx\n", speedup_spm4v_vectorization);
//     fprintf(stderr, "\n");
//     fprintf(stderr, "Algorithme (scalaire):\n");
//     fprintf(stderr, "  SpM4V vs 4×SpMV: %.2fx\n", speedup_algorithm);
//     fprintf(stderr, "\n");
//     fprintf(stderr, "🎯 COMPARAISON PRINCIPALE (AVX2):\n");
//     fprintf(stderr, "  SpM4V vs 4×SpMV: %.2fx\n", speedup_avx2);
//     fprintf(stderr, "\n");
    
//     fprintf(stderr, "=== ANALYSE ===\n");
//     if (speedup_avx2 > 1.0) {
//         fprintf(stderr, "✅ SpM4V_AVX2 est %.2fx plus rapide que 4×SpMV_AVX2\n", speedup_avx2);
//         fprintf(stderr, "   Gain de performance: %.1f%%\n", (speedup_avx2 - 1.0) * 100);
//     } else {
//         fprintf(stderr, "❌ SpM4V_AVX2 est %.2fx plus lent que 4×SpMV_AVX2\n", 1.0/speedup_avx2);
//         fprintf(stderr, "   Surcoût: %.1f%%\n", (1.0/speedup_avx2 - 1.0) * 100);
//     }
    
//     fprintf(stderr, "Erreur relative: %.2e\n", rel_error);
//     if (error < 1e-12) {
//         fprintf(stderr, "✅ Résultats numériquement identiques\n");
//     } else if (error < 1e-6) {
//         fprintf(stderr, "⚠️  Petite différence numérique (acceptable)\n");
//     } else {
//         fprintf(stderr, "❌ Différence numérique significative\n");
//     }
    
//     fprintf(stderr, "\nCoût de préparation amortisé après: %.1f exécutions\n", 
//             prep_time / avg_time_spm4v);
    
//     fprintf(stderr, "\n=== RECOMMANDATIONS ===\n");
//     if (speedup_avx2 > 1.2) {
//         fprintf(stderr, "🚀 Utiliser SpM4V_AVX2 pour des applications répétitives\n");
//     } else if (speedup_avx2 > 1.05) {
//         fprintf(stderr, "💡 SpM4V_AVX2 légèrement avantageux\n");
//     } else {
//         fprintf(stderr, "🔄 Préférer 4×SpMV_AVX2 pour sa simplicité\n");
//     }
    
//     // Analyse détaillée des performances
//     double best_gflops = std::max({flops_4spmv_scalar, flops_4spmv, flops_spm4v_scalar, flops_spm4v});
//     fprintf(stderr, "\nPerformance pic atteinte: %.2f GFLOPS\n", best_gflops / 1e9);
    
//     return 0;
// }

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
        return 1;
    }

    // ------------------------ Lecture de la matrice ------------------------
    char fname[256]; strcpy(fname, argv[1]);
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "Failed to open %s\n", fname); return 1; }

    char buf[1024]; fgets(buf, 256, fp);
    int nrow, ncol, nnz;
    while (fgets(buf, 256, fp)) {
        if (buf[0] != '%') {
            sscanf(buf, "%d %d %d", &nrow, &ncol, &nnz);
            break;
        }
    }

    std::vector<int> irow(nnz), jcol(nnz);
    std::vector<double> val(nnz);
    for (int i = 0; i < nnz; i++) {
        float vtmp;
        int itmp, jtmp;
        fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
        irow[i] = itmp - 1;
        jcol[i] = jtmp - 1;
        val[i] = static_cast<double>(vtmp);
    }
    fclose(fp);

    // ----------------------------------------------------------------------
    // Conversion COO → CSR
    csrmatrix A;
    COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

    // Conversion CSR → BCSR 4×4
    bcsr4x4_matrix A_bcsr;
    std::list<std::pair<int, std::array<double, 16>>>* tmp_blocks = new std::list<std::pair<int, std::array<double, 16>>>[nrow / 4 + 1];
    generate_BCSR4(tmp_blocks, nrow, nnz, irow.data(), jcol.data(), val.data(), A_bcsr);
    delete[] tmp_blocks;

    // ----------------------------------------------------------------------
    // Allocation des vecteurs
    std::vector<double> x(nrow, 1.0), b(nrow);
    std::vector<double> x1(nrow), x2(nrow);  // ref
    std::vector<double> y1(nrow), y2(nrow);  // SpM2V (CSR)
    std::vector<double> z1(nrow), z2(nrow);  // SpMV AVX2
    std::vector<double> w1(nrow), w2(nrow);  // SpM2V AVX2 (BCSR)

    std::vector<int> ptrowend1(nnz); // inutile pour BCSR

    // ----------------------------------------------------------------------
    // 2×SpMV scalaire (référence)
    auto t_ref2_0 = std::chrono::high_resolution_clock::now();
    SpMV(x1.data(), b.data(), A);
    SpMV(x2.data(), x1.data(), A);
    auto t_ref2_1 = std::chrono::high_resolution_clock::now();
    double time_ref2 = std::chrono::duration<double, std::milli>(t_ref2_1 - t_ref2_0).count();
    fprintf(stderr, "[SpMV ×2 (scalar)] time = %.3f ms\n", time_ref2);

    // ----------------------------------------------------------------------
    // 2×SpMV AVX2
    auto t_avx2_0 = std::chrono::high_resolution_clock::now();
    SpMV_AVX2(z1.data(), b.data(), A);
    SpMV_AVX2(z2.data(), z1.data(), A);
    auto t_avx2_1 = std::chrono::high_resolution_clock::now();
    double time_avx2 = std::chrono::duration<double, std::milli>(t_avx2_1 - t_avx2_0).count();
    double err_avx2 = rel_error(x2, z2);
    fprintf(stderr, "[SpMV ×2 (AVX2)]   time = %.3f ms | rel. error = %.2e\n", time_avx2, err_avx2);

    // ----------------------------------------------------------------------
    // SpM2V (CSR scalar)
    auto t_spm2v_0 = std::chrono::high_resolution_clock::now();
    SpM2V(y2.data(), y1.data(), b.data(), A, ptrowend1);
    auto t_spm2v_1 = std::chrono::high_resolution_clock::now();
    double time_spm2v = std::chrono::duration<double, std::milli>(t_spm2v_1 - t_spm2v_0).count();
    double err_spm2v = rel_error(x2, y2);
    fprintf(stderr, "[SpM2V (CSR)]      time = %.3f ms | rel. error = %.2e\n", time_spm2v, err_spm2v);

    // ----------------------------------------------------------------------
    // SpM2V (BCSR 4x4 + AVX2)
    auto t_bcsr_0 = std::chrono::high_resolution_clock::now();
    SpM2V_BCSR4_AVX2(w2.data(), w1.data(), b.data(), A_bcsr);
    auto t_bcsr_1 = std::chrono::high_resolution_clock::now();
    double time_bcsr = std::chrono::duration<double, std::milli>(t_bcsr_1 - t_bcsr_0).count();
    double err_bcsr = rel_error(x2, w2);
    fprintf(stderr, "[SpM2V (BCSR4×4)]  time = %.3f ms | rel. error = %.2e\n", time_bcsr, err_bcsr);

    // ----------------------------------------------------------------------
    // Résumé
    fprintf(stderr, "\n--- Résumé des performances ---\n");
    fprintf(stderr, "SpMV ×2 (scalar) : %.3f ms [ref]\n", time_ref2);
    fprintf(stderr, "SpMV ×2 (AVX2)   : %.3f ms | speedup = %.2fx\n", time_avx2, time_ref2 / time_avx2);
    fprintf(stderr, "SpM2V   (CSR)    : %.3f ms | speedup = %.2fx\n", time_spm2v, time_ref2 / time_spm2v);
    fprintf(stderr, "SpM2V   (BCSR4x4): %.3f ms | speedup = %.2fx\n", time_bcsr, time_ref2 / time_bcsr);
}