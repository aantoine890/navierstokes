#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <list>
#include <cmath>
#include <immintrin.h>
#include <algorithm>

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

void generate_CSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp, 
                  int nrow, int nnz, 
                  int *irow, int *jcol, double* val) {
  for (int i = 0; i < nnz; i++) {
    int ii = irow[i], jj = jcol[i];
    if (ind_cols_tmp[ii].empty() || ind_cols_tmp[ii].back() < jj) {
      ind_cols_tmp[ii].push_back(jj);
      val_tmp[ii].push_back(val[i]);
    } else {
      auto it = ind_cols_tmp[ii].begin();
      auto iv = val_tmp[ii].begin();
      for (; it != ind_cols_tmp[ii].end(); ++it, ++iv) {
        if (*it == jj) break;
        if (*it > jj) {
          ind_cols_tmp[ii].insert(it, jj);
          val_tmp[ii].insert(iv, val[i]);
          break;
        }
      }
    }
  }
}

void COO2CSR(csrmatrix &a, int nrow, int nnz, int *irow, int *jcol, double *val) {
  a.n = nrow; a.nnz = nnz;
  a.ptrow.resize(nrow + 1);
  a.indcol.resize(nnz);
  a.coef.resize(nnz);
  std::vector<std::list<int>> ind_cols_tmp(nrow);
  std::vector<std::list<double>> val_tmp(nrow);
  generate_CSR(&ind_cols_tmp[0], &val_tmp[0], nrow, nnz, irow, jcol, val);
  int k = 0;
  a.ptrow[0] = 0;
  for (int i = 0; i < nrow; i++) {
    auto jt = ind_cols_tmp[i].begin();
    auto jv = val_tmp[i].begin();
    for (; jt != ind_cols_tmp[i].end(); ++jt, ++jv) {
      a.indcol[k] = *jt;
      a.coef[k] = *jv;
      k++;
    }
    a.ptrow[i + 1] = k;
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

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpMV(double *y, double *x, csrmatrix &a) {
  int nrow = a.n;
  for (int i = 0; i < nrow; i++) {
    y[i] = 0.0;
    for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++) {
      y[i] += a.coef[ia] * x[a.indcol[ia]];
    }
  }
}

__attribute__((optimize("O3")))
__attribute__((target("fma,avx2")))
void SpMV_AVX2(double *y, const double *x, const csrmatrix &a) {
  int nrow = a.n;
  for (int i = 0; i < nrow; i++) {
    __m256d acc = _mm256_setzero_pd();
    int k = a.ptrow[i], end = a.ptrow[i + 1];
    for (; k <= end - 4; k += 4) {
      __m256d xval = _mm256_set_pd(x[a.indcol[k + 3]], x[a.indcol[k + 2]], x[a.indcol[k + 1]], x[a.indcol[k + 0]]);
      __m256d aval = _mm256_loadu_pd(&a.coef[k]);
      acc = _mm256_fmadd_pd(aval, xval, acc);
    }
    double temp[4]; _mm256_storeu_pd(temp, acc);
    y[i] = temp[0] + temp[1] + temp[2] + temp[3];
  }
}

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

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1) {
    int nrow = A.n;
    std::memset(z, 0, nrow * sizeof(double));
    std::memset(y, 0, nrow * sizeof(double));
    
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    const int* ptrowend1_ptr = ptrowend1.data();
    
    // Calcul de y = A * x d'abord
    for (int i = 0; i < nrow; i++) {
        for (int ia = ptrow[i]; ia < ptrow[i + 1]; ia++) {
            int j = indcol[ia];
            y[i] += coef[ia] * x[j];
        }
    }
    
    // Puis calcul de z = A * y
    for (int i = 0; i < nrow; i++) {
        for (int ia = ptrow[i]; ia < ptrow[i + 1]; ia++) {
            int j = indcol[ia];
            z[i] += coef[ia] * y[j];
        }
    }
}

// __attribute__((optimize("O3")))
// __attribute__((target("no-sse,no-avx2,no-fma")))
// void SpM2V(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1) {
//     int nrow = A.n;
    
//     // Initialisation des vecteurs
//     for (int i = 0; i < nrow; i++) {
//         z[i] = y[i] = 0.0;
//     }
    
//     // Calcul optimisé utilisant ptrowend1
//     for (int i = 0; i < nrow; i++) {
//         for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
//             int j = A.indcol[ia];
            
//             // Calculer y[j] seulement la première fois qu'on rencontre j
//             // ptrowend1[ia] indique jusqu'où aller pour cette première occurrence
//             for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
//                 int k = A.indcol[jb];
//                 y[j] += A.coef[jb] * x[k];
//             }
            
//             // Calculer z[i] avec le y[j] mis à jour
//             z[i] += A.coef[ia] * y[j];
//         }
//     }
// }

__attribute__((optimize("O3")))
__attribute__((target("avx2,fma")))
void SpM2V_BCSR4_AVX2(double *z, double *y, const double *x, const bcsr4x4_matrix &A) {
    const int nrow = A.nrows * 4;
    std::memset(z, 0, nrow * sizeof(double));
    std::memset(y, 0, nrow * sizeof(double));
    
    // ÉTAPE 1: y = A * x avec FMA
    for (int bi = 0; bi < A.nrows; bi++) {
        const int row_base = 4 * bi;
        
        // Accumulateurs pour les 4 lignes de ce bloc
        __m256d y_block = _mm256_loadu_pd(&y[row_base]);
        
        for (int k = A.ptrow[bi]; k < A.ptrow[bi + 1]; ++k) {
            const int bj = A.indcol[k];
            const int col_base = 4 * bj;
            const std::array<double, 16>& block = A.coef[k];
            
            // Chargement du vecteur x pour ce bloc de colonnes
            __m256d x_vec = _mm256_loadu_pd(&x[col_base]);
            
            // Multiplication matrice 4x4 * vecteur avec FMA
            // Approche plus efficace: transposer la logique
            
            // Ligne 0: block[0:3] * x_vec
            __m256d row0 = _mm256_loadu_pd(&block[0]);
            __m256d prod0 = _mm256_mul_pd(row0, x_vec);
            __m256d temp0 = _mm256_hadd_pd(prod0, prod0);
            double sum0 = _mm256_cvtsd_f64(temp0) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp0, temp0, 1));
            
            // Ligne 1: block[4:7] * x_vec
            __m256d row1 = _mm256_loadu_pd(&block[4]);
            __m256d prod1 = _mm256_mul_pd(row1, x_vec);
            __m256d temp1 = _mm256_hadd_pd(prod1, prod1);
            double sum1 = _mm256_cvtsd_f64(temp1) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp1, temp1, 1));
            
            // Ligne 2: block[8:11] * x_vec
            __m256d row2 = _mm256_loadu_pd(&block[8]);
            __m256d prod2 = _mm256_mul_pd(row2, x_vec);
            __m256d temp2 = _mm256_hadd_pd(prod2, prod2);
            double sum2 = _mm256_cvtsd_f64(temp2) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp2, temp2, 1));
            
            // Ligne 3: block[12:15] * x_vec
            __m256d row3 = _mm256_loadu_pd(&block[12]);
            __m256d prod3 = _mm256_mul_pd(row3, x_vec);
            __m256d temp3 = _mm256_hadd_pd(prod3, prod3);
            double sum3 = _mm256_cvtsd_f64(temp3) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp3, temp3, 1));
            
            // Création du vecteur résultat pour ce bloc
            __m256d block_result = _mm256_set_pd(sum3, sum2, sum1, sum0);
            
            // Accumulation avec FMA
            y_block = _mm256_fmadd_pd(_mm256_set1_pd(1.0), block_result, y_block);
        }
        
        // Stockage du résultat y pour ce bloc de lignes
        _mm256_storeu_pd(&y[row_base], y_block);
    }
    
    // ÉTAPE 2: z = A * y avec FMA
    for (int bi = 0; bi < A.nrows; bi++) {
        const int row_base = 4 * bi;
        
        // Accumulateurs pour les 4 lignes de ce bloc
        __m256d z_block = _mm256_loadu_pd(&z[row_base]);
        
        for (int k = A.ptrow[bi]; k < A.ptrow[bi + 1]; ++k) {
            const int bj = A.indcol[k];
            const int col_base = 4 * bj;
            const std::array<double, 16>& block = A.coef[k];
            
            // Chargement du vecteur y pour ce bloc de colonnes
            __m256d y_vec = _mm256_loadu_pd(&y[col_base]);
            
            // Multiplication matrice 4x4 * vecteur avec FMA
            
            // Ligne 0: block[0:3] * y_vec
            __m256d row0 = _mm256_loadu_pd(&block[0]);
            __m256d prod0 = _mm256_mul_pd(row0, y_vec);
            __m256d temp0 = _mm256_hadd_pd(prod0, prod0);
            double sum0 = _mm256_cvtsd_f64(temp0) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp0, temp0, 1));
            
            // Ligne 1: block[4:7] * y_vec
            __m256d row1 = _mm256_loadu_pd(&block[4]);
            __m256d prod1 = _mm256_mul_pd(row1, y_vec);
            __m256d temp1 = _mm256_hadd_pd(prod1, prod1);
            double sum1 = _mm256_cvtsd_f64(temp1) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp1, temp1, 1));
            
            // Ligne 2: block[8:11] * y_vec
            __m256d row2 = _mm256_loadu_pd(&block[8]);
            __m256d prod2 = _mm256_mul_pd(row2, y_vec);
            __m256d temp2 = _mm256_hadd_pd(prod2, prod2);
            double sum2 = _mm256_cvtsd_f64(temp2) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp2, temp2, 1));
            
            // Ligne 3: block[12:15] * y_vec
            __m256d row3 = _mm256_loadu_pd(&block[12]);
            __m256d prod3 = _mm256_mul_pd(row3, y_vec);
            __m256d temp3 = _mm256_hadd_pd(prod3, prod3);
            double sum3 = _mm256_cvtsd_f64(temp3) + _mm256_cvtsd_f64(_mm256_permute2f128_pd(temp3, temp3, 1));
            
            // Création du vecteur résultat pour ce bloc
            __m256d block_result = _mm256_set_pd(sum3, sum2, sum1, sum0);
            
            // Accumulation avec FMA
            z_block = _mm256_fmadd_pd(_mm256_set1_pd(1.0), block_result, z_block);
        }
        
        // Stockage du résultat z pour ce bloc de lignes
        _mm256_storeu_pd(&z[row_base], z_block);
    }
}

double rel_error(const std::vector<double> &ref, const std::vector<double> &test) {
  double s = 0.0, ref_norm = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double diff = ref[i] - test[i];
    s += diff * diff;
    ref_norm += ref[i] * ref[i];
  }
  return std::sqrt(s / ref_norm);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s matrix.mtx\n", argv[0]);
    return 1;
  }

  FILE *fp = fopen(argv[1], "r");
  if (!fp) { fprintf(stderr, "Failed to open %s\n", argv[1]); return 1; }

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
    int itmp, jtmp;
    float vtmp;
    fscanf(fp, "%d %d %f", &itmp, &jtmp, &vtmp);
    irow[i] = itmp - 1;
    jcol[i] = jtmp - 1;
    val[i] = static_cast<double>(vtmp);
  }
  fclose(fp);

  csrmatrix A;
  COO2CSR(A, nrow, nnz, irow.data(), jcol.data(), val.data());

  bcsr4x4_matrix A_bcsr;
  std::list<std::pair<int, std::array<double, 16>>>* tmp_blocks = new std::list<std::pair<int, std::array<double, 16>>>[nrow / 4 + 1];
  generate_BCSR4(tmp_blocks, nrow, nnz, irow.data(), jcol.data(), val.data(), A_bcsr);
  delete[] tmp_blocks;

  std::vector<double> x(nrow, 1.0);
  std::vector<double> b(nrow, 1.0);
  std::vector<double> x1(nrow), x2(nrow);
  std::vector<double> y1(nrow), y2(nrow);
  std::vector<double> z1(nrow), z2(nrow);
  std::vector<double> w1(nrow), w2(nrow);
  std::vector<int> ptrowend1(nnz);

  auto t_ref2_0 = std::chrono::high_resolution_clock::now();
  SpMV(x1.data(), b.data(), A);
  SpMV(x2.data(), x1.data(), A);
  auto t_ref2_1 = std::chrono::high_resolution_clock::now();
  double time_ref2 = std::chrono::duration<double, std::milli>(t_ref2_1 - t_ref2_0).count();
  fprintf(stderr, "[SpMV \u00d72 (scalar)] time = %.3f ms\n", time_ref2);

  auto t_avx2_0 = std::chrono::high_resolution_clock::now();
  SpMV_AVX2(z1.data(), b.data(), A);
  SpMV_AVX2(z2.data(), z1.data(), A);
  auto t_avx2_1 = std::chrono::high_resolution_clock::now();
  double time_avx2 = std::chrono::duration<double, std::milli>(t_avx2_1 - t_avx2_0).count();
  double err_avx2 = rel_error(x2, z2);
  fprintf(stderr, "[SpMV \u00d72 (AVX2)]   time = %.3f ms | rel. error = %.2e\n", time_avx2, err_avx2);

  Generate1stlayer(ptrowend1, A);

  auto t_spm2v_0 = std::chrono::high_resolution_clock::now();
  SpM2V(y2.data(), y1.data(), b.data(), A, ptrowend1);
  auto t_spm2v_1 = std::chrono::high_resolution_clock::now();
  double time_spm2v = std::chrono::duration<double, std::milli>(t_spm2v_1 - t_spm2v_0).count();
  double err_spm2v = rel_error(x2, y2);
  fprintf(stderr, "[SpM2V (CSR)]      time = %.3f ms | rel. error = %.2e\n", time_spm2v, err_spm2v);

  auto t_bcsr_0 = std::chrono::high_resolution_clock::now();
  SpM2V_BCSR4_AVX2(w2.data(), w1.data(), b.data(), A_bcsr);
  auto t_bcsr_1 = std::chrono::high_resolution_clock::now();
  double time_bcsr = std::chrono::duration<double, std::milli>(t_bcsr_1 - t_bcsr_0).count();
  double err_bcsr = rel_error(x2, w2);
  fprintf(stderr, "[SpM2V (BCSR4\u00d74)]  time = %.3f ms | rel. error = %.2e\n", time_bcsr, err_bcsr);

  fprintf(stderr, "\n--- R\u00e9sum\u00e9 des performances ---\n");
  fprintf(stderr, "SpMV \u00d72 (scalar) : %.3f ms [ref]\n", time_ref2);
  fprintf(stderr, "SpMV \u00d72 (AVX2)   : %.3f ms | speedup = %.2fx\n", time_avx2, time_ref2 / time_avx2);
  fprintf(stderr, "SpM2V   (CSR)    : %.3f ms | speedup = %.2fx\n", time_spm2v, time_ref2 / time_spm2v);
  fprintf(stderr, "SpM2V   (BCSR4x4): %.3f ms | speedup = %.2fx\n", time_bcsr, time_ref2 / time_bcsr);

  return 0;
}