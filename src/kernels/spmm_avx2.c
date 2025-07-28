#include <immintrin.h>
#include <petsc/private/matimpl.h>
#include <petscksp.h>
#include "kernels.h"


PetscErrorCode MatMatMult_SeqBAIJ_4_AVX2(Mat A, Mat X, Mat Y, PetscInt s_step) {
    Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
    const PetscScalar *xarray;
    PetscScalar *yarray;
    const MatScalar *aval;
    const PetscInt *aj, *ai, *ridx = NULL;
    PetscInt mbs, brow, b, r;
    PetscInt bs = 4, bs2 = 16;
    PetscBool useCompressed = a->compressedrow.use;
    
    PetscFunctionBegin;
    
    // Initialisation
    PetscCall(MatDenseGetArrayRead(X, &xarray));
    PetscCall(MatDenseGetArray(Y, &yarray));
    
    PetscInt lda = bs * a->mbs; // leading dimension
    
    aj = a->j;
    aval = a->a;
    
    if (useCompressed) {
        mbs = a->compressedrow.nrows;
        ai = a->compressedrow.i;
        ridx = a->compressedrow.rindex;
        PetscCall(PetscArrayzero(yarray, bs * a->mbs * s_step));
    } else {
        mbs = a->mbs;
        ai = a->i;
        PetscCall(PetscArrayzero(yarray, lda * s_step));
    }
    
    // Traitement par blocs de s-step (optimisé pour construction base Krylov)
    for (PetscInt s_block = 0; s_block < s_step; s_block += 4) {
        PetscInt s_remaining = PetscMin(4, s_step - s_block);
        
        for (brow = 0; brow < mbs; ++brow) {
            PetscInt rowOffset = useCompressed ? 4 * ridx[brow] : 4 * brow;
            PetscInt start = ai[brow], end = ai[brow+1];
            
            // Accumulateurs pour le bloc 4x4 de lignes
            __m256d sum[4][4]; // [row_in_block][s_vector]
            for (r = 0; r < 4; ++r) {
                for (PetscInt k = 0; k < 4; ++k) {
                    sum[r][k] = _mm256_setzero_pd();
                }
            }
            
            // Parcours des blocs non-zéros
            for (b = start; b < end; ++b) {
                PetscInt blkCol = aj[b];
                const PetscScalar *Xblock = xarray + bs * blkCol + s_block * lda;
                const MatScalar *Ablock = aval + b * bs2;
                
                // Charger les vecteurs X du bloc courant
                __m256d x[4][4]; // [row_in_X_block][s_vector]
                for (r = 0; r < 4; ++r) {
                    if (s_remaining == 4) {
                        for (PetscInt k = 0; k < 4; ++k) {
                            x[r][k] = _mm256_set1_pd(Xblock[r + k * lda]);
                        }
                    } else {
                        for (PetscInt k = 0; k < s_remaining; ++k) {
                            x[r][k] = _mm256_set1_pd(Xblock[r + k * lda]);
                        }
                        for (PetscInt k = s_remaining; k < 4; ++k) {
                            x[r][k] = _mm256_setzero_pd();
                        }
                    }
                }
                
                // Multiplication du bloc A (4x4) par X (4x4)
                for (r = 0; r < 4; ++r) {
                    for (PetscInt k = 0; k < 4; ++k) {
                        __m256d acc = _mm256_setzero_pd();
                        for (int c = 0; c < 4; ++c) {
                            __m256d a_val = _mm256_set1_pd(Ablock[4 * r + c]);
                            acc = _mm256_fmadd_pd(a_val, x[c][k], acc);
                        }
                        sum[r][k] = _mm256_add_pd(sum[r][k], acc);
                    }
                }
            }
            
            // Stockage dans Y avec réduction horizontale
            for (r = 0; r < 4; ++r) {
                for (PetscInt k = 0; k < s_remaining; ++k) {
                    // Réduction horizontale
                    __m128d low = _mm256_castpd256_pd128(sum[r][k]);
                    __m128d high = _mm256_extractf128_pd(sum[r][k], 1);
                    __m128d sum_128 = _mm_add_pd(low, high);
                    __m128d sum_64 = _mm_hadd_pd(sum_128, sum_128);
                    
                    yarray[rowOffset + r + (s_block + k) * lda] = _mm_cvtsd_f64(sum_64);
                }
            }
        }
    }
    
    PetscCall(MatDenseRestoreArrayRead(X, &xarray));
    PetscCall(MatDenseRestoreArray(Y, &yarray));
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode BuildKrylovBasis_AVX2(KSP ksp, Vec v0, Mat *V_out, PetscInt s_step) {
    Mat A, V;
    PetscInt n;
    Vec *v_work;
    
    PetscFunctionBegin;
    
    PetscCall(KSPGetOperators(ksp, &A, NULL));
    PetscCall(VecGetSize(v0, &n));
    
    // Créer la matrice dense pour stocker les vecteurs de base
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 
                             n, s_step + 1, NULL, &V));
    
    // Initialiser avec v0
    PetscCall(MatDenseGetColumn(V, 0, &v_work));
    PetscCall(VecCopy(v0, *v_work));
    PetscCall(MatDenseRestoreColumn(V, 0, &v_work));
    
    // Construire les s vecteurs suivants : v_{k+1} = A * v_k
    for (PetscInt k = 0; k < s_step; ++k) {
        Mat X_k, Y_k;
        
        // Extraire le vecteur courant
        PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                                 n, 1, NULL, &X_k));
        PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                                 n, 1, NULL, &Y_k));
        
        // Copier v_k dans X_k
        PetscCall(MatDenseGetColumn(V, k, &v_work));
        PetscScalar *x_array;
        PetscCall(MatDenseGetArray(X_k, &x_array));
        PetscCall(VecGetArray(*v_work, &x_array));
        PetscCall(VecRestoreArray(*v_work, &x_array));
        PetscCall(MatDenseRestoreArray(X_k, &x_array));
        PetscCall(MatDenseRestoreColumn(V, k, &v_work));
        
        // Calculer Y_k = A * X_k avec AVX2
        PetscCall(MatMatMult_SeqBAIJ_4_AVX2(A, X_k, Y_k, 1));
        
        // Stocker le résultat dans V(:, k+1)
        PetscCall(MatDenseGetColumn(V, k + 1, &v_work));
        PetscScalar *y_array;
        PetscCall(MatDenseGetArray(Y_k, &y_array));
        PetscCall(VecPlaceArray(*v_work, y_array));
        PetscCall(VecResetArray(*v_work));
        PetscCall(MatDenseRestoreArray(Y_k, &y_array));
        PetscCall(MatDenseRestoreColumn(V, k + 1, &v_work));
        
        PetscCall(MatDestroy(&X_k));
        PetscCall(MatDestroy(&Y_k));
    }
    
    *V_out = V;
    PetscFunctionReturn(PETSC_SUCCESS);
}