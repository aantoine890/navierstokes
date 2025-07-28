#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include <petscmat.h>
#include <petscvec.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/aij/seq/aij.h>

// AIJ
PetscErrorCode MatMult_SeqAIJ(Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_SeqAIJ_FMA   (Mat A, Vec X, Vec Y);

// BAIJ 4×4
PetscErrorCode MatMult_SeqBAIJ_4    (Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_SeqBAIJ_4_FMA(Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_SeqBAIJ_4_AVX2(Mat A, Vec X, Vec Y);

// BAIJ 8×8
PetscErrorCode MatMult_SeqBAIJ_8_AVX512(Mat A, Vec X, Vec Y);

// Sélecteur de variantes (MatMult selon -matmult_variant)
PetscErrorCode MatMult_SeqBAIJ_4_VariantSelector(Mat A, Vec X, Vec Y);

// BAIJ 4×4 Inplace LU Factorization
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_AVX2(Mat C, Mat A, const MatFactorInfo *info);
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat C, Mat A, const MatFactorInfo *info);

// BAIJ 4×4 Solve
PetscErrorCode MatSolve_SeqBAIJ_4(Mat A, Vec bb, Vec xx);
PetscErrorCode MatSolve_SeqBAIJ_4_AVX2(Mat A, Vec bb, Vec xx);

PetscErrorCode MatSolve_SeqBAIJ_4_inplace(Mat A, Vec bb, Vec xx);
PetscErrorCode MatSolve_SeqBAIJ_4_inplace_AVX2(Mat A, Vec bb, Vec xx);

// BAIJ 4×4 MatMatMult
PetscErrorCode MatMatMult_SeqBAIJ_4_AVX2(Mat A, Mat X, Mat Y, PetscInt s_step);

// spmv/kernels.h (ajoutez en bas)
double diff_time_s(const struct timespec*, const struct timespec*);
void   SpMVKernelTimerReset(void);
void   SpMVKernelTimerGet(double *time_s, long *calls);

#endif // SPMV_KERNELS_H