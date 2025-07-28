#include <immintrin.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/kernels/blockinvert.h>
#include "kernels.h"
#include <time.h>

static inline void Kernel_A_gets_inverse_A_4_nopivot_scalar(MatScalar *mat) {
    double d, di;
    di = mat[0];
    mat[0] = d = 1.0 / di;
    mat[4] *= -d; mat[8] *= -d; mat[12] *= -d;
    mat[1] *= d; mat[2] *= d; mat[3] *= d;
    mat[5] += mat[4] * mat[1] * di;
    mat[6] += mat[4] * mat[2] * di;
    mat[7] += mat[4] * mat[3] * di;
    mat[9] += mat[8] * mat[1] * di;
    mat[10] += mat[8] * mat[2] * di;
    mat[11] += mat[8] * mat[3] * di;
    mat[13] += mat[12] * mat[1] * di;
    mat[14] += mat[12] * mat[2] * di;
    mat[15] += mat[12] * mat[3] * di;
    di = mat[5]; mat[5] = d = 1.0 / di;
    mat[1] *= -d; mat[9] *= -d; mat[13] *= -d;
    mat[4] *= d; mat[6] *= d; mat[7] *= d;
    mat[0] += mat[1] * mat[4] * di;
    mat[2] += mat[1] * mat[6] * di;
    mat[3] += mat[1] * mat[7] * di;
    mat[8] += mat[9] * mat[4] * di;
    mat[10] += mat[9] * mat[6] * di;
    mat[11] += mat[9] * mat[7] * di;
    mat[12] += mat[13] * mat[4] * di;
    mat[14] += mat[13] * mat[6] * di;
    mat[15] += mat[13] * mat[7] * di;
    di = mat[10]; mat[10] = d = 1.0 / di;
    mat[2] *= -d; mat[6] *= -d; mat[14] *= -d;
    mat[8] *= d; mat[9] *= d; mat[11] *= d;
    mat[0] += mat[2] * mat[8] * di;
    mat[1] += mat[2] * mat[9] * di;
    mat[3] += mat[2] * mat[11] * di;
    mat[4] += mat[6] * mat[8] * di;
    mat[5] += mat[6] * mat[9] * di;
    mat[7] += mat[6] * mat[11] * di;
    mat[12] += mat[14] * mat[8] * di;
    mat[13] += mat[14] * mat[9] * di;
    mat[15] += mat[14] * mat[11] * di;
    di = mat[15]; mat[15] = d = 1.0 / di;
    mat[3] *= -d; mat[7] *= -d; mat[11] *= -d;
    mat[12] *= d; mat[13] *= d; mat[14] *= d;
    mat[0] += mat[3] * mat[12] * di;
    mat[1] += mat[3] * mat[13] * di;
    mat[2] += mat[3] * mat[14] * di;
    mat[4] += mat[7] * mat[12] * di;
    mat[5] += mat[7] * mat[13] * di;
    mat[6] += mat[7] * mat[14] * di;
    mat[8] += mat[11] * mat[12] * di;
    mat[9] += mat[11] * mat[13] * di;
    mat[10] += mat[11] * mat[14] * di;
}

static inline PetscErrorCode Kernel_A_gets_inverse_A_4(MatScalar *a, PetscReal shift, PetscBool allowzeropivot, PetscBool *zeropivotdetected) {
    *zeropivotdetected = PETSC_FALSE;
    for (int k = 0; k < 4; ++k) {
        if (fabs(a[k * 5]) < 1e-12) {
            if (!allowzeropivot) return -1;
            *zeropivotdetected = PETSC_TRUE;
        }
    }
    Kernel_A_gets_inverse_A_4_nopivot_scalar(a);
    return 0;
}

static inline void block4x4_matmul(const double *A, const double *B, double *C) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += A[4 * i + k] * B[4 * k + j];
            C[4 * i + j] = sum;
        }
}

static inline void block4x4_matmul_sub(double *A, const double *B, const double *C) {
    double tmp[16];
    block4x4_matmul(B, C, tmp);
    for (int k = 0; k < 16; ++k) A[k] -= tmp[k];
}

__attribute__((optimize("O0")))
__attribute__((target("no-avx,no-sse,no-fma")))
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat B, Mat A, const MatFactorInfo *info) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    Mat             C = B;
    Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
    IS              isrow = b->row, isicol = b->icol;
    const PetscInt *r, *ic;
    PetscInt        i, j, k, nz, nzL, row;
    const PetscInt  n = a->mbs, *ai = a->i, *aj = a->j, *bi = b->i, *bj = b->j;
    const PetscInt *ajtmp, *bjtmp, *bdiag = b->diag, *pj, bs2 = a->bs2;
    MatScalar      *rtmp, *pc, *mwork, *v, *pv, *aa = a->a;
    PetscInt        flg;
    PetscReal       shift;
    PetscBool       allowzeropivot, zeropivotdetected;

    PetscFunctionBegin;
    allowzeropivot = PetscNot(A->erroriffailure);
    PetscCall(ISGetIndices(isrow, &r));
    PetscCall(ISGetIndices(isicol, &ic));

    shift = (info->shifttype == (PetscReal)MAT_SHIFT_NONE) ? 0.0 : info->shiftamount;

    PetscCall(PetscMalloc2(bs2 * n, &rtmp, bs2, &mwork));
    PetscCall(PetscArrayzero(rtmp, bs2 * n));

    for (i = 0; i < n; i++) {
        nz    = bi[i + 1] - bi[i];
        bjtmp = bj + bi[i];
        for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

        nz    = bdiag[i] - bdiag[i + 1];
        bjtmp = bj + bdiag[i + 1] + 1;
        for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

        nz    = ai[r[i] + 1] - ai[r[i]];
        ajtmp = aj + ai[r[i]];
        v     = aa + bs2 * ai[r[i]];
        for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(rtmp + bs2 * ic[ajtmp[j]], v + bs2 * j, bs2));

        bjtmp = bj + bi[i];
        nzL   = bi[i + 1] - bi[i];
        for (k = 0; k < nzL; k++) {
            row = bjtmp[k];
            pc  = rtmp + bs2 * row;
            for (flg = 0, j = 0; j < bs2; j++) {
                if (pc[j] != 0.0) {
                    flg = 1;
                    break;
                }
            }
            if (flg) {
                pv = b->a + bs2 * bdiag[row];
                block4x4_matmul(pc, pv, mwork);

                pj = b->j + bdiag[row + 1] + 1;
                pv = b->a + bs2 * (bdiag[row + 1] + 1);
                nz = bdiag[row] - bdiag[row + 1] - 1;
                for (j = 0; j < nz; j++) {
                    v = rtmp + bs2 * pj[j];
                    block4x4_matmul_sub(v, pc, pv);
                    pv += bs2;
                }
                PetscCall(PetscLogFlops(128.0 * nz + 112));
            }
        }

        pv = b->a + bs2 * bi[i];
        pj = b->j + bi[i];
        nz = bi[i + 1] - bi[i];
        for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));

        pv = b->a + bs2 * bdiag[i];
        pj = b->j + bdiag[i];
        PetscCall(PetscArraycpy(pv, rtmp + bs2 * pj[0], bs2));
        PetscCall(Kernel_A_gets_inverse_A_4(pv, shift, allowzeropivot, &zeropivotdetected));
        if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

        pv = b->a + bs2 * (bdiag[i + 1] + 1);
        pj = b->j + bdiag[i + 1] + 1;
        nz = bdiag[i] - bdiag[i + 1] - 1;
        for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));
    }

    PetscCall(PetscFree2(rtmp, mwork));
    PetscCall(ISRestoreIndices(isicol, &ic));
    PetscCall(ISRestoreIndices(isrow, &r));

    C->ops->solve = MatSolve_SeqBAIJ_4_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4_inplace;
    C->assembled = PETSC_TRUE;
    PetscCall(PetscLogFlops(1.333333333 * 64 * n));

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    PetscPrintf(PETSC_COMM_SELF, "[LU Scalar] Time: %f seconds\n", elapsed);
    PetscFunctionReturn(PETSC_SUCCESS);
}


static inline void block4x4_matmul_avx2(const double *A, const double *B, double *C) {
    for (int i = 0; i < 4; ++i) {
        __m256d a_row = _mm256_loadu_pd(&A[4 * i]);
        for (int j = 0; j < 4; ++j) {
            __m256d b_col = _mm256_set_pd(B[j + 12], B[j + 8], B[j + 4], B[j]);
            __m256d prod = _mm256_mul_pd(a_row, b_col);
            __m128d hi = _mm256_extractf128_pd(prod, 1);
            __m128d lo = _mm256_castpd256_pd128(prod);
            __m128d sum = _mm_add_pd(lo, hi);
            sum = _mm_hadd_pd(sum, sum);
            _mm_store_sd(&C[4 * i + j], sum);
        }
    }
}

static inline void block4x4_matmul_sub_avx2(double *A, const double *B, const double *C) {
    for (int i = 0; i < 4; ++i) {
        __m256d b_row = _mm256_loadu_pd(B + 4 * i);
        for (int j = 0; j < 4; ++j) {
            __m256d c_col = _mm256_set_pd(C[12 + j], C[8 + j], C[4 + j], C[j]);
            __m256d prod  = _mm256_mul_pd(b_row, c_col);
            __m128d low = _mm256_castpd256_pd128(prod);
            __m128d high = _mm256_extractf128_pd(prod, 1);
            __m128d sum = _mm_add_pd(low, high);
            sum = _mm_hadd_pd(sum, sum);
            double total = _mm_cvtsd_f64(sum);
            A[4 * i + j] -= total;
        }
    }
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_AVX2(Mat C, Mat A, const MatFactorInfo *info) {
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Entering AVX2 LU factorization\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    Mat_SeqBAIJ *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
    PetscInt i, j, n = a->mbs, *bi = b->i, *bj = b->j;
    PetscInt *ajtmpold, *ajtmp, nz, row;
    PetscInt *diag_offset = b->diag, *ai = a->i, *aj = a->j, *pj;
    MatScalar *pv, *v, *rtmp, *pc, *w, *x;
    MatScalar *ba = b->a, *aa = a->a;
    PetscBool pivotinblocks = b->pivotinblocks;
    PetscReal shift = info->shiftamount;
    PetscBool allowzeropivot, zeropivotdetected = PETSC_FALSE;

    PetscFunctionBegin;
    allowzeropivot = PetscNot(A->erroriffailure);
    PetscCall(PetscMalloc1(16 * (n + 1), &rtmp));

    for (i = 0; i < n; i++) {
        nz = bi[i + 1] - bi[i];
        ajtmp = bj + bi[i];
        for (j = 0; j < nz; j++) {
            x = rtmp + 16 * ajtmp[j];
            for (int k = 0; k < 16; ++k) x[k] = 0.0;
        }
        nz = ai[i + 1] - ai[i];
        ajtmpold = aj + ai[i];
        v = aa + 16 * ai[i];
        for (j = 0; j < nz; j++) {
            x = rtmp + 16 * ajtmpold[j];
            for (int k = 0; k < 16; ++k) x[k] = v[k];
            v += 16;
        }
        row = *ajtmp++;
        while (row < i) {
            pc = rtmp + 16 * row;
            int not_zero = 0;
            for (int k = 0; k < 16; ++k) if (pc[k] != 0.0) { not_zero = 1; break; }
            if (not_zero) {
                pv = ba + 16 * diag_offset[row];
                pj = bj + diag_offset[row] + 1;
                double tmp[16];
                block4x4_matmul_avx2(pc, pv, tmp);
                for (int k = 0; k < 16; ++k) pc[k] = tmp[k];
                nz = bi[row + 1] - diag_offset[row] - 1;
                pv += 16;
                for (j = 0; j < nz; j++) {
                    x = rtmp + 16 * pj[j];
                    block4x4_matmul_sub_avx2(x, pc, pv);
                    pv += 16;
                }
                PetscCall(PetscLogFlops(128.0 * nz + 112.0));
            }
            row = *ajtmp++;
        }
        pv = ba + 16 * bi[i];
        pj = bj + bi[i];
        nz = bi[i + 1] - bi[i];
        for (j = 0; j < nz; j++) {
            x = rtmp + 16 * pj[j];
            for (int k = 0; k < 16; ++k) pv[k] = x[k];
            pv += 16;
        }
        w = ba + 16*diag_offset[i];
        if (pivotinblocks) {
            PetscCall(Kernel_A_gets_inverse_A_4(w, shift, allowzeropivot, &zeropivotdetected));
            if (zeropivotdetected) {
                C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
                // Regularisation d'urgence
                for (int k=0; k<16; k+=5) w[k] += 1e-8;
                PetscCall(Kernel_A_gets_inverse_A_4(w, 0.0, PETSC_TRUE, NULL));
            }
        } else {
            PetscCall(Kernel_A_gets_inverse_A_4(w, 0.0, PETSC_TRUE, NULL));
        }
    }
    PetscCall(PetscFree(rtmp));
    C->ops->solve = MatSolve_SeqBAIJ_4_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4_inplace;
    C->assembled = PETSC_TRUE;
    PetscCall(PetscLogFlops(1.333333333 * 64 * b->mbs));
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    PetscPrintf(PETSC_COMM_SELF, "[LU AVX2] Time: %f seconds\n", elapsed);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc function to solve a linear system Ax = b for 4x4 blocks 
PetscErrorCode MatSolve_SeqBAIJ_4_inplace(Mat A, Vec bb, Vec xx) {
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, x1, x2, x3, x4, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  /* forward solve the lower triangular */
  idx  = 4 * (*r++);
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    idx = 4 * (*r++);
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idx        = 4 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
  }
  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * diag[i] + 16;
    vi  = aj + diag[i] + 1;
    nz  = ai[i + 1] - diag[i] - 1;
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    while (nz--) {
      idx = 4 * (*vi++);
      x1  = t[idx];
      x2  = t[1 + idx];
      x3  = t[2 + idx];
      x4  = t[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
      v += 16;
    }
    idc    = 4 * (*c--);
    v      = aa + 16 * diag[i];
    x[idc] = t[idt] = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc function to solve a linear transpose system A^T x = b for 4x4 blocks 
PetscErrorCode MatSolveTranspose_SeqBAIJ_4_inplace(Mat A, Vec bb, Vec xx) {
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt    *r, *c, *rout, *cout;
  const PetscInt    *diag = a->diag, n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, ii, ic, ir, oidx;
  const MatScalar   *aa = a->a, *v;
  PetscScalar        s1, s2, s3, s4, x1, x2, x3, x4, *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i = 0; i < n; i++) {
    ic        = 4 * c[i];
    t[ii]     = b[ic];
    t[ii + 1] = b[ic + 1];
    t[ii + 2] = b[ic + 2];
    t[ii + 3] = b[ic + 3];
    ii += 4;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i = 0; i < n; i++) {
    v = aa + 16 * diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];
    x2 = t[1 + idx];
    x3 = t[2 + idx];
    x4 = t[3 + idx];
    s1 = v[0] * x1 + v[1] * x2 + v[2] * x3 + v[3] * x4;
    s2 = v[4] * x1 + v[5] * x2 + v[6] * x3 + v[7] * x4;
    s3 = v[8] * x1 + v[9] * x2 + v[10] * x3 + v[11] * x4;
    s4 = v[12] * x1 + v[13] * x2 + v[14] * x3 + v[15] * x4;
    v += 16;

    vi = aj + diag[i] + 1;
    nz = ai[i + 1] - diag[i] - 1;
    while (nz--) {
      oidx = 4 * (*vi++);
      t[oidx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4;
      t[oidx + 1] -= v[4] * s1 + v[5] * s2 + v[6] * s3 + v[7] * s4;
      t[oidx + 2] -= v[8] * s1 + v[9] * s2 + v[10] * s3 + v[11] * s4;
      t[oidx + 3] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4;
      v += 16;
    }
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
    idx += 4;
  }
  /* backward solve the L^T */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * diag[i] - 16;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    while (nz--) {
      idx = 4 * (*vi--);
      t[idx] -= v[0] * s1 + v[1] * s2 + v[2] * s3 + v[3] * s4;
      t[idx + 1] -= v[4] * s1 + v[5] * s2 + v[6] * s3 + v[7] * s4;
      t[idx + 2] -= v[8] * s1 + v[9] * s2 + v[10] * s3 + v[11] * s4;
      t[idx + 3] -= v[12] * s1 + v[13] * s2 + v[14] * s3 + v[15] * s4;
      v -= 16;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i = 0; i < n; i++) {
    ir        = 4 * r[i];
    x[ir]     = t[ii];
    x[ir + 1] = t[ii + 1];
    x[ir + 2] = t[ii + 2];
    x[ir + 3] = t[ii + 3];
    ii += 4;
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}