#include <petsc/private/matimpl.h>
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include "kernels.h"

PetscErrorCode MatSolve_SeqBAIJ_4_AVX2(Mat A, Vec bb, Vec xx) {
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  IS iscol = a->col, isrow = a->row;
  const PetscInt n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt i, nz, idx, idt, idc, m;
  const PetscInt *r, *c;
  const MatScalar *aa = a->a, *v;
  PetscScalar *x, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(iscol, &c));

  // Initialisation premier bloc
  idx = 4 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];

  // Forward substitution avec AVX2 + FMA
  for (i = 1; i < n; i++) {
    v = aa + 16 * ai[i];
    vi = aj + ai[i];
    nz = ai[i + 1] - ai[i];
    idx = 4 * r[i];

    __m256d s = _mm256_set_pd(b[3 + idx], b[2 + idx], b[1 + idx], b[idx]);

    for (m = 0; m < nz; m++) {
      const double *xp = t + 4 * vi[m];

      __m256d v_col0 = _mm256_set_pd(v[3], v[2], v[1], v[0]);
      __m256d v_col1 = _mm256_set_pd(v[7], v[6], v[5], v[4]);
      __m256d v_col2 = _mm256_set_pd(v[11], v[10], v[9], v[8]);
      __m256d v_col3 = _mm256_set_pd(v[15], v[14], v[13], v[12]);

      s = _mm256_fnmadd_pd(v_col0, _mm256_set1_pd(xp[0]), s);
      s = _mm256_fnmadd_pd(v_col1, _mm256_set1_pd(xp[1]), s);
      s = _mm256_fnmadd_pd(v_col2, _mm256_set1_pd(xp[2]), s);
      s = _mm256_fnmadd_pd(v_col3, _mm256_set1_pd(xp[3]), s);

      v += 16;
    }
    idt = 4 * i;
    _mm256_store_pd(t + idt, s);
  }

  // Backward substitution avec AVX2 + FMA
  for (i = n - 1; i >= 0; i--) {
    v = aa + 16 * (adiag[i + 1] + 1);
    vi = aj + adiag[i + 1] + 1;
    nz = adiag[i] - adiag[i + 1] - 1;
    idt = 4 * i;

    __m256d s = _mm256_load_pd(t + idt);

    for (m = 0; m < nz; m++) {
      const double *xp = t + 4 * vi[m];

      __m256d v_col0 = _mm256_set_pd(v[3], v[2], v[1], v[0]);
      __m256d v_col1 = _mm256_set_pd(v[7], v[6], v[5], v[4]);
      __m256d v_col2 = _mm256_set_pd(v[11], v[10], v[9], v[8]);
      __m256d v_col3 = _mm256_set_pd(v[15], v[14], v[13], v[12]);

      s = _mm256_fnmadd_pd(v_col0, _mm256_set1_pd(xp[0]), s);
      s = _mm256_fnmadd_pd(v_col1, _mm256_set1_pd(xp[1]), s);
      s = _mm256_fnmadd_pd(v_col2, _mm256_set1_pd(xp[2]), s);
      s = _mm256_fnmadd_pd(v_col3, _mm256_set1_pd(xp[3]), s);

      v += 16;
    }

    // Calcul final : x = D^-1 * s (avec D^-1 codé par v)
    idc = 4 * c[i];

    double s_array[4];
    _mm256_storeu_pd(s_array, s);

    __m256d v_col0 = _mm256_set_pd(v[3], v[2], v[1], v[0]);
    __m256d v_col1 = _mm256_set_pd(v[7], v[6], v[5], v[4]);
    __m256d v_col2 = _mm256_set_pd(v[11], v[10], v[9], v[8]);
    __m256d v_col3 = _mm256_set_pd(v[15], v[14], v[13], v[12]);

    __m256d result = _mm256_mul_pd(v_col0, _mm256_set1_pd(s_array[0]));
    result = _mm256_add_pd(result, _mm256_mul_pd(v_col1, _mm256_set1_pd(s_array[1])));
    result = _mm256_add_pd(result, _mm256_mul_pd(v_col2, _mm256_set1_pd(s_array[2])));
    result = _mm256_add_pd(result, _mm256_mul_pd(v_col3, _mm256_set1_pd(s_array[3])));

    _mm256_store_pd(x + idc, result);
    _mm256_store_pd(t + idt, result);
  }

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(iscol, &c));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_inplace_AVX2(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *diag = a->diag;
  const PetscInt    *r, *c, *rout, *cout;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, *t, s1, s2, s3, s4;
  const PetscScalar *b;
  PetscInt           i, nz, idx, idt, idc;
  double             tmp[4];

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout + (n - 1);

  // --- Forward solve ---
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
      PetscInt blkidx = 4 * (*vi++);
      __m256d vx = _mm256_load_pd(&t[blkidx]);

      __m256d r0 = _mm256_fmadd_pd(_mm256_set_pd(v[12], v[8], v[4], v[0]), vx, _mm256_setzero_pd());
      __m256d r1 = _mm256_fmadd_pd(_mm256_set_pd(v[13], v[9], v[5], v[1]), vx, _mm256_setzero_pd());
      __m256d r2 = _mm256_fmadd_pd(_mm256_set_pd(v[14], v[10], v[6], v[2]), vx, _mm256_setzero_pd());
      __m256d r3 = _mm256_fmadd_pd(_mm256_set_pd(v[15], v[11], v[7], v[3]), vx, _mm256_setzero_pd());

      _mm256_store_pd(tmp, r0); s1 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r1); s2 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r2); s3 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r3); s4 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];

      v += 16;
    }

    idx        = 4 * i;
    t[idx]     = s1;
    t[1 + idx] = s2;
    t[2 + idx] = s3;
    t[3 + idx] = s4;
  }

  // --- Backward solve ---
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
      PetscInt blkidx = 4 * (*vi++);
      __m256d vx = _mm256_loadu_pd(&t[blkidx]);

      __m256d r0 = _mm256_fmadd_pd(_mm256_set_pd(v[12], v[8], v[4], v[0]), vx, _mm256_setzero_pd());
      __m256d r1 = _mm256_fmadd_pd(_mm256_set_pd(v[13], v[9], v[5], v[1]), vx, _mm256_setzero_pd());
      __m256d r2 = _mm256_fmadd_pd(_mm256_set_pd(v[14], v[10], v[6], v[2]), vx, _mm256_setzero_pd());
      __m256d r3 = _mm256_fmadd_pd(_mm256_set_pd(v[15], v[11], v[7], v[3]), vx, _mm256_setzero_pd());

      _mm256_store_pd(tmp, r0); s1 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r1); s2 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r2); s3 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];
      _mm256_store_pd(tmp, r3); s4 -= tmp[0] + tmp[1] + tmp[2] + tmp[3];

      v += 16;
    }

    idc = 4 * (*c--);
    v   = aa + 16 * diag[i];
    __m256d s = _mm256_set_pd(s4, s3, s2, s1);

    __m256d r0 = _mm256_mul_pd(_mm256_load_pd(&v[0]), s);
    __m256d r1 = _mm256_mul_pd(_mm256_load_pd(&v[1]), s);
    __m256d r2 = _mm256_mul_pd(_mm256_load_pd(&v[2]), s);
    __m256d r3 = _mm256_mul_pd(_mm256_load_pd(&v[3]), s);

    _mm256_store_pd(tmp, r0); x[idc]     = t[idt]     = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_store_pd(tmp, r1); x[1 + idc] = t[1 + idt] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_store_pd(tmp, r2); x[2 + idc] = t[2 + idt] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_store_pd(tmp, r3); x[3 + idc] = t[3 + idt] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
