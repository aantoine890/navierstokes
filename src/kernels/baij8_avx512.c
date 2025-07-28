#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "kernels.h"
#include <petsc/private/matimpl.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>

__attribute__((optimize("O3")))
__attribute__((target("fma,avx512f")))
PetscErrorCode MatMult_SeqBAIJ_8_AVX512(Mat A, Vec xx, Vec zz) {
  Mat_SeqBAIJ       *a      = (Mat_SeqBAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *zarray;
  const MatScalar   *v;
  const PetscInt    *idx, *ii, *ridx = NULL;
  PetscInt           mbs, i, j, n, bs2 = a->bs2;
  PetscBool          usecprow = a->compressedrow.use;

  PetscFunctionBegin;

  // Vérifie que la matrice est bien au bon format
  PetscCheck(A->rmap->bs == 8, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
           "MatMult_SeqBAIJ_8_AVX512 requires block size 8, got %d", (int)A->rmap->bs);
  PetscCheck(bs2 == 64, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
             "MatMult_SeqBAIJ_8_AVX512 expects bs2 = 64, got %d", bs2);

  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArrayWrite(zz, &zarray));

  idx = a->j;
  v   = a->a;
  if (usecprow) {
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    PetscCall(PetscArrayzero(zarray, 8 * a->mbs));
  } else {
    mbs = a->mbs;
    ii  = a->i;
  }
  // PetscPrintf(PETSC_COMM_SELF, "[DEBUG] x = %p (mod 64 = %ld)\n", (void*)x, ((uintptr_t)x) % 64);
  // PetscPrintf(PETSC_COMM_SELF, "[DEBUG] z = %p (mod 64 = %ld)\n", (void*)zarray, ((uintptr_t)zarray) % 64);
  // PetscPrintf(PETSC_COMM_SELF, "[DEBUG] v = %p (mod 64 = %ld)\n", (void*)v, ((uintptr_t)v) % 64);

  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;

    // Initialisation des sommes
    __m512d sum0 = _mm512_setzero_pd();
    __m512d sum1 = _mm512_setzero_pd();
    __m512d sum2 = _mm512_setzero_pd();
    __m512d sum3 = _mm512_setzero_pd();
    __m512d sum4 = _mm512_setzero_pd();
    __m512d sum5 = _mm512_setzero_pd();
    __m512d sum6 = _mm512_setzero_pd();
    __m512d sum7 = _mm512_setzero_pd();

    for (j = 0; j < n; j++) {
      PetscInt col = idx[j];

      // Sanity check : indice valide
      PetscCheck(col >= 0 && 8 * col + 7 < A->cmap->n,
                 PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                 "Invalid access: 8 * idx[%d]=%d exceeds x[] size %d",
                 j, col, (int)A->cmap->n);

      const PetscScalar *xb = x + 8 * col;
      const MatScalar   *vb = v + j * 64;

      // Alignement optionnel : DEBUG uniquement
      if (((uintptr_t)xb % 64) != 0 || ((uintptr_t)vb % 64) != 0) {
        PetscPrintf(PETSC_COMM_SELF,
          "⚠️  AVX512 LOAD NON ALIGNED (i=%d j=%d): xb=%p vb=%p\n",
          i, j, (void*)xb, (void*)vb);
      }

      // Chargement bloc 8x8
      __m512d m0 = _mm512_load_pd(vb +  0);
      __m512d m1 = _mm512_load_pd(vb +  8);
      __m512d m2 = _mm512_load_pd(vb + 16);
      __m512d m3 = _mm512_load_pd(vb + 24);
      __m512d m4 = _mm512_load_pd(vb + 32);
      __m512d m5 = _mm512_load_pd(vb + 40);
      __m512d m6 = _mm512_load_pd(vb + 48);
      __m512d m7 = _mm512_load_pd(vb + 56);

      // Broadcast scalaires
      __m512d vx0 = _mm512_set1_pd(xb[0]);
      __m512d vx1 = _mm512_set1_pd(xb[1]);
      __m512d vx2 = _mm512_set1_pd(xb[2]);
      __m512d vx3 = _mm512_set1_pd(xb[3]);
      __m512d vx4 = _mm512_set1_pd(xb[4]);
      __m512d vx5 = _mm512_set1_pd(xb[5]);
      __m512d vx6 = _mm512_set1_pd(xb[6]);
      __m512d vx7 = _mm512_set1_pd(xb[7]);

      // Accumulation
      sum0 = _mm512_fmadd_pd(m0, vx0, sum0);
      sum1 = _mm512_fmadd_pd(m1, vx1, sum1);
      sum2 = _mm512_fmadd_pd(m2, vx2, sum2);
      sum3 = _mm512_fmadd_pd(m3, vx3, sum3);
      sum4 = _mm512_fmadd_pd(m4, vx4, sum4);
      sum5 = _mm512_fmadd_pd(m5, vx5, sum5);
      sum6 = _mm512_fmadd_pd(m6, vx6, sum6);
      sum7 = _mm512_fmadd_pd(m7, vx7, sum7);
    }

    // Réduction finale
    __m512d t0 = _mm512_add_pd(sum0, sum1);
    __m512d t1 = _mm512_add_pd(sum2, sum3);
    __m512d t2 = _mm512_add_pd(sum4, sum5);
    __m512d t3 = _mm512_add_pd(sum6, sum7);
    __m512d sum01 = _mm512_add_pd(t0, t1);
    __m512d sum23 = _mm512_add_pd(t2, t3);
    __m512d res = _mm512_add_pd(sum01, sum23);

    PetscScalar local[8];
    _mm512_storeu_pd(local, res);

    PetscScalar *z = usecprow ? zarray + 8 * ridx[i] : zarray + 8 * i;
    for (int k = 0; k < 8; ++k) z[k] = local[k];

    idx += n;
    v   += n * bs2;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArrayWrite(zz, &zarray));
  PetscCall(PetscLogFlops(2.0 * a->nz * bs2 - 8.0 * a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}