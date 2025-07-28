#include <time.h>
#include "kernels.h"
#include <petsc/private/matimpl.h>
#include <immintrin.h>
#include <stdint.h>

// Produit matrice-vecteur optimisé AVX2 pour des blocs 4×4 au format MATSEQBAIJ
PetscErrorCode MatMult_SeqBAIJ_4_AVX2(Mat A, Vec xx, Vec zz) {
  Mat_SeqBAIJ       *a      = (Mat_SeqBAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *zarray;
  const MatScalar   *v;
  const PetscInt    *idx, *ii, *ridx = NULL;
  PetscInt           mbs, i, j, n, bs2 = a->bs2;
  PetscBool          usecprow = a->compressedrow.use;

  PetscFunctionBegin;

  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArrayWrite(zz, &zarray));

  idx = a->j;
  v   = a->a;

  if (usecprow) {
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    PetscCall(PetscArrayzero(zarray, 4 * a->mbs));
  } else {
    mbs = a->mbs;
    ii  = a->i;
  }

  for (i = 0; i < mbs; i++) {
    n = ii[1] - ii[0];
    ii++;

    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();

    for (j = 0; j < n; j++) {
      const PetscScalar *xb = x + 4 * idx[j];

      __m256d m0  = _mm256_load_pd(v + j*16 +  0);
      __m256d m1  = _mm256_load_pd(v + j*16 +  4);
      __m256d m2  = _mm256_load_pd(v + j*16 +  8);
      __m256d m3  = _mm256_load_pd(v + j*16 + 12);

      __m256d vx0 = _mm256_set1_pd(xb[0]);
      __m256d vx1 = _mm256_set1_pd(xb[1]);
      __m256d vx2 = _mm256_set1_pd(xb[2]);
      __m256d vx3 = _mm256_set1_pd(xb[3]);

      sum0 = _mm256_fmadd_pd(m0, vx0, sum0);
      sum1 = _mm256_fmadd_pd(m1, vx1, sum1);
      sum2 = _mm256_fmadd_pd(m2, vx2, sum2);
      sum3 = _mm256_fmadd_pd(m3, vx3, sum3);
    }

    __m256d t0  = _mm256_add_pd(sum0, sum1);
    __m256d t1  = _mm256_add_pd(sum2, sum3);
    __m256d res = _mm256_add_pd(t0, t1);

    PetscScalar local[4];
    _mm256_store_pd(local, res);

    PetscScalar *z = usecprow ? (zarray + 4 * ridx[i]) : (zarray + 4 * i);
    z[0] = local[0];
    z[1] = local[1];
    z[2] = local[2];
    z[3] = local[3];

    idx += n;
    v   += n * bs2;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArrayWrite(zz, &zarray));
  PetscCall(PetscLogFlops(2.0 * a->nz * bs2 - 4.0 * a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}