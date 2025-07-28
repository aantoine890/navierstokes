#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "kernels.h"
#include <petsc/private/matimpl.h>
#include <math.h>

// __attribute__((optimize("O0")))
// __attribute__((target("fma")))
PetscErrorCode MatMult_SeqBAIJ_4_FMA(Mat A, Vec xx, Vec zz) {
  Mat_SeqBAIJ       *a      = (Mat_SeqBAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *zarray, *z;
  PetscInt           mbs, i, start, end, n;
  PetscBool          usecprow = a->compressedrow.use;
  const PetscInt    *iiptr, *jptr, *ridx = NULL;
  const MatScalar   *vbase;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx,  &x));
  PetscCall(VecGetArrayWrite(zz, &zarray));

  if (usecprow) {
    mbs   = a->compressedrow.nrows;
    iiptr = a->compressedrow.i;
    ridx  = a->compressedrow.rindex;
    PetscCall(PetscArrayzero(zarray, 4 * a->mbs));
  } else {
    mbs   = a->mbs;
    iiptr = a->i;
  }
  jptr = a->j;

  // PetscInt nblocks = iiptr[mbs];

  for (i = 0; i < mbs; i++) {
    start = iiptr[i];
    end   = iiptr[i+1];
    n     = end - start;

    // PetscCheck(start >= 0 && end >= start && end <= nblocks,
    //            PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
    //            "Invalid block row range [%d,%d) for row %d (nblocks=%d, mbs=%d)",
    //            start, end, i, nblocks, mbs);

    vbase = a->a + 16 * start;

    // Accumulateurs uniques (1 par ligne du bloc)
    PetscScalar sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (PetscInt j = 0; j < n; j++) {
      PetscInt colblock = jptr[start + j];

      // PetscCheck(colblock >= 0 && colblock < nblocks,
      //            PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      //            "Invalid colblock %d at row %d, j=%d (nblocks=%d)",
      //            colblock, i, j, nblocks);

      const PetscScalar *xb = x + 4 * colblock;
      const MatScalar   *v = vbase + 16 * j;

      sum0 = __builtin_fma(v[0],  xb[0], sum0);
      sum1 = __builtin_fma(v[1],  xb[0], sum1);
      sum2 = __builtin_fma(v[2],  xb[0], sum2);
      sum3 = __builtin_fma(v[3],  xb[0], sum3);

      sum0 = __builtin_fma(v[4],  xb[1], sum0);
      sum1 = __builtin_fma(v[5],  xb[1], sum1);
      sum2 = __builtin_fma(v[6],  xb[1], sum2);
      sum3 = __builtin_fma(v[7],  xb[1], sum3);

      sum0 = __builtin_fma(v[8],  xb[2], sum0);
      sum1 = __builtin_fma(v[9],  xb[2], sum1);
      sum2 = __builtin_fma(v[10], xb[2], sum2);
      sum3 = __builtin_fma(v[11], xb[2], sum3);

      sum0 = __builtin_fma(v[12], xb[3], sum0);
      sum1 = __builtin_fma(v[13], xb[3], sum1);
      sum2 = __builtin_fma(v[14], xb[3], sum2);
      sum3 = __builtin_fma(v[15], xb[3], sum3);

      // PetscCheck(v >= a->a && v + 15 < a->a + 16*nblocks,
      //            PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      //            "v out-of-range for block %d on row %d", j, i);
    }

    z = usecprow
      ? zarray + 4 * ridx[i]
      : zarray + 4 * i;
    z[0] = sum0; z[1] = sum1; z[2] = sum2; z[3] = sum3;
  }

  PetscCall(VecRestoreArrayRead(xx,  &x));
  PetscCall(VecRestoreArrayWrite(zz, &zarray));
  PetscCall(PetscLogFlops(32.0 * a->nz - 4.0 * a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}
