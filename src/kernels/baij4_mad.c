
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "kernels.h"
#include <petsc/private/matimpl.h>

// __attribute__((optimize("O3")))
// __attribute__((optimize("fp-contract=on"))) // Pour activer les FMA
// __attribute__((target("fma,avx2")))
// __attribute__((optimize("fp-contract=off")))
// __attribute__((target("no-avx,no-sse,no-fma")))

PetscErrorCode MatMult_SeqBAIJ_4(Mat A, Vec xx, Vec zz) {
  Mat_SeqBAIJ       *a      = (Mat_SeqBAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *zarray; 
  PetscInt           mbs, i, start, end, n;
  PetscBool          usecprow = a->compressedrow.use;
  const PetscInt    *iiptr, *jptr, *ridx = NULL;
  const MatScalar   *vbase;
  
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx,  &x));
  PetscCall(VecGetArrayWrite(zz, &zarray));

  // Sélection des pointeurs selon compressedrow ou pas
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

  //PetscInt nblocks = iiptr[mbs];

  // Boucle sur les blocs de lignes
  #pragma novector
  for (i = 0; i < mbs; i++) {
    // bornes des blocs sur cette ligne
    start = iiptr[i];
    end   = iiptr[i+1];
    n     = end - start;

    // PetscCheck(start >= 0 && end >= start && end <= nblocks,
    //            PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
    //            "Invalid block row range [%d,%d) for row %d (nblocks=%d, mbs=%d)",
    //            start, end, i, nblocks, mbs);
    
    // pointeur de base sur les valeurs du premier bloc
    vbase = a->a + 16 * start;

    // accumulations
    PetscScalar sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    // boucle sur chaque bloc
    #pragma novector
    for (PetscInt j = 0; j < n; j++) {
      PetscInt colblock = jptr[start + j];

      // PetscCheck(colblock >= 0 && colblock < nblocks,
      //            PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      //            "Invalid colblock %d at row %d, j=%d (nblocks=%d)",
      //            colblock, i, j, nblocks);

      const PetscScalar *xb = x + 4 * colblock;
      const MatScalar   *v = vbase + 16 * j;


      sum0 += v[0]*xb[0] + v[4]*xb[1] + v[8]*xb[2]  + v[12]*xb[3];
      sum1 += v[1]*xb[0] + v[5]*xb[1] + v[9]*xb[2]  + v[13]*xb[3];
      sum2 += v[2]*xb[0] + v[6]*xb[1] + v[10]*xb[2] + v[14]*xb[3];
      sum3 += v[3]*xb[0] + v[7]*xb[1] + v[11]*xb[2] + v[15]*xb[3];

    //   PetscCheck(v >= a->a && v + 15 < a->a + 16*nblocks,
    //              PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
    //              "v out-of-range for block %d on row %d", j, i);
    }

    // écriture dans zarray
    PetscScalar *z = usecprow
      ? zarray + 4 * ridx[i]
      : zarray + 4 * i;
    z[0] = sum0; z[1] = sum1; z[2] = sum2; z[3] = sum3;
  }

  PetscCall(VecRestoreArrayRead(xx,  &x));
  PetscCall(VecRestoreArrayWrite(zz, &zarray));
  PetscCall(PetscLogFlops(32.0 * a->nz - 4.0 * a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

