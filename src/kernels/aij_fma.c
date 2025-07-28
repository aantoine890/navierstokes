
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include "kernels.h"
#include <petsc/private/matimpl.h>
#include <math.h>  

// __attribute__((optimize("O0")))
// __attribute__((target("fma")))
PetscErrorCode MatMult_SeqAIJ_FMA(Mat A, Vec xx, Vec zz) {
  Mat_SeqAIJ        *a   = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa = a->a;
  const PetscInt    *aj = a->j, *ai = a->i;
  PetscInt           i, j;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArrayWrite(zz, &y));

  for (i = 0; i < A->rmap->n; i++) {
    PetscScalar sum = 0.0;
    for (j = ai[i]; j < ai[i+1]; j++) {
      sum = __builtin_fma(aa[j], x[aj[j]], sum);
    }
    y[i] = sum;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArrayWrite(zz, &y));
  PetscCall(PetscLogFlops(2.0 * a->nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}
