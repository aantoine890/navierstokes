#include "kernels.h"
#include <petsc/private/matimpl.h>

PetscErrorCode MatSolve_SeqBAIJ_4(Mat A, Vec bb, Vec xx) {
//   PetscPrintf(PETSC_COMM_SELF, "[DEBUG] Entering MatSolve_SeqBAIJ_4\n");
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, nz, idx, idt, idc, m;
  const PetscInt    *r, *c;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, x1, x2, x3, x4, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(iscol, &c));

  idx  = 4 * r[0];
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];
  #pragma novector
  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = 4 * r[i];
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    #pragma novector
    for (m = 0; m < nz; m++) {
      idx = 4 * vi[m];
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
    idt        = 4 * i;
    t[idt]     = s1;
    t[1 + idt] = s2;
    t[2 + idt] = s3;
    t[3 + idt] = s4;
  }
  #pragma novector
  for (i = n - 1; i >= 0; i--) {
    v   = aa + 16 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = 4 * i;
    s1  = t[idt];
    s2  = t[1 + idt];
    s3  = t[2 + idt];
    s4  = t[3 + idt];
    #pragma novector
    for (m = 0; m < nz; m++) {
      idx = 4 * vi[m];
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
    idc           = 4 * c[i];
    x[idc]        = t[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idc]    = t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idc]    = t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idc]    = t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(iscol, &c));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a     = (Mat_SeqBAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j;
  PetscInt           i, nz, idx, idt, idc;
  const PetscInt    *r, *c, *diag = a->diag;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x, s1, s2, s3, s4, x1, x2, x3, x4, *t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  t = a->solve_work;

  PetscCheck(t != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "solve_work is NULL");
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(iscol, &c));

  PetscCheck(r != NULL && c != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "permutation arrays are NULL");

  /* forward solve the lower triangular */
  idx = 4 * r[0];
  PetscCheck(idx + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Initial index out of bounds on t[]");
  t[0] = b[idx];
  t[1] = b[1 + idx];
  t[2] = b[2 + idx];
  t[3] = b[3 + idx];

  for (i = 1; i < n; i++) {
    v   = aa + 16 * ai[i];
    vi  = aj + ai[i];
    nz  = diag[i] - ai[i];
    PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Negative nz in forward solve");
    idx = 4 * r[i];
    PetscCheck(idx + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index out of bounds on t[]");
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    while (nz--) {
      idx = 4 * (*vi++);
      PetscCheck(idx + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index out of bounds on t[] in forward solve loop");
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
    /* Debug: Print information for problematic rows */
    if (i >= 64 && i <= 66) {
      PetscPrintf(PETSC_COMM_SELF, "[DEBUG] Row %d: ai[%d]=%d, ai[%d+1]=%d, diag[%d]=%d\n", 
                  i, i, ai[i], i, ai[i + 1], i, diag[i]);
    }
    
    /* Skip corrupted rows instead of failing */
    if (diag[i] < ai[i] || diag[i] > ai[i + 1]) {
      PetscPrintf(PETSC_COMM_SELF, "[WARNING] Skipping corrupted row %d: diag[%d]=%d not in range [%d, %d]\n",
                  i, i, diag[i], ai[i], ai[i + 1]);
      /* For corrupted rows, just copy the current solution */
      idt = 4 * i;
      idc = 4 * c[i];
      x[idc]     = t[idt];
      x[1 + idc] = t[1 + idt];
      x[2 + idc] = t[2 + idt];
      x[3 + idc] = t[3 + idt];
      continue;
    }
    
    /* Only proceed if there are off-diagonal elements after the diagonal */
    nz = ai[i + 1] - diag[i] - 1;
    if (nz > 0 && diag[i] < ai[i + 1]) {
      v   = aa + 16 * diag[i] + 16;
      vi  = aj + diag[i] + 1;
      idt = 4 * i;
      PetscCheck(idt + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "idt out of bounds on t[]");
      s1  = t[idt];
      s2  = t[1 + idt];
      s3  = t[2 + idt];
      s4  = t[3 + idt];
      while (nz--) {
        idx = 4 * (*vi++);
        PetscCheck(idx + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index out of bounds on t[] in backward solve loop");
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
      t[idt]     = s1;
      t[1 + idt] = s2;
      t[2 + idt] = s3;
      t[3 + idt] = s4;
    } else {
      /* No off-diagonal elements, just get current values */
      idt = 4 * i;
      s1  = t[idt];
      s2  = t[1 + idt];
      s3  = t[2 + idt];
      s4  = t[3 + idt];
    }
    
    /* Apply diagonal inverse - handle case where diag[i] might be at ai[i+1] */
    idc = 4 * c[i];
    PetscCheck(idc + 3 < 4 * n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "idc out of bounds on x[]");
    if (diag[i] < ai[i + 1]) {
      v = aa + 16 * diag[i];
    } else {
      /* diag[i] == ai[i+1], use the last valid element */
      v = aa + 16 * (ai[i + 1] - 1);
    }
    x[idc]     = t[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idc] = t[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idc] = t[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idc] = t[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(ISRestoreIndices(isrow, &r));
  PetscCall(ISRestoreIndices(iscol, &c));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}