#include "kernels.h"

PetscErrorCode MatMult_SeqBAIJ_4_VariantSelector(Mat A, Vec xx, Vec zz) {
  PetscInt variant = 0;
  PetscOptionsGetInt(NULL, NULL, "-matmult_variant", &variant, NULL);
  switch (variant) {
    case 0: return MatMult_SeqAIJ(A, xx, zz);
    case 1: return MatMult_SeqAIJ_FMA   (A, xx, zz);
    case 2: return MatMult_SeqBAIJ_4    (A, xx, zz);
    case 3: return MatMult_SeqBAIJ_4_FMA(A, xx, zz);
    case 4: return MatMult_SeqBAIJ_4_AVX2(A, xx, zz);
    case 5: return MatMult_SeqBAIJ_8_AVX512(A, xx, zz);
    default: return MatMult_SeqAIJ(A, xx, zz);
  }
}
