#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "kernels.h"

static double diff_time(const struct timespec *start,
                        const struct timespec *end) {
  return (end->tv_sec  - start->tv_sec)
       + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char **argv) {
  Mat        A, M_var;
  Vec        x, y;
  PetscInt   variant     = 0;
  PetscInt   matrix_id   = -1;
  PetscInt   niter       = 1;
  char       matrix_root[PETSC_MAX_PATH_LEN] = "";
  char       output_file[PETSC_MAX_PATH_LEN] = "benchmark.csv";
  char       result_file[PETSC_MAX_PATH_LEN] = "result_vector.bin";
  PetscBool  use_file    = PETSC_FALSE;
  FILE      *out;
  double     best_avg, exec_time;
  PetscInt   n;
  PetscReal  nnz;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscOptionsBegin(PETSC_COMM_SELF, NULL, "Benchmark options", NULL);
    PetscOptionsGetInt   (NULL, NULL, "-matrix_id",        &matrix_id, NULL);
    PetscOptionsGetInt   (NULL, NULL, "-matmult_variant",  &variant,   NULL);
    PetscOptionsGetInt   (NULL, NULL, "-niter",            &niter,     NULL);
    PetscOptionsGetString(NULL, NULL, "-matrix_file",      matrix_root,
                          sizeof(matrix_root), &use_file);
    PetscOptionsGetString(NULL, NULL, "-output_file",      output_file,
                          sizeof(output_file), NULL);
    PetscOptionsGetString(NULL, NULL, "-result_file",      result_file,
                          sizeof(result_file), NULL);
  PetscOptionsEnd();

  if (!use_file || matrix_root[0] == '\0' || matrix_id < 1) {
    PetscPrintf(PETSC_COMM_SELF, "Error: you must specify -matrix_file ROOT and -matrix_id >= 1\n");
    return 1;
  }

  char matrix_filename[PETSC_MAX_PATH_LEN];
  const char *fmt;
  if (variant < 2) fmt = "/home/users/u0001668/spmv/mat2/matrix_aij_stokes_%s.bin";
  else if (variant < 5) fmt = "/home/users/u0001668/spmv/mat2/matrix_baij4_stokes_%s.bin";
  else fmt = "/home/users/u0001668/spmv/mat/matrix_%s_baij8.bin";
  snprintf(matrix_filename, sizeof(matrix_filename), fmt, matrix_root);

  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "[DEBUG] Loading matrix from %s\n", matrix_filename));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, matrix_filename, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  if (variant < 2) {
    PetscCall(MatSetType(A, MATSEQAIJ));
  } else {
    PetscCall(MatSetType(A, MATSEQBAIJ));
    PetscCall(MatSetBlockSize(A, (variant < 5) ? 4 : 8));
  }
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt nrows;
  MatGetSize(A, &nrows, NULL);
  double sum = 0;
  for (PetscInt i = 0; i < PetscMin(nrows, 10); ++i) {
      PetscScalar val = 0.0;
      MatGetValue(A, i, i, &val);
      printf("A[%d,%d] = %e\n", (int)i, (int)i, (double)val);
      sum += fabs((double)val);
  }
  printf("Diag sum (10 premiers) = %e\n", sum);

  PetscCall(MatGetSize(A, &n, NULL));
  {
    MatInfo info;
    PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
    nnz = info.nz_used;
  }

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(x, 1.0));

  PetscCall(MatConvert(A, (variant < 2 ? MATSEQAIJ : MATSEQBAIJ), MAT_INITIAL_MATRIX, &M_var));
  PetscCall(MatDestroy(&A));

  PetscInt bs;
  PetscCall(MatGetBlockSize(M_var, &bs));
  //PetscPrintf(PETSC_COMM_SELF, "[DEBUG] Loaded matrix block size = %d\n", bs);

  if ((variant >= 2 && variant < 5 && bs != 4) || (variant == 5 && bs != 8)) {
    PetscPrintf(PETSC_COMM_SELF, "Error: block size %d incompatible with variant %d\n", bs, variant);
    PetscFinalize();
    return 1;
  }

  out = fopen(output_file, "w");
  if (!out) {
    PetscPrintf(PETSC_COMM_SELF, "Error: cannot open %s\n", output_file);
    PetscFinalize();
    return 1;
  }
  fprintf(out, "matrix\tvariant\tsize\tnnz\texec_time\n");

  switch (variant) {
    case 0: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqAIJ); break;
    case 1: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqAIJ_FMA); break;
    case 2: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqBAIJ_4); break;
    case 3: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqBAIJ_4_FMA); break;
    case 4: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqBAIJ_4_AVX2); break;
    case 5: MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqBAIJ_8_AVX512); break;
    default:
      PetscPrintf(PETSC_COMM_SELF, "Unknown variant %d, using AIJ default\n", variant);
      MatSetOperation(M_var, MATOP_MULT, (void(*)(void))MatMult_SeqAIJ);
  }

  MatMult(M_var, x, y);  // Warm-up

  best_avg = 1e30;
  for (int run = 0; run < 5; ++run) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    for (PetscInt i = 0; i < niter; ++i) MatMult(M_var, x, y);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    double total = diff_time(&t0, &t1);
    double avg = total / (double)niter;
    if (avg < best_avg) best_avg = avg;
  }
  exec_time = best_avg;

  fprintf(out, "%d\t%d\t%lld\t%.0f\t%.9f\n", matrix_id, (int)variant, (long long)n, nnz, exec_time);
  fclose(out);


  PetscScalar *y_arr;
  PetscInt local_size;
  PetscCall(VecGetArray(y, &y_arr));
  PetscCall(VecGetLocalSize(y, &local_size));

  double min_val = y_arr[0], max_val = y_arr[0];
  for (PetscInt i = 1; i < local_size; ++i) {
      if (y_arr[i] < min_val) min_val = y_arr[i];
      if (y_arr[i] > max_val) max_val = y_arr[i];
  }
  PetscPrintf(PETSC_COMM_SELF, "Stat: min(y)=%e, max(y)=%e\n", min_val, max_val);

  double sum2 = 0;
  for (PetscInt i = 0; i < local_size; ++i) sum2 += y_arr[i] * y_arr[i];
  PetscPrintf(PETSC_COMM_SELF, "Stat: ||y||_2 = %e\n", sqrt(sum2));

  int has_nan = 0;
  for (PetscInt i = 0; i < local_size; ++i) {
    if (fabs((double)y_arr[i]) > 1e+200) {
      PetscPrintf(PETSC_COMM_SELF, "⚠️  Très grosse valeur dans y[%lld] = %e\n", (long long)i, (double)y_arr[i]);
    }
    if (isnan((double)y_arr[i]) || isinf((double)y_arr[i])) {
      PetscPrintf(PETSC_COMM_SELF, "❌ Result vector contains NaN or Inf at index %lld: %e\n", (long long)i, (double)y_arr[i]);
      has_nan = 1;
      break;
    }
  }
  PetscCall(VecRestoreArray(y, &y_arr));

  if (has_nan) {
    PetscPrintf(PETSC_COMM_SELF, "❌ Aborting: result vector contains invalid values (NaN/Inf).\n");
    VecDestroy(&x);
    VecDestroy(&y);
    MatDestroy(&M_var);
    PetscFinalize();
    return 1;
  }

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, result_file, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(y, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscPrintf(PETSC_COMM_SELF, "Result vector written to %s\n", result_file);

  VecDestroy(&x);
  VecDestroy(&y);
  MatDestroy(&M_var);
  PetscFinalize();
  return 0;
}