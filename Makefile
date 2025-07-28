# PETSc configuration
PETSC_DIR   = /home/kempf/petsc-3.23.2
PETSC_ARCH  = arch-linux-r-intel

CC          = mpiicpx

CFLAGS      = -O3 -qopenmp -mfma -mavx2 -ffp-contract=on \
              -I$(PETSC_DIR)/include \
              -I$(PETSC_DIR)/$(PETSC_ARCH)/include \
              -Isrc/include \
              -Isrc/kernels

CFLAGS2 		= -O3 -no-fma -ffp-contract=off \
							-mno-sse \
							-I$(PETSC_DIR)/include \
							-I$(PETSC_DIR)/$(PETSC_ARCH)/include \
							-Isrc/include \
							-Isrc/kernels

LDFLAGS     = -Xlinker -rpath=$(PETSC_DIR)/$(PETSC_ARCH)/lib \
              -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lm

# Cibles utiles
TARGETS = solve_newton spmvb gmres

# --- SPMV BENCHMARK ---

SPMV_SRC_COMMON = \
  src/benchmark_spmv.c \
  src/integration.c \
  src/kernels/aij_fma.c \
  src/kernels/baij4_fma.c \
  src/kernels/baij4_avx2.c

SPMV_SRC_MAD = \
  src/kernels/aij_mad.c \
  src/kernels/baij4_mad.c

SPMV_OBJ_COMMON = $(SPMV_SRC_COMMON:.c=.o)
SPMV_OBJ_MAD    = $(SPMV_SRC_MAD:.c=.o)

spmvb: $(SPMV_OBJ_COMMON) $(SPMV_OBJ_MAD)
	$(CC) $^ -o $@ $(LDFLAGS)

# Compilation avec CFLAGS (optimisé AVX2)
$(SPMV_OBJ_COMMON): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation avec CFLAGS2 (FP-contract off)
$(SPMV_OBJ_MAD): %.o: %.c
	$(CC) $(CFLAGS2) -c $< -o $@

# --- SOLVEUR NEWTON ---

SOLVE_NEWTON_SRC = \
  src/solve_newton.c \
  src/integration.c

solve_newton: $(SOLVE_NEWTON_SRC)
	$(CC) $(CFLAGS) $(SOLVE_NEWTON_SRC) -o $@ $(LDFLAGS)

# --- GMRES SSTEP SHELL ---

GMRES_SRC = \
  src/sstepgmres.c \
  src/integration.c \
  src/kernels/baij4_avx2.c 

gmres: $(GMRES_SRC)
	$(CC) $(CFLAGS) $(GMRES_SRC) -o $@ $(LDFLAGS)

# --- Nettoyage ---
clean:
	rm -f $(TARGETS) *.o src/*.o