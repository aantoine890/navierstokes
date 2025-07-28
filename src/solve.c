#include <petscksp.h>
#include <petscdmplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "integration.h"
#include "kernels.h"

#ifdef __cplusplus
#include <set>
#include <vector>
#else
#error "Ce code nécessite d'\u00eatre compil\u00e9 en C++ pour std::set/std::vector"
#endif

int (*tet)[4];
double (*coords)[3];
int *node_surface_tags = NULL;

void save_matrix(Mat M, const char *filename) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    MatView(M, viewer);
    PetscViewerDestroy(&viewer);
}

void save_matrix_mtx(Mat M, const char *filename) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(M, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}

typedef enum { ORDER_BLOCK_NODE, ORDER_BY_COMPONENT } DoFOrdering;

static inline void get_local_to_global(int nv, const int nodes[4], int local2global_v[12], int local2global_p[4], DoFOrdering ordering) {   
    if (ordering == ORDER_BLOCK_NODE) {
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = 4*nodes[i] + comp;
            local2global_p[i] = 4*nodes[i] + 3;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = nodes[i] + comp*nv;
            local2global_p[i] = nodes[i] + 3*nv;
        }
    }
}

int contains(int *array, int size, int value) {
    for (int i = 0; i < size; ++i)
        if (array[i] == value)
            return 1;
    return 0;
}

void read_mesh(const char *filename, int *nv, int *ne, int **boundary_nodes, int *nb_boundary_nodes) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open .msh file: %s\n", filename);
        exit(1);
    }
    
    char line[256];
    *nv = *ne = 0;
    coords = NULL;
    tet = NULL;
    
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "$Nodes", 6) == 0) {
            fscanf(f, "%d\n", nv);
            coords = (double (*)[3]) malloc((*nv) * sizeof(double[3]));
            *boundary_nodes = (int *) malloc((*nv) * sizeof(int));
            node_surface_tags = (int *) malloc((*nv) * sizeof(int));
            for (int i = 0; i < *nv; ++i) {
                node_surface_tags[i] = -1;
            }
            
            *nb_boundary_nodes = 0;
            for (int i = 0; i < *nv; ++i) {
                int dummy;
                double x, y, z;
                fscanf(f, "%d %lf %lf %lf\n", &dummy, &x, &y, &z);
                coords[i][0] = x;
                coords[i][1] = y;
                coords[i][2] = z;
            }
        }
        
        if (strncmp(line, "$Elements", 9) == 0) {
            int total_elem;
            fscanf(f, "%d\n", &total_elem);
            
            int tet_count = 0;
            long file_pos = ftell(f);
            for (int i = 0; i < total_elem; ++i) {
                int id, type, ntags;
                fscanf(f, "%d %d %d", &id, &type, &ntags);
                for (int t = 0; t < ntags; ++t) { 
                    int dummy; 
                    fscanf(f, "%d", &dummy); 
                }
                if (type == 4) { 
                    int dummy; 
                    for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); 
                    tet_count++; 
                }
                else fgets(line, sizeof(line), f);
            }
            
            *ne = tet_count;
            tet = (int (*)[4]) malloc((*ne) * sizeof(int[4]));
            
            fseek(f, file_pos, SEEK_SET);
            int count = 0;
            
            for (int i = 0; i < total_elem; ++i) {
                int id, type, ntags;
                fscanf(f, "%d %d %d", &id, &type, &ntags);
                int tags[10];
                for (int t = 0; t < ntags; ++t)
                    fscanf(f, "%d", &tags[t]);
                
                if (type == 4) {
                    int v[4];
                    for (int j = 0; j < 4; ++j) fscanf(f, "%d", &v[j]);
                    for (int j = 0; j < 4; ++j) tet[count][j] = v[j] - 1;
                    count++;
                }
                else if (type == 2 || type == 3) {
                    int surface_tag = -1;
                    for (int t = 0; t < ntags; ++t) {
                        if (tags[t] == 1 || tags[t] == 2 || tags[t] == 4 ||
                            tags[t] == 5 || tags[t] == 6 || tags[t] == 7) {
                            surface_tag = tags[t];
                            break;
                        }
                    }
                    
                    int nverts = (type == 2 ? 3 : 4);
                    int v[4];
                    for (int j = 0; j < nverts; ++j)
                        fscanf(f, "%d", &v[j]);
                    
                    if (surface_tag != -1) {
                        for (int j = 0; j < nverts; ++j) {
                            int node = v[j] - 1;
                            node_surface_tags[node] = surface_tag;
                            if (!contains(*boundary_nodes, *nb_boundary_nodes, node)) {
                                (*boundary_nodes)[*nb_boundary_nodes] = node;
                                (*nb_boundary_nodes)++;
                            }
                        }
                    }
                }
                else {
                    fgets(line, sizeof(line), f);
                }
            }
        }
    }
    fclose(f);
}

void assemble_matrix(Mat A, int nv, int ne, DoFOrdering ordering,
                    double dt, double delta, double Re, int Ndof_v, int Ndof_tot) {
    
    for (int k = 0; k < ne; ++k) {
        int local2global_v[12], local2global_p[4];
        get_local_to_global(nv, tet[k], local2global_v, local2global_p, ordering);

        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            a[i][0] = coords[gi][0];
            a[i][1] = coords[gi][1];
            a[i][2] = coords[gi][2];
        }

        double Uloc[3][4];
        for (int comp = 0; comp < 3; ++comp)
            for (int i = 0; i < 4; ++i)
                Uloc[comp][i] = 1.0; 

        double Mloc[12][12], A0loc[12][12], A1loc[12][12], A2loc[12][12], Bloc[4][12], Dloc[4][4];
        mass_matrix(a[0], a[1], a[2], a[3], Mloc);
        diffusion_matrix(a, Re, A0loc);
        convection_matrix1(a, Uloc, A1loc);
        double grad[4][3], M4[4][4];
        tet_gradients(a, grad);
        mass_matrix_tet(tet_volum(a[0], a[1], a[2], a[3]), M4);
        convection_matrix2(Uloc, grad, M4, A2loc);
        divergence_matrix(grad, tet_volum(a[0], a[1], a[2], a[3]), Bloc);
        pressure_stabilization_matrix(a, delta, Dloc);
        
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double block4x4[16] = {0};
                for (int alpha = 0; alpha < 3; ++alpha)
                    for (int beta = 0; beta < 3; ++beta) {
                        int ii = 3*i + alpha;
                        int jj = 3*j + beta;
                        block4x4[alpha*4 + beta] = (Mloc[ii][jj]/dt)+ A0loc[ii][jj] + A1loc[ii][jj] + A2loc[ii][jj];
                    }
                for (int alpha = 0; alpha < 3; ++alpha) {
                    int ii = 3*i + alpha;
                    block4x4[alpha*4 + 3] = Bloc[j][ii];
                }
                for (int beta = 0; beta < 3; ++beta) {
                    int jj = 3*j + beta;
                    block4x4[3*4 + beta] = -Bloc[i][jj];
                }
                block4x4[15] = Dloc[i][j];

                PetscInt row = tet[k][i];
                PetscInt col = tet[k][j];
                MatSetValuesBlocked(A, 1, &row, 1, &col, block4x4, ADD_VALUES);
            }
        }
    }
}

typedef struct {
    Mat A_real;
    PetscBool use_avx2;
    PetscBool use_naive;
} MatShellCtx;

typedef struct {
    Mat mat_fact;
    Mat prealloc_mat;
    PetscBool use_avx2;
    PetscBool use_naive;
} PCShellCtx;

static PetscLogEvent MY_STD_MULT, MY_AVX2_MULT, MY_AVX2_SOLVE, MY_STD_SOLVE;

static PetscErrorCode MatShellMult(Mat M, Vec x, Vec y) {
    MatShellCtx *ctx;
    PetscCall(MatShellGetContext(M, &ctx));

    if (ctx->use_naive) {
        PetscLogEventBegin(MY_STD_MULT, 0,0,0,0);
        PetscCall(MatMult_SeqBAIJ_4(ctx->A_real, x, y));
        PetscLogEventEnd(MY_STD_MULT, 0,0,0,0);
    } else if (ctx->use_avx2) {
        PetscLogEventBegin(MY_AVX2_MULT, 0,0,0,0);
        PetscCall(MatMult_SeqBAIJ_4_AVX2(ctx->A_real, x, y));
        PetscLogEventEnd(MY_AVX2_MULT, 0,0,0,0);
    } else {
        PetscCall(MatMult(ctx->A_real, x, y));
    }
    return 0;
}

PetscErrorCode PCApply_Shell(PC pc, Vec x, Vec y) {
    PCShellCtx *ctx;
    PetscCall(PCShellGetContext(pc, &ctx));
    
    if (ctx->use_naive) {
        PetscLogEventBegin(MY_STD_SOLVE, 0,0,0,0);
        PetscCall(MatSolve_SeqBAIJ_4(ctx->mat_fact, x, y));
        PetscLogEventEnd(MY_STD_SOLVE, 0,0,0,0);
    } else if (ctx->use_avx2) {
        PetscLogEventBegin(MY_AVX2_SOLVE, 0,0,0,0);
        PetscCall(MatSolve_SeqBAIJ_4_AVX2(ctx->mat_fact, x, y));
        PetscLogEventEnd(MY_AVX2_SOLVE, 0,0,0,0);
    } else {
        PetscCall(MatSolve(ctx->mat_fact, x, y));
    }

    return 0;
}

PetscErrorCode PCDestroy_Shell(PC pc) {
    PCShellCtx *ctx;
    PetscCall(PCShellGetContext(pc, &ctx));
    PetscCall(PetscFree(ctx));
    return 0;
}

int main(int argc, char **args) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, NULL, NULL); if (ierr) return ierr;
    PetscCall(PetscLogEventRegister("STD_Mult", MAT_CLASSID, &MY_STD_MULT));
    PetscCall(PetscLogEventRegister("AVX2_Mult", MAT_CLASSID, &MY_AVX2_MULT));
    PetscCall(PetscLogEventRegister("AVX2_Solve", MAT_CLASSID, &MY_AVX2_SOLVE));
    PetscCall(PetscLogEventRegister("STD_Solve", MAT_CLASSID, &MY_STD_SOLVE));

    char mshname[PETSC_MAX_PATH_LEN];
    PetscBool flg, use_avx2 = PETSC_FALSE, use_naive = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-naive", &use_naive, NULL);
    PetscOptionsGetBool(NULL, NULL, "-avx2", &use_avx2, NULL);
    PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

    char full_msh_path[PETSC_MAX_PATH_LEN];
    snprintf(full_msh_path, sizeof(full_msh_path),
             "/home/kempf/spmv/mesh/%s", mshname);

    int nv, ne, *boundary_nodes = NULL, nb_boundary_nodes = 0;
    read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);
    PetscPrintf(PETSC_COMM_WORLD, "Parsed %d vertices, %d tetrahedra\n", nv, ne);
    // PetscPrintf(PETSC_COMM_WORLD, "Found %d boundary nodes\n", nb_boundary_nodes);
    // PetscPrintf(PETSC_COMM_WORLD, "Boundary nodes: ");
    // for (int i = 0; i < nb_boundary_nodes; ++i) {
    //     PetscPrintf(PETSC_COMM_WORLD, "%d ", boundary_nodes[i]);
    // }
    // PetscPrintf(PETSC_COMM_WORLD, "\n");

    double dt = 0.1, delta = 0.1, Re = 1.0;
    int Ndof_v = 3 * nv,
        Ndof_p = nv,
        Ndof_tot = Ndof_v + Ndof_p;

    Mat A = NULL, A_real = NULL;
    Vec rhs, x, r;
    KSP ksp;
    PC pc;

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A_real));
    PetscCall(MatSetSizes(A_real, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot));
    PetscCall(MatSetType(A_real, MATSEQBAIJ));
    PetscCall(MatSetBlockSize(A_real, 4));
    PetscCall(MatSetFromOptions(A_real));
    PetscCall(MatSetUp(A_real));

    assemble_matrix(A_real, nv, ne, ORDER_BLOCK_NODE, dt, delta, Re, Ndof_v, Ndof_tot);
    PetscCall(MatAssemblyBegin(A_real, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_real, MAT_FINAL_ASSEMBLY));
    
    PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    PetscScalar *values = (PetscScalar*)malloc(nb_boundary_nodes * 3 * sizeof(PetscScalar));
    
    double surface2_velocity[3] = {1.0, 0.0, 0.0};
    
    PetscCall(MatCreateVecs(A_real, NULL, &rhs));
    PetscCall(VecSet(rhs, 0.0));
    
    int count = 0;
    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];
        
        if (node_surface_tags[node] == 2) {
            rows[count] = 4*node + 0;
            values[count] = surface2_velocity[0];
            count++;
            
            rows[count] = 4*node + 1;
            values[count] = surface2_velocity[1];
            count++;
            
            rows[count] = 4*node + 2;
            values[count] = surface2_velocity[2];
            count++;
        } else {
            rows[count] = 4*node + 0;
            values[count] = 0.0;
            count++;
            
            rows[count] = 4*node + 1;
            values[count] = 0.0;
            count++;
            
            rows[count] = 4*node + 2;
            values[count] = 0.0;
            count++;
        }
    }
    
    PetscCall(VecSetValues(rhs, count, rows, values, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyEnd(rhs));

    PetscCall(MatZeroRows(A_real, count, rows, 1, rhs, NULL));
    
    free(rows);
    free(values);
    
    MatShellCtx *ctx;
    PetscCall(PetscNew(&ctx));
    ctx->A_real = A_real;
    ctx->use_avx2 = use_avx2;
    ctx->use_naive = use_naive;
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, Ndof_tot, Ndof_tot, Ndof_tot, Ndof_tot, ctx, &A));
    PetscCall(MatShellSetOperation(A, MATOP_MULT, (void(*)(void))MatShellMult));
    PetscCall(MatSetFromOptions(A));

    PetscCall(VecDuplicate(rhs, &x));
    PetscCall(VecDuplicate(rhs, &r));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A_real));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-12, PETSC_DEFAULT, 600));
    PetscCall(KSPGMRESSetRestart(ksp, 30));
    
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCILU));
    PetscCall(PCFactorSetLevels(pc, 0));
    
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));

    if (use_naive || use_avx2) {
        Mat matLU;
        PetscCall(PCFactorGetMatrix(pc, &matLU));
        PetscCall(PetscObjectReference((PetscObject)matLU));

        PCShellCtx *pc_ctx;
        PetscCall(PetscNew(&pc_ctx));
        pc_ctx->mat_fact = matLU;
        pc_ctx->use_naive = use_naive;
        pc_ctx->use_avx2 = use_avx2;

        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetContext(pc, pc_ctx));
        PetscCall(PCShellSetApply(pc, PCApply_Shell));
        PetscCall(PCShellSetDestroy(pc, PCDestroy_Shell));

        if (use_naive) {
            PetscCall(MatSetOperation(matLU, MATOP_SOLVE,
                      (void (*)(void))MatSolve_SeqBAIJ_4));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[INFO] Using NAIVE MatSolve (natural ordering)\n"));
        } else {
            PetscCall(MatSetOperation(matLU, MATOP_SOLVE,
                      (void (*)(void))MatSolve_SeqBAIJ_4_AVX2));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[INFO] Using AVX2 MatSolve (natural ordering)\n"));
        }
    }

    Mat A_export;
    PetscCall(MatConvert(A_real, MATSEQAIJ, MAT_INITIAL_MATRIX, &A_export));

    PetscReal norm_frobenius;
    PetscCall(MatNorm(A_export, NORM_FROBENIUS, &norm_frobenius));
    PetscPrintf(PETSC_COMM_WORLD, "Frobenius norm of A_export = %g\n", (double)norm_frobenius);

    PetscViewer viewer_mtx;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "A_real.mtx", &viewer_mtx));
    PetscCall(PetscViewerPushFormat(viewer_mtx, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(MatView(A_export, viewer_mtx));
    PetscCall(PetscViewerDestroy(&viewer_mtx));
    PetscCall(MatDestroy(&A_export));

    PetscViewer viewer_rhs;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "rhs.data", &viewer_rhs));
    PetscCall(PetscViewerPushFormat(viewer_rhs, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(VecView(rhs, viewer_rhs));
    PetscCall(PetscViewerDestroy(&viewer_rhs));

    PetscCall(KSPSetUp(ksp));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, rhs, x));

    PetscViewer viewer_x;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution_x.data", &viewer_x));
    PetscCall(PetscViewerPushFormat(viewer_x, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(VecView(x, viewer_x));
    PetscCall(PetscViewerDestroy(&viewer_x));

    KSPConvergedReason reason;
    PetscInt its;
    PetscReal res;
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &res));
    PetscPrintf(PETSC_COMM_WORLD, "Convergence reason = %d\n", reason);
    PetscPrintf(PETSC_COMM_WORLD, "Iterations = %d\n", its);
    PetscPrintf(PETSC_COMM_WORLD, "||r||_2 (KSP) = %g\n", (double)res);

    PetscCall(MatMult(A, x, r));
    PetscCall(VecAXPY(r, -1.0, rhs));
    PetscCall(VecNorm(r, NORM_2, &res));
    PetscPrintf(PETSC_COMM_WORLD, "||Ax - b||_2 = %g\n", (double)res);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "solution_x.bin",
                                   FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscPrintf(PETSC_COMM_WORLD, "PETSc scalar size: %lu bytes\n", sizeof(PetscScalar));

    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&rhs));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&A_real));

    free(boundary_nodes);
    PetscCall(PetscFinalize());
    return 0;
}