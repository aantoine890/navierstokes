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

// Globales dynamiques
int (*tet)[4];
double (*coords)[3];

// --------- Utils Save PETSc Mat ----------
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

// --------- Mapping ----------
typedef enum { ORDER_BLOCK_NODE, ORDER_BY_COMPONENT } DoFOrdering;

static inline void
get_local_to_global(int nv, const int nodes[4], int local2global_v[12], int local2global_p[4], DoFOrdering ordering) {   

    if (ordering == ORDER_BLOCK_NODE) {
        // BAIJ4 - bloc de 4 par noeud (u_x,u_y,u_z,p)^0, (u_x,u_y,u_z,p)^1, ...
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = 4*nodes[i] + comp; // u_x, u_y, u_z
            local2global_p[i] = 4*nodes[i] + 3; // p
        }
    } else {
        // AIJ - toutes les u_x, puis u_y, puis u_z, puis p (par composante)
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = nodes[i] + comp*nv;  // vx, vy, vz
            local2global_p[i] = nodes[i] + 3*nv;                 // p
        }
    }
}

// --------- Mesh Parsing ---------
int contains(int *array, int size, int value) {
    for (int i = 0; i < size; ++i)
        if (array[i] == value)
            return 1;
    return 0;
}

// void read_mesh(const char *filename, int *nv, int *ne, int **boundary_nodes, int *nb_boundary_nodes) {
//     FILE *f = fopen(filename, "r");
//     if (!f) {
//         fprintf(stderr, "Cannot open .msh file: %s\n", filename);
//         exit(1);
//     }

//     char line[256];
//     *nv = *ne = 0;
//     coords = NULL;
//     tet = NULL;

//     while (fgets(line, sizeof(line), f)) {
//         if (strncmp(line, "$Nodes", 6) == 0) {
//             fscanf(f, "%d\n", nv);
//             coords = (double (*)[3]) malloc((*nv) * sizeof(double[3]));
//             *boundary_nodes = (int *) malloc((*nv) * sizeof(int)); 
//             *nb_boundary_nodes = 0;

//             for (int i = 0; i < *nv; ++i) {
//                 int dummy;
//                 double x, y, z;
//                 fscanf(f, "%d %lf %lf %lf\n", &dummy, &x, &y, &z);
//                 coords[i][0] = x;
//                 coords[i][1] = y;
//                 coords[i][2] = z;
//             }
//         }

//         if (strncmp(line, "$Elements", 9) == 0) {
//             int total_elem;
//             fscanf(f, "%d\n", &total_elem); 

//             // count tetrahedra
//             int tet_count = 0;
//             long file_pos = ftell(f);
//             for (int i = 0; i < total_elem; ++i) {
//                 int id, type, ntags;
//                 fscanf(f, "%d %d %d", &id, &type, &ntags);
//                 for (int t = 0; t < ntags; ++t) { int dummy; fscanf(f, "%d", &dummy); }
//                 if (type == 4) { int dummy; for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); tet_count++; }
//                 else if (type == 2) { int dummy; for (int j = 0; j < 3; ++j) fscanf(f, "%d", &dummy); }
//                 else if (type == 3) { int dummy; for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); }
//                 else fgets(line, sizeof(line), f);
//             }

//             *ne = tet_count;
//             tet = (int (*)[4]) malloc((*ne) * sizeof(int[4]));

//             // read the file
//             fseek(f, file_pos, SEEK_SET);
//             int count = 0;
//             for (int i = 0; i < total_elem; ++i) {
//                 int id, type, ntags;
//                 fscanf(f, "%d %d %d", &id, &type, &ntags);
//                 int tags[10];
//                 for (int t = 0; t < ntags; ++t)
//                     fscanf(f, "%d", &tags[t]);

//                 if (type == 4) {
//                     int v[4];
//                     for (int j = 0; j < 4; ++j) fscanf(f, "%d", &v[j]);
//                     for (int j = 0; j < 4; ++j) tet[count][j] = v[j] - 1;
//                     count++;
//                 }
//                 else if (type == 2 || type == 3) {
//                     int is_dirichlet = 0;
//                     for (int t = 0; t < ntags; ++t) {
//                         if (tags[t] == 1 || tags[t] == 2 || tags[t] == 4 ||
//                             tags[t] == 5 || tags[t] == 6 || tags[t] == 7) {
//                             is_dirichlet = 1;
//                             break;
//                         }
//                     }
//                     int nverts = (type == 2 ? 3 : 4);
//                     int v[4];
//                     for (int j = 0; j < nverts; ++j)
//                         fscanf(f, "%d", &v[j]);

//                     if (is_dirichlet) {
//                         for (int j = 0; j < nverts; ++j) {
//                             int node = v[j] - 1;
//                             if (!contains(*boundary_nodes, *nb_boundary_nodes, node)) {
//                                 (*boundary_nodes)[*nb_boundary_nodes] = node;
//                                 (*nb_boundary_nodes)++;
//                             }
//                         }
//                     }
//                 }
//                 else {
//                     fgets(line, sizeof(line), f); 
//                 }
//             }
//         }
//     }

//     fclose(f);
// }
int *node_surface_tags = NULL;  // Tableau global pour stocker le tag de chaque nœud

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
            
            // Initialiser le tableau des tags (-1 = nœud intérieur, >=1 = surface)
            node_surface_tags = (int *) malloc((*nv) * sizeof(int));
            for (int i = 0; i < *nv; ++i) {
                node_surface_tags[i] = -1;  // Par défaut : nœud intérieur
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
            
            // Compter tetrahedra
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
                else if (type == 2) { 
                    int dummy; 
                    for (int j = 0; j < 3; ++j) fscanf(f, "%d", &dummy); 
                }
                else if (type == 3) { 
                    int dummy; 
                    for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); 
                }
                else fgets(line, sizeof(line), f);
            }
            
            *ne = tet_count;
            tet = (int (*)[4]) malloc((*ne) * sizeof(int[4]));
            
            // Lire le fichier
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
                            
                            // Stocker le tag de surface pour ce nœud
                            node_surface_tags[node] = surface_tag;
                            
                            // Ajouter à la liste des nœuds de frontière si pas déjà présent
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

// --------- Matrix Assembly ---------
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

        if (k == 0) {  // Premier élément seulement
            printf("=== DEBUG ELEMENT 0 ===\n");
            printf("Coordinates:\n");
            for (int i = 0; i < 4; i++) {
                printf("Node %d (global %d): [%.6e, %.6e, %.6e]\n", 
                    i, tet[k][i], a[i][0], a[i][1], a[i][2]);
            }
            
            // Calculer et afficher les gradients
            double grad[4][3];
            tet_gradients(a, grad);
            printf("Gradients:\n");
            for (int i = 0; i < 4; i++) {
                printf("grad[%d] = [%.6e, %.6e, %.6e]\n", 
                    i, grad[i][0], grad[i][1], grad[i][2]);
            }
            
            // Volume
            double vol = tet_volum(a[0], a[1], a[2], a[3]);
            printf("Volume = %.6e\n", vol);
            printf("========================\n");
        }
            

        // --- Champ vitesse U constant ---
        double Uloc[3][4];
        for (int comp = 0; comp < 3; ++comp)
            for (int i = 0; i < 4; ++i)
                Uloc[comp][i] = 1.0; 

        // --- Champ vitesse U custom ---
        // double Uloc[3][4] = {
        //     {1.0, 2.0, 3.0, 4.0},      // Ux
        //     {0.0, -1.0, 0.5, 2.0},     // Uy
        //     {-2.0, 1.0, 0.0, -1.0}     // Uz
        // };

        // --- Appel de tes fonctions locales ---
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
        

        // ---------- Assemblage générique ------------
       if (ordering == ORDER_BY_COMPONENT) {
                // AIJ (par composante)
                
                // Bloc vitesse-vitesse : 12x12
                for (int i = 0; i < 12; ++i) {
                    for (int j = 0; j < 12; ++j) {
                        double val = (Mloc[i][j]/dt) + A0loc[i][j] + A1loc[i][j] + A2loc[i][j];
                        PetscInt row = local2global_v[i];
                        PetscInt col = local2global_v[j];
                        MatSetValue(A, row, col, val, ADD_VALUES);
                    }
                }

                // Bloc vitesse-pression : 12x4
                for (int i = 0; i < 12; ++i) {
                    for (int l = 0; l < 4; ++l) {
                        PetscInt row = local2global_v[i];
                        PetscInt col = local2global_p[l];
                        MatSetValue(A, row, col, Bloc[l][i], ADD_VALUES);
                    }
                }

                // Bloc pression-vitesse : 4x12
                for (int l = 0; l < 4; ++l) {
                    for (int j = 0; j < 12; ++j) {
                        PetscInt row = local2global_p[l];
                        PetscInt col = local2global_v[j];
                        MatSetValue(A, row, col, -Bloc[l][j], ADD_VALUES);
                    }
                }

                // Bloc pression-pression : 4x4
                for (int l = 0; l < 4; ++l) {
                    for (int k = 0; k < 4; ++k) {
                        PetscInt row = local2global_p[l];
                        PetscInt col = local2global_p[k];
                        MatSetValue(A, row, col, Dloc[l][k], ADD_VALUES);
                    }
                }

            } else {
                //printf("→ Assemblage BAIJ4\n");
                // BAIJ4 (bloc par noeud)
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        double block4x4[16] = {0};
                        // v-v
                        for (int alpha = 0; alpha < 3; ++alpha)
                            for (int beta = 0; beta < 3; ++beta) {
                                int ii = 3*i + alpha;
                                int jj = 3*j + beta;
                                block4x4[alpha*4 + beta] = (Mloc[ii][jj]/dt) + A0loc[ii][jj] + A1loc[ii][jj] + A2loc[ii][jj];
                                // block4x4[alpha*4 + beta] = (A1loc[ii][jj]);
                            }
                        // v-p
                        for (int alpha = 0; alpha < 3; ++alpha) {
                            int ii = 3*i + alpha;
                            block4x4[alpha*4 + 3] = Bloc[j][ii];
                        }
                        // p-v
                        for (int beta = 0; beta < 3; ++beta) {
                            int jj = 3*j + beta;
                            block4x4[3*4 + beta] = -Bloc[i][jj];
                        }
                        // p-p
                        block4x4[15] = Dloc[i][j];

                        PetscInt row = tet[k][i];
                        PetscInt col = tet[k][j];
                        MatSetValuesBlocked(A, 1, &row, 1, &col, block4x4, ADD_VALUES);
                    }
                }
        }
    }
}

// ------------------------------------------------------------------
//  Contexte du Shell pour stocker la matrice réelle + le flag AVX2
// ------------------------------------------------------------------
typedef struct {
  Mat        A_real;
  PetscBool  use_avx2;
  PetscBool  use_naive;
} MatShellCtx;

// ------------------------------------------------------------------
//  Contexte du Shell pour stocker la matrice LU factorisée
// -------------------------------------------------------------------
typedef struct {
    Mat mat_fact;      
    Mat prealloc_mat;  
    PetscBool use_avx2;
    PetscBool  use_naive;
} PCShellCtx;
// ------------------------------------------------------------------
//  Callback MatMult du Shell : appelle AVX2 ou MatMult standard
// ------------------------------------------------------------------

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

PetscErrorCode PCApply_Shell(PC pc, Vec x, Vec y)
{
    PCShellCtx *ctx;
    PetscCall(PCShellGetContext(pc, &ctx));
    
    // Debug
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, 
    //            "[DEBUG] Applying %s solver\n", 
    //            ctx->use_avx2 ? "AVX2" : (ctx->use_naive ? "Naive" : "Default")));

    // Choix de la routine en fonction des flags
    if (ctx->use_naive) {
        PetscLogEventBegin(MY_STD_SOLVE, 0,0,0,0);
        PetscCall(MatSolve_SeqBAIJ_4(ctx->mat_fact, x, y));
        PetscLogEventEnd(MY_STD_SOLVE, 0,0,0,0);
    } else if (ctx->use_avx2) {
        PetscLogEventBegin(MY_AVX2_SOLVE, 0,0,0,0);
        PetscCall(MatSolve_SeqBAIJ_4_AVX2(ctx->mat_fact, x, y));
        PetscLogEventEnd(MY_AVX2_SOLVE, 0,0,0,0);
    } else {
        PetscCall(MatSolve(ctx->mat_fact, x, y)); // fallback standard PETSc
    }

    return 0;
}

PetscErrorCode PCDestroy_Shell(PC pc) {
    PCShellCtx *ctx;
    PetscCall(PCShellGetContext(pc, &ctx));
    PetscCall(PetscFree(ctx));
    return 0;
}

// int main(int argc, char **args) {
//   PetscErrorCode ierr;
//   ierr = PetscInitialize(&argc, &args, NULL, NULL); if (ierr) return ierr;
//   PetscCall(PetscLogEventRegister("STD_Mult", MAT_CLASSID, &MY_STD_MULT));
//   PetscCall(PetscLogEventRegister("AVX2_Mult", MAT_CLASSID, &MY_AVX2_MULT));

//   // ----- Lecture des options -----
//   char        mshname[PETSC_MAX_PATH_LEN];
//   PetscBool   flg, use_baij = PETSC_FALSE, use_avx2 = PETSC_FALSE , use_naive = PETSC_FALSE;
//   PetscOptionsGetBool(NULL, NULL, "-naive", &use_naive, NULL);
//   PetscOptionsGetBool(NULL, NULL, "-avx2", &use_avx2, NULL);
//   PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
//   PetscOptionsGetBool(NULL, NULL, "-baij", &use_baij, NULL);
//   if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

//   char full_msh_path[PETSC_MAX_PATH_LEN];
//   snprintf(full_msh_path, sizeof(full_msh_path),
//            "/home/kempf/spmv/mesh/%s", mshname);

//   // ----- Lecture du maillage -----
//   int nv, ne, *boundary_nodes = NULL, nb_boundary_nodes = 0;
//   read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);
//   PetscPrintf(PETSC_COMM_WORLD, "Parsed %d vertices, %d tetrahedra\n", nv, ne);

//   // ----- Paramètres du problème -----
//   double dt = 0.1, delta = 0.1, Re = 1.0;
//   int Ndof_v  = 3 * nv,
//       Ndof_p  = nv,
//       Ndof_tot = Ndof_v + Ndof_p;

//   // Variables pour matrices/vecteurs
//   Mat   A = NULL, A_real = NULL;
//   Vec   rhs, x, r;
//   KSP   ksp;
//   PC    pc;

//   if (use_baij) {
//     // ----------------------------------------------------------
//     //  1) On crée et assemble la "vraie" MATSEQBAIJ dans A_real
//     // ----------------------------------------------------------
//     PetscCall(MatCreate(PETSC_COMM_WORLD, &A_real));
//     PetscCall(MatSetSizes(A_real, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot));
//     PetscCall(MatSetType(A_real, MATSEQBAIJ));
//     PetscCall(MatSetBlockSize(A_real, 4));
//     PetscCall(MatSetFromOptions(A_real));
//     PetscCall(MatSetUp(A_real));

//     assemble_matrix(A_real, nv, ne, ORDER_BLOCK_NODE,
//                     dt, delta, Re, Ndof_v, Ndof_tot);
//     PetscCall(MatAssemblyBegin(A_real, MAT_FINAL_ASSEMBLY));
//     PetscCall(MatAssemblyEnd( A_real, MAT_FINAL_ASSEMBLY));

//     // ----------------------------------------------------------
//     //  2) On applique les conditions de Dirichlet SUR A_real
//     // ----------------------------------------------------------
//     PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
//     int count = 0;
//     for (int i = 0; i < nb_boundary_nodes; ++i) {
//       int node = boundary_nodes[i];
//       rows[count++] = 4*node + 0;
//       rows[count++] = 4*node + 1;
//       rows[count++] = 4*node + 2;
//     }
//     PetscCall(MatCreateVecs(A_real, NULL, &rhs));
//     PetscCall(VecSet(rhs, 1.0));
//     PetscCall(MatZeroRows(A_real, count, rows, 1.0, rhs, rhs));
//     free(rows);

//     // ----------------------------------------------------------
//     //  3) On crée le Shell "A" qui wrappe A_real
//     // ----------------------------------------------------------
//     MatShellCtx *ctx;
//     PetscCall(PetscNew(&ctx));
//     ctx->A_real  = A_real;
//     ctx->use_avx2 = use_avx2;
//     ctx->use_naive = use_naive;

//     PetscCall(MatCreateShell(PETSC_COMM_WORLD,
//                              Ndof_tot, Ndof_tot,
//                              Ndof_tot, Ndof_tot,
//                              ctx, &A));
//     PetscCall(MatShellSetOperation(A,
//                  MATOP_MULT,
//                  (void(*)(void))MatShellMult));
//     PetscCall(MatSetFromOptions(A));

//   } else {
//     // ----------------------------------------------------------
//     //  Cas AIJ classique
//     // ----------------------------------------------------------
//     PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
//     PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot));
//     PetscCall(MatSetType(A, MATSEQAIJ));
//     PetscCall(MatSetFromOptions(A));
//     PetscCall(MatSetUp(A));

//     assemble_matrix(A, nv, ne, ORDER_BY_COMPONENT,
//                     dt, delta, Re, Ndof_v, Ndof_tot);
//     PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
//     PetscCall(MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY));

//     // Conditions de Dirichlet
//     PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
//     int count = 0;
//     for (int i = 0; i < nb_boundary_nodes; ++i) {
//       int node = boundary_nodes[i];
//       rows[count++] = node;
//       rows[count++] = node + nv;
//       rows[count++] = node + 2*nv;
//     }
//     PetscCall(MatCreateVecs(A, NULL, &rhs));
//     PetscCall(VecSet(rhs, 1.0));
//     PetscCall(MatZeroRows(A, count, rows, 1.0, rhs, rhs));
//     free(rows);
//   }

//   // ----------------------------------------------------------
//   //  Résolution KSP
//   // ----------------------------------------------------------
//   PetscCall(VecDuplicate(rhs, &x));
//   PetscCall(VecDuplicate(rhs, &r));

//   PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
//   if (use_baij) {
//     // opérateur = shell, préconditionneur = A_real
//     PetscCall(KSPSetOperators(ksp, A, A_real));
//   } else {
//     PetscCall(KSPSetOperators(ksp, A, A));
//   }
//   PetscCall(KSPSetType(ksp, KSPGMRES));
//   PetscCall(KSPGetPC(ksp, &pc));
//   PetscCall(PCSetType(pc, PCILU));
//   PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-10, PETSC_DEFAULT, 2000));
//   PetscCall(KSPSetFromOptions(ksp));

//   if (use_baij && use_avx2) {
//     Mat matLU;
//     PetscCall(PCFactorGetMatrix(pc, &matLU));
//     extern PetscErrorCode MatSolve_SeqBAIJ_4_AVX2(Mat, Vec, Vec);
//     PetscCall(MatSetOperation(matLU, MATOP_SOLVE,
//                                 (void (*)(void)) MatSolve_SeqBAIJ_4_AVX2));

//     PetscPrintf(PETSC_COMM_WORLD, "[INFO] AVX2 MatSolve activé.\n");
// }

//   PetscCall(KSPSolve(ksp, rhs, x));

//   // Infos de convergence
//   KSPConvergedReason reason;
//   PetscInt its;
//   PetscReal res;
//   PetscCall(KSPGetConvergedReason(ksp, &reason));
//   PetscCall(KSPGetIterationNumber(ksp, &its));
//   PetscCall(KSPGetResidualNorm(ksp, &res));
//   PetscPrintf(PETSC_COMM_WORLD, "Convergence reason = %d\n", reason);
//   PetscPrintf(PETSC_COMM_WORLD, "Iterations = %d\n", its);
//   PetscPrintf(PETSC_COMM_WORLD, "||r||_2 (KSP) = %g\n", (double)res);

//   // Vérif. résidu explicite
//   PetscCall(MatMult(A, x, r));
//   PetscCall(VecAXPY(r, -1.0, rhs));
//   PetscCall(VecNorm(r, NORM_2, &res));
//   PetscPrintf(PETSC_COMM_WORLD, "||Ax - b||_2 = %g\n", (double)res);

//   // Sauvegarde de la solution
//   PetscViewer viewer;
//   PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "solution_x.bin",
//                                   FILE_MODE_WRITE, &viewer));
//   PetscCall(VecView(x, viewer));
//   PetscCall(PetscViewerDestroy(&viewer));

//   // ----------------------------------------------------------------
//   //  Nettoyage
//   // ----------------------------------------------------------------
//   PetscCall(VecDestroy(&x));
//   PetscCall(VecDestroy(&r));
//   PetscCall(VecDestroy(&rhs));
//   PetscCall(KSPDestroy(&ksp));
//   if (use_baij) {
//     PetscCall(MatDestroy(&A));        // shell
//     PetscCall(MatDestroy(&A_real));   // vraie BAIJ
//   } else {
//     PetscCall(MatDestroy(&A));
//   }

//   free(boundary_nodes);
//   PetscCall(PetscFinalize());
//   return 0;
// }

int main(int argc, char **args) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, NULL, NULL); if (ierr) return ierr;
    PetscCall(PetscLogEventRegister("STD_Mult", MAT_CLASSID, &MY_STD_MULT));
    PetscCall(PetscLogEventRegister("AVX2_Mult", MAT_CLASSID, &MY_AVX2_MULT));
    PetscCall(PetscLogEventRegister("AVX2_Solve", MAT_CLASSID, &MY_AVX2_SOLVE));
    PetscCall(PetscLogEventRegister("STD_Solve", MAT_CLASSID, &MY_STD_SOLVE));

    // ----- Lecture des options -----
    char mshname[PETSC_MAX_PATH_LEN];
    PetscBool flg, use_baij = PETSC_FALSE, use_avx2 = PETSC_FALSE, use_naive = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-naive", &use_naive, NULL);
    PetscOptionsGetBool(NULL, NULL, "-avx2", &use_avx2, NULL);
    PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
    PetscOptionsGetBool(NULL, NULL, "-baij", &use_baij, NULL);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

    char full_msh_path[PETSC_MAX_PATH_LEN];
    snprintf(full_msh_path, sizeof(full_msh_path),
             "/home/kempf/spmv/mesh/%s", mshname);

    // ----- Lecture du maillage -----
    int nv, ne, *boundary_nodes = NULL, nb_boundary_nodes = 0;
    read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);

    // dump les nœuds de frontière
    PetscPrintf(PETSC_COMM_WORLD, "Parsed %d vertices, %d tetrahedra\n", nv, ne);

    // ----- Paramètres du problème -----
    double dt = 0.1, delta = 0.1, Re = 1.0;
    int Ndof_v = 3 * nv,
        Ndof_p = nv,
        Ndof_tot = Ndof_v + Ndof_p;

    // Variables pour matrices/vecteurs
    Mat A = NULL, A_real = NULL;
    Vec rhs, x, r;
    KSP ksp;
    PC pc;

    if (use_baij) {
        // Création et assemblage de la MATSEQBAIJ
        PetscCall(MatCreate(PETSC_COMM_WORLD, &A_real));
        PetscCall(MatSetSizes(A_real, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot));
        PetscCall(MatSetType(A_real, MATSEQBAIJ));
        PetscCall(MatSetBlockSize(A_real, 4));
        PetscCall(MatSetFromOptions(A_real));
        PetscCall(MatSetUp(A_real));
        assemble_matrix(A_real, nv, ne, ORDER_BLOCK_NODE, dt, delta, Re, Ndof_v, Ndof_tot);
        PetscCall(MatAssemblyBegin(A_real, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A_real, MAT_FINAL_ASSEMBLY));
        
        // Application des conditions de Dirichlet avec vitesse sur Surface 2
        PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
        PetscScalar *values = (PetscScalar*)malloc(nb_boundary_nodes * 3 * sizeof(PetscScalar));
        
        // Vitesse à imposer sur Surface 2
        double surface2_velocity[3] = {1.0, 0.0, 0.0};  // Modifiez selon vos besoins
        
        // Créer et initialiser le RHS
        PetscCall(MatCreateVecs(A_real, NULL, &rhs));
        PetscCall(VecSet(rhs, 0.0));
        
        int count = 0;
        for (int i = 0; i < nb_boundary_nodes; ++i) {
            int node = boundary_nodes[i];
            
            if (node_surface_tags[node] == 2) {
                // Surface 2 : imposer la vitesse définie
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
                // Autres surfaces : vitesse nulle
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
        
        // Appliquer les conditions aux limites
        PetscCall(MatZeroRows(A_real, count, rows, 1.0, rhs, NULL));
        PetscCall(VecSetValues(rhs, count, rows, values, INSERT_VALUES));
        PetscCall(VecAssemblyBegin(rhs));
        PetscCall(VecAssemblyEnd(rhs));
        
        free(rows);
        free(values);
        
        // Création du Shell "A" qui wrappe A_real
        MatShellCtx *ctx;
        PetscCall(PetscNew(&ctx));
        ctx->A_real = A_real;
        ctx->use_avx2 = use_avx2;
        ctx->use_naive = use_naive;
        PetscCall(MatCreateShell(PETSC_COMM_WORLD, Ndof_tot, Ndof_tot, Ndof_tot, Ndof_tot, ctx, &A));
        PetscCall(MatShellSetOperation(A, MATOP_MULT, (void(*)(void))MatShellMult));
        PetscCall(MatSetFromOptions(A));
    } else {
        // Cas AIJ classique
        PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
        PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot));
        PetscCall(MatSetType(A, MATSEQAIJ));
        PetscCall(MatSetFromOptions(A));
        PetscCall(MatSetUp(A));

        assemble_matrix(A, nv, ne, ORDER_BY_COMPONENT, dt, delta, Re, Ndof_v, Ndof_tot);
        PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

        // Conditions de Dirichlet
        PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
        int count = 0;
        for (int i = 0; i < nb_boundary_nodes; ++i) {
            int node = boundary_nodes[i];
            rows[count++] = node;
            rows[count++] = node + nv;
            rows[count++] = node + 2*nv;
        }
        PetscCall(MatCreateVecs(A, NULL, &rhs));
        PetscCall(VecSet(rhs, 0.0));
        PetscCall(MatZeroRows(A, count, rows, 1.0, rhs, rhs));
        free(rows);
    }

    // ----- Configuration du solveur -----
    PetscCall(VecDuplicate(rhs, &x));
    PetscCall(VecDuplicate(rhs, &r));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, use_baij ? A_real : A));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(ksp, 1e-12, 1e-12, PETSC_DEFAULT, 600));
    PetscCall(KSPGMRESSetRestart(ksp, 30)); // Nombre de vecteurs de Krylov
    
    // Configuration du préconditionneur
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCILU));
    PetscCall(PCFactorSetLevels(pc, 0));  // Niveau de remplissage pour ILU

    
    // Paramètres supplémentaires en ligne de commande
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));

    // ----- Remplacement de MatSolve pour BAIJ -----
    // Dans la configuration du solveur :
    if (use_baij && (use_naive || use_avx2)) {
        // Récupérer la matrice LU du PC existant
        Mat matLU;
        PetscCall(PCFactorGetMatrix(pc, &matLU));

        // Empêcher que PETSc détruise matLU lors de PCSetType(pc, PCSHELL)
        PetscCall(PetscObjectReference((PetscObject)matLU));

        // Créer le contexte shell
        PCShellCtx *pc_ctx;
        PetscCall(PetscNew(&pc_ctx));
        pc_ctx->mat_fact   = matLU;
        pc_ctx->use_naive  = use_naive;
        pc_ctx->use_avx2   = use_avx2;

        // Changer le PC en PCSHELL
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetContext(pc, pc_ctx));
        PetscCall(PCShellSetApply(pc, PCApply_Shell));
        PetscCall(PCShellSetDestroy(pc, PCDestroy_Shell));

        // Remplacer l'opération MatSolve sur matLU
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

    // calcule la norme de l amtrice frobenius
    PetscReal norm_frobenius;
    PetscCall(MatNorm(A_export, NORM_FROBENIUS, &norm_frobenius));
    PetscPrintf(PETSC_COMM_WORLD, "Frobenius norm of A_export = %g\n", (double)norm_frobenius);

    PetscViewer viewer_mtx;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "A_real.mtx", &viewer_mtx));
    PetscCall(PetscViewerPushFormat(viewer_mtx, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(MatView(A_export, viewer_mtx));
    PetscCall(PetscViewerDestroy(&viewer_mtx));
    PetscCall(MatDestroy(&A_export));

    //exporte le rhs au format .data
    PetscViewer viewer_rhs;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "rhs.data", &viewer_rhs));
    PetscCall(PetscViewerPushFormat(viewer_rhs, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(VecView(rhs, viewer_rhs));
    PetscCall(PetscViewerDestroy(&viewer_rhs));

    // Force la factorisation
    PetscCall(KSPSetUp(ksp));

    PetscCall(VecSet(x, 0.0)); // Initialisation de x à zéro

    // ----- Résolution du système -----
    PetscCall(KSPSolve(ksp, rhs, x));

    PetscViewer viewer_x;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution_x.data", &viewer_x));
    PetscCall(PetscViewerPushFormat(viewer_x, PETSC_VIEWER_ASCII_MATRIXMARKET));
    PetscCall(VecView(x, viewer_x));
    PetscCall(PetscViewerDestroy(&viewer_x));

    // Affichage des informations de convergence
    KSPConvergedReason reason;
    PetscInt its;
    PetscReal res;
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &res));
    PetscPrintf(PETSC_COMM_WORLD, "Convergence reason = %d\n", reason);
    PetscPrintf(PETSC_COMM_WORLD, "Iterations = %d\n", its);
    PetscPrintf(PETSC_COMM_WORLD, "||r||_2 (KSP) = %g\n", (double)res);

    // Vérification du résidu explicite
    PetscCall(MatMult(A, x, r));
    PetscCall(VecAXPY(r, -1.0, rhs));
    PetscCall(VecNorm(r, NORM_2, &res));
    PetscPrintf(PETSC_COMM_WORLD, "||Ax - b||_2 = %g\n", (double)res);

    // Sauvegarde de la solution
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "solution_x.bin",
                                   FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscPrintf(PETSC_COMM_WORLD, "PETSc scalar size: %lu bytes\n", sizeof(PetscScalar));


    // ----------------------------------------------------------
    // Benchmark SpMV AVX2 sur Ak = A^k pour k = 1..4
    // ----------------------------------------------------------

    // PetscPrintf(PETSC_COMM_WORLD, "\n[Benchmark SpMV AVX2 sur A^k]\n");

    // const PetscInt kmax = 10;
    // const PetscInt nrepeat = 1;

    // // Convertir A_real (BAIJ4) → AIJ pour permettre MatMatMult
    // Mat A_aij;
    // PetscCall(MatConvert(A_real, MATAIJ, MAT_INITIAL_MATRIX, &A_aij));

    // Mat A_power = A_aij; // A^1 en AIJ
    // Vec x_test, y_result;
    // PetscCall(VecDuplicate(rhs, &x_test));
    // PetscCall(VecDuplicate(rhs, &y_result));
    // PetscCall(VecSet(x_test, 1.0)); // x = [1, 1, ..., 1]

    // for (PetscInt k = 1; k <= kmax; ++k) {
    //     if (k > 1) {
    //         Mat A_next;
    //         PetscCall(MatMatMult(A_power, A_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_next));
    //         if (k > 2) PetscCall(MatDestroy(&A_power));
    //         A_power = A_next;
    //     }

    //     // Convertir A^k → BAIJ 4x4 pour tester AVX2
    //     Mat A_power_baij4;
    //     PetscCall(MatConvert(A_power, MATSEQBAIJ, MAT_INITIAL_MATRIX, &A_power_baij4));
    //     PetscCall(MatSetBlockSize(A_power_baij4, 4));

    //     // MatMult avec routine AVX2
    //     PetscLogDouble t1, t2;
    //     PetscCall(PetscTime(&t1));
    //     for (PetscInt rep = 0; rep < nrepeat; ++rep) {
    //         PetscCall(MatMult_SeqBAIJ_4_AVX2(A_power_baij4, x_test, y_result));
    //     }
    //     PetscCall(PetscTime(&t2));
    //     PetscPrintf(PETSC_COMM_WORLD, "k = %d | Average AVX2 SpMV time = %.6f s\n", k, (t2 - t1) / nrepeat);

    //     PetscCall(MatDestroy(&A_power_baij4));

    //         // Nettoyage
            // PetscCall(VecDestroy(&x_test));
            // PetscCall(VecDestroy(&y_result));
            // if (kmax > 1) PetscCall(MatDestroy(&A_power));
            // PetscCall(MatDestroy(&A_aij));
    // }


    // Mat A_baij, X_dense, Y_result;
    // PetscInt numBlocks = 2;       // nombre de blocs (chaque bloc = 4x4)
    // PetscInt blockSize = 4;
    // PetscInt globalSize = numBlocks * blockSize;
    // PetscInt numCols = 4;         // nombre de colonnes de la matrice dense X

    // // ---------- Création de A (BAIJ 4x4) ----------
    // MatCreateSeqBAIJ(PETSC_COMM_SELF, blockSize, globalSize, globalSize, 2, NULL, &A_baij);

    // for (PetscInt blkRow = 0; blkRow < numBlocks; ++blkRow) {
    // for (PetscInt blkCol = PetscMax(0, blkRow - 1); blkCol <= PetscMin(numBlocks - 1, blkRow + 1); ++blkCol) {
    //     PetscScalar blockValues[16];
    //     for (int k = 0; k < 16; ++k) {
    //     blockValues[k] = (PetscScalar)(1.0 + 100 * blkRow + 10 * blkCol + k);
    //     }
    //     MatSetValuesBlocked(A_baij, 1, &blkRow, 1, &blkCol, blockValues, INSERT_VALUES);
    // }
    // }

    // MatAssemblyBegin(A_baij, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(A_baij, MAT_FINAL_ASSEMBLY);

    // // ---------- Création de X (matrice dense d’entrée) ----------
    // MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, globalSize, numCols, NULL, &X_dense);

    // for (PetscInt row = 0; row < globalSize; ++row) {
    // for (PetscInt col = 0; col < numCols; ++col) {
    //     PetscScalar value = (PetscScalar)(row + 10 * col + 1);
    //     MatSetValue(X_dense, row, col, value, INSERT_VALUES);
    // }
    // }

    // MatAssemblyBegin(X_dense, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(X_dense, MAT_FINAL_ASSEMBLY);

    // // ---------- Allocation de Y ----------
    // MatDuplicate(X_dense, MAT_DO_NOT_COPY_VALUES, &Y_result);

    // // ---------- Appel de ta routine AVX2 ----------
    // PetscCall(MatMatMult_SeqBAIJ_4_AVX2(A_baij, X_dense, Y_result, 4));

    // // ---------- Affichage du résultat ----------
    // PetscPrintf(PETSC_COMM_SELF, "\nRésultat : Y = A * X (bloc 4x4 AVX2)\n");
    // MatView(Y_result, PETSC_VIEWER_STDOUT_SELF);

    // // ---------- Nettoyage ----------
    // MatDestroy(&A_baij);
    // MatDestroy(&X_dense);
    // MatDestroy(&Y_result);


    // ----- Nettoyage -----
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&rhs));
    PetscCall(KSPDestroy(&ksp));
    if (use_baij) {
        PetscCall(MatDestroy(&A));        // shell
        PetscCall(MatDestroy(&A_real));   // vraie BAIJ
    } else {
        PetscCall(MatDestroy(&A));
    }

    free(boundary_nodes);
    PetscCall(PetscFinalize());
    return 0;
}