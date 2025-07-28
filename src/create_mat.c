#include <petscmat.h>
#include <stdio.h>
#include <stdlib.h>
#include "integration.h"

#ifdef __cplusplus
#include <set>
#include <vector>
#else
#error "Ce code nécessite d'être compilé en C++ pour std::set/std::vector"
#endif

// #define MAX_ELEMS 100000
// #define MAX_VERTS 100000

// int tet[MAX_ELEMS][4];
// double coords[MAX_VERTS][3];

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
get_local_to_global(int nv, const int nodes[4], int local2global_v[12], int local2global_p[4], DoFOrdering ordering)
{   

    if (ordering == ORDER_BLOCK_NODE) {
        // BAIJ4 - bloc de 4 par noeud (u_x,u_y,u_z,p)^0, (u_x,u_y,u_z,p)^1, ...
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = 4*nodes[i] + comp; // u_x, u_y, u_z
            local2global_p[i] = 4*nodes[i] + 3; // p
        }
        //printf("→ Indices BAIJ4 générés\n");
    } else {
        // AIJ - toutes les u_x, puis u_y, puis u_z, puis p (par composante)
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp)
                local2global_v[3*i + comp] = nodes[i] + comp*nv;  // vx, vy, vz
            local2global_p[i] = nodes[i] + 3*nv;                 // p
        }
        //printf("→ Indices AIJ générés\n");
    }
}



// --------- Mesh Parsing ---------
// void read_mesh(const char *filename, int *nv, int *ne) {
//     FILE *f = fopen(filename, "r");
//     if (!f) { fprintf(stderr, "Cannot open .msh file: %s\n", filename); exit(1); }

//     char line[256];
//     *nv = *ne = 0;
//     while (fgets(line, sizeof(line), f)) {
//         if (strncmp(line, "$Nodes", 6) == 0) {
//             fscanf(f, "%d\n", nv);
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
//             int count = 0;
//             for (int i = 0; i < total_elem; ++i) {
//                 int id, type, ntags, tag1, v[4];
//                 fscanf(f, "%d %d %d", &id, &type, &ntags); 
//                 for (int t = 0; t < ntags; ++t) fscanf(f, "%d", &tag1);
//                 if (type == 4) {
//                     for (int j = 0; j < 4; ++j)
//                         fscanf(f, "%d", &v[j]);
//                     for (int j = 0; j < 4; ++j)
//                         tet[count][j] = v[j] - 1;
//                     count++;
//                 } else {
//                     fgets(line, sizeof(line), f);
//                 }
//             }
//             *ne = count;
//         }
//     }
//     fclose(f);
// }

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
            *boundary_nodes = (int *) malloc((*nv) * sizeof(int)); // alloue à nv max
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

            // Première passe pour compter les tétraèdres
            int tet_count = 0;
            long file_pos = ftell(f);
            for (int i = 0; i < total_elem; ++i) {
                int id, type, ntags;
                fscanf(f, "%d %d %d", &id, &type, &ntags);
                for (int t = 0; t < ntags; ++t) { int dummy; fscanf(f, "%d", &dummy); }
                if (type == 4) { int dummy; for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); tet_count++; }
                else if (type == 2) { int dummy; for (int j = 0; j < 3; ++j) fscanf(f, "%d", &dummy); }
                else if (type == 3) { int dummy; for (int j = 0; j < 4; ++j) fscanf(f, "%d", &dummy); }
                else fgets(line, sizeof(line), f);
            }

            // Allocation
            *ne = tet_count;
            tet = (int (*)[4]) malloc((*ne) * sizeof(int[4]));

            // Deuxième passe pour lire vraiment
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
                    int is_dirichlet = 0;
                    for (int t = 0; t < ntags; ++t) {
                        if (tags[t] == 1 || tags[t] == 2 || tags[t] == 4 ||
                            tags[t] == 5 || tags[t] == 6 || tags[t] == 7) {
                            is_dirichlet = 1;
                            break;
                        }
                    }
                    int nverts = (type == 2 ? 3 : 4);
                    int v[4];
                    for (int j = 0; j < nverts; ++j)
                        fscanf(f, "%d", &v[j]);

                    if (is_dirichlet) {
                        for (int j = 0; j < nverts; ++j) {
                            int node = v[j] - 1;
                            if (!contains(*boundary_nodes, *nb_boundary_nodes, node)) {
                                (*boundary_nodes)[*nb_boundary_nodes] = node;
                                (*nb_boundary_nodes)++;
                            }
                        }
                    }
                }
                else {
                    fgets(line, sizeof(line), f); // skip
                }
            }
        }
    }

    fclose(f);
}
// --------- Matrix Assembly ---------
void assemble_matrix(Mat A, int nv, int ne, DoFOrdering ordering,
                    double dt, double delta, double Re, int Ndof_v, int Ndof_tot) {

    //printf("ordering = %d\n", ordering);
    
    for (int k = 0; k < ne; ++k) {
        int local2global_v[12], local2global_p[4];
        get_local_to_global(nv, tet[k], local2global_v, local2global_p, ordering);

        // if (k == 0 && ordering == ORDER_BLOCK_NODE) {
        //     printf("DOFs pour élément 0 :\n");
        //     for (int i = 0; i < 4; ++i) {
        //         printf("%d %d %d %d ",
        //             4*tet[k][i] + 0,
        //             4*tet[k][i] + 1,
        //             4*tet[k][i] + 2,
        //             4*tet[k][i] + 3);
        //     }
        //     printf("\n");
        // }
        

        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            a[i][0] = coords[gi][0];
            a[i][1] = coords[gi][1];
            a[i][2] = coords[gi][2];
        }
        

        // --- Champ vitesse U constant ---
        // double Uloc[3][4];
        // for (int comp = 0; comp < 3; ++comp)
        //     for (int i = 0; i < 4; ++i)
        //         Uloc[comp][i] = 1.0; 

        // --- Champ vitesse U custom ---
        double Uloc[3][4] = {
            {1.0, 2.0, 3.0, 4.0},      // Ux
            {0.0, -1.0, 0.5, 2.0},     // Uy
            {-2.0, 1.0, 0.0, -1.0}     // Uz
        };

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
                //printf("→ Assemblage AIJ\n");
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
                        block4x4[3*4 + 3] = Dloc[i][j];

                        PetscInt row = tet[k][i];
                        PetscInt col = tet[k][j];
                        MatSetValuesBlocked(A, 1, &row, 1, &col, block4x4, ADD_VALUES);
                    }
                }
        }
    }
}


int main(int argc, char **args) {
    PetscInitialize(&argc, &args, NULL, NULL);

    char mshname[PETSC_MAX_PATH_LEN];
    PetscBool flg;
    PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

    char full_msh_path[PETSC_MAX_PATH_LEN];
    snprintf(full_msh_path, sizeof(full_msh_path), "/home/users/u0001668/spmv/mesh/%s", mshname);

    int nv, ne;
    int *boundary_nodes = NULL; 
    int nb_boundary_nodes = 0;
    read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);
    PetscPrintf(PETSC_COMM_WORLD, "Parsed %d vertices, %d tetrahedra\n", nv, ne);

    // ----------- Paramètres physiques -----------
    double dt = 0.1, delta = 0.1, Re = 1.0;
    int Ndof_v = 3 * nv, Ndof_p = nv, Ndof_tot = Ndof_v + Ndof_p;

    // --- Matrice AIJ ---
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(A, MATSEQAIJ);
    MatSetFromOptions(A);
    MatSetUp(A);
    
    assemble_matrix(A, nv, ne, ORDER_BY_COMPONENT, dt, delta, Re, Ndof_v, Ndof_tot);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // --- Appliquer conditions aux limites Dirichlet ---
    PetscInt *rows_aij = (PetscInt*) malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    int count_aij = 0;
    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];
        rows_aij[count_aij++] = node;        // u_x
        rows_aij[count_aij++] = node + nv;   // u_y
        rows_aij[count_aij++] = node + 2 * nv; // u_z
    }

    Vec rhs_aij;
    MatCreateVecs(A, NULL, &rhs_aij);
    VecSet(rhs_aij, 0.0);
    MatZeroRows(A, count_aij, rows_aij, 1.0, rhs_aij, rhs_aij);
    VecDestroy(&rhs_aij);
    free(rows_aij);

    save_matrix(A, "/home/users/u0001668/spmv/mat2/matrix_aij_stokes.bin");
    save_matrix_mtx(A, "/home/users/u0001668/spmv/mat/matrix_aij_stokes.mtx");
    PetscPrintf(PETSC_COMM_WORLD, "Matrice AIJ sauvegardée.\n");
    PetscViewer vAIJ;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrixAIJ.txt", &vAIJ);
    MatView(A, vAIJ); PetscViewerDestroy(&vAIJ);
    MatDestroy(&A);

    // --- Matrice AIJ' ---
    Mat Ap;
    MatCreate(PETSC_COMM_WORLD, &Ap);
    MatSetSizes(Ap, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(Ap, MATSEQAIJ);
    MatSetFromOptions(Ap);
    MatSetUp(Ap);
    
    assemble_matrix(Ap, nv, ne, ORDER_BLOCK_NODE, dt, delta, Re, Ndof_v, Ndof_tot);

    MatAssemblyBegin(Ap, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(Ap, MAT_FINAL_ASSEMBLY);

    // --- Appliquer conditions aux limites Dirichlet ---
    PetscInt *rows_aijp = (PetscInt*) malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    int count_aijp = 0;
    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];
        rows_aijp[count_aijp++] = 4 * node + 0; // u_x
        rows_aijp[count_aijp++] = 4 * node + 1; // u_y
        rows_aijp[count_aijp++] = 4 * node + 2; // u_z
    }

    Vec rhs_aijp;
    MatCreateVecs(Ap, NULL, &rhs_aijp);
    VecSet(rhs_aijp, 0.0);
    MatZeroRows(Ap, count_aij, rows_aijp, 1.0, rhs_aijp, rhs_aijp);
    VecDestroy(&rhs_aijp);
    free(rows_aijp);

    save_matrix(Ap, "/home/users/u0001668/spmv/mat2/matrix_aijp_stokes.bin");
    save_matrix_mtx(Ap, "/home/users/u0001668/spmv/mat/matrix_aijp_stokes.mtx");
    PetscPrintf(PETSC_COMM_WORLD, "Matrice AIJp sauvegardée.\n");
    PetscViewer vAIJp;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrixAIJp.txt", &vAIJp);
    MatView(Ap, vAIJp); PetscViewerDestroy(&vAIJp);
    MatDestroy(&Ap);

    // --- Matrice BAIJ 4x4 ---
    Mat B4;
    MatCreate(PETSC_COMM_WORLD, &B4);
    MatSetSizes(B4, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(B4, MATSEQBAIJ);
    MatSetBlockSize(B4, 4);
    MatSetFromOptions(B4);
    MatSetUp(B4);

    assemble_matrix(B4, nv, ne, ORDER_BLOCK_NODE, dt, delta, Re, Ndof_v, Ndof_tot);

    MatAssemblyBegin(B4, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(B4, MAT_FINAL_ASSEMBLY);

    // --- Appliquer conditions aux limites Dirichlet ---
    PetscInt *rows_baij = (PetscInt*) malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    int count_baij = 0;
    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];
        rows_baij[count_baij++] = 4 * node + 0; // u_x
        rows_baij[count_baij++] = 4 * node + 1; // u_y
        rows_baij[count_baij++] = 4 * node + 2; // u_z
    }

    Vec rhs_baij;
    MatCreateVecs(B4, NULL, &rhs_baij);
    VecSet(rhs_baij, 0.0);
    MatZeroRows(B4, count_baij, rows_baij, 1.0, rhs_baij, rhs_aij);
    VecDestroy(&rhs_baij);
    free(rows_baij);

    save_matrix(B4, "/home/users/u0001668/spmv/mat2/matrix_baij4_stokes.bin");
    save_matrix_mtx(B4, "/home/users/u0001668/spmv/mat/matrix_baij4_stokes.mtx");
    PetscPrintf(PETSC_COMM_WORLD, "Matrice BAIJ4 sauvegardée.\n");
    PetscViewer vBAIJ4;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrixBAIJ4.txt", &vBAIJ4);
    MatView(B4, vBAIJ4); PetscViewerDestroy(&vBAIJ4);
    MatDestroy(&B4);

    free(tet);
    free(coords);
    free(boundary_nodes);

    PetscFinalize();
    return 0;
}
