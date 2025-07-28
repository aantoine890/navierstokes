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
#error "Ce code nécessite d'être compilé en C++ pour std::set/std::vector"
#endif

int (*tet)[4];
double (*coords)[3];
int *node_surface_tags = NULL;

// Structure pour stocker les matrices constantes pré-calculées
typedef struct {
    double Mloc[12][12];     // Matrice de masse
    double A0loc[12][12];    // Matrice de diffusion
    double Bloc[4][12];      // Matrice de divergence
    double Dloc[4][4];       // Matrice de stabilisation de pression
    double vol;              // Volume de l'élément
    double grad[4][3];       // Gradients des fonctions de forme
} ElementMatrices;

ElementMatrices *element_matrices = NULL;

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
                    for (int t = 1; t < ntags; ++t) {
                    
                        // int t=1;
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

// Pré-calcul des matrices constantes pour tous les éléments
void precompute_constant_matrices(int ne, double dt, double delta, double Re) {
    element_matrices = (ElementMatrices*)malloc(ne * sizeof(ElementMatrices));
    
    for (int k = 0; k < ne; ++k) {
        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            a[i][0] = coords[gi][0];
            a[i][1] = coords[gi][1];
            a[i][2] = coords[gi][2];
        }
        
        // Calcul du volume et des gradients
        element_matrices[k].vol = tet_volum(a[0], a[1], a[2], a[3]);
        tet_gradients(a, element_matrices[k].grad);
        
        // Matrices constantes
        mass_matrix(a[0], a[1], a[2], a[3], element_matrices[k].Mloc);
        diffusion_matrix(a, Re, element_matrices[k].A0loc);
        divergence_matrix(element_matrices[k].grad, element_matrices[k].vol, element_matrices[k].Bloc);
        pressure_stabilization_matrix(a, delta, element_matrices[k].Dloc);
    }
}

// Calcul du résidu optimisé (réutilise les matrices pré-calculées)
void compute_residual(Vec F, Vec u, Vec u_old, int nv, int ne, DoFOrdering ordering,
                               double dt, int Ndof_v, int Ndof_tot) {
    
    VecZeroEntries(F);
    
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

        // Vitesses actuelles et précédentes
        double Uloc[3][4], Uloc_old[3][4];
        double Ploc[4];
        
        for (int comp = 0; comp < 3; ++comp) {
            for (int i = 0; i < 4; ++i) {
                PetscScalar val_u, val_u_old;
                PetscInt idx = local2global_v[3*i + comp];
                VecGetValues(u, 1, &idx, &val_u);
                VecGetValues(u_old, 1, &idx, &val_u_old);
                Uloc[comp][i] = val_u;
                Uloc_old[comp][i] = val_u_old;
            }
        }
        
        for (int i = 0; i < 4; ++i) {
            PetscScalar val_p;
            PetscInt idx = local2global_p[i];
            VecGetValues(u, 1, &idx, &val_p);
            Ploc[i] = val_p;
        }

        // Récupération des matrices pré-calculées
        ElementMatrices *em = &element_matrices[k];
        
        // Calcul des matrices de convection (seules dépendantes de la solution)
        double A1loc[12][12], A2loc[12][12];
        convection_matrix1(a, Uloc, A1loc);
        double M4[4][4];
        mass_matrix_tet(em->vol, M4);
        convection_matrix2(Uloc, em->grad, M4, A2loc);

        // Calcul du résidu local
        double Floc_v[12] = {0}, Floc_p[4] = {0};
        
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp) {
                int ii = 3*i + comp;
                
                // Terme temporel : M(u - u_old)/dt (matrice M pré-calculée)
                for (int j = 0; j < 4; ++j) {
                    for (int comp2 = 0; comp2 < 3; ++comp2) {
                        int jj = 3*j + comp2;
                        // Floc_v[ii] += (em->Mloc[ii][jj]/dt) * (Uloc[comp2][j] - Uloc_old[comp2][j]);
                    }
                }
                
                // Terme de diffusion : A0 * u (matrice A0 pré-calculée)
                for (int j = 0; j < 4; ++j) {
                    for (int comp2 = 0; comp2 < 3; ++comp2) {
                        int jj = 3*j + comp2;
                        Floc_v[ii] += em->A0loc[ii][jj] * Uloc[comp2][j];
                    }
                }
                
                // Terme de convection : (A1 + A2) * u (recalculé à chaque itération)
                for (int j = 0; j < 4; ++j) {
                    for (int comp2 = 0; comp2 < 3; ++comp2) {
                        int jj = 3*j + comp2;
                        // Floc_v[ii] += (A1loc[ii][jj] + A2loc[ii][jj]) * Uloc[comp2][j];
                        Floc_v[ii] += (A1loc[ii][jj]) * Uloc[comp2][j];
                    }
                }
                
                // Terme de gradient de pression : B^T * p (matrice B pré-calculée)
                for (int j = 0; j < 4; ++j) {
                    Floc_v[ii] += em->Bloc[j][ii] * Ploc[j];
                }
            }
        }
        
        for (int i = 0; i < 4; ++i) {
            // Terme de divergence : -B * u (matrice B pré-calculée)
            for (int j = 0; j < 4; ++j) {
                for (int comp = 0; comp < 3; ++comp) {
                    int jj = 3*j + comp;
                    Floc_p[i] -= em->Bloc[i][jj] * Uloc[comp][j];
                }
            }
            
            // Terme de stabilisation de pression : D * p (matrice D pré-calculée)
            for (int j = 0; j < 4; ++j) {
                Floc_p[i] += em->Dloc[i][j] * Ploc[j];
            }
        }

        // Assemblage dans le vecteur résidu global
        for (int i = 0; i < 4; ++i) {
            for (int comp = 0; comp < 3; ++comp) {
                PetscInt idx = local2global_v[3*i + comp];
                VecSetValue(F, idx, Floc_v[3*i + comp], ADD_VALUES);
            }
            PetscInt idx_p = local2global_p[i];
            VecSetValue(F, idx_p, Floc_p[i], ADD_VALUES);
        }
    }
    
    VecAssemblyBegin(F);
    VecAssemblyEnd(F);
}

void convection_jacobian(double a[4][3], double Uloc[3][4], double grad[4][3], 
                                  double vol, double A1_jac[12][12], double A2_jac[12][12]) {
    
    memset(A1_jac, 0, sizeof(double) * 12 * 12);
    memset(A2_jac, 0, sizeof(double) * 12 * 12);
    
    // **A1_jac : Terme ∂/∂u[u·∇u] où la dérivée porte sur le premier u**
    // Formule : ∫_Ω φ_i δ_jk ∂u_α/∂x_j dΩ
    for (int i = 0; i < 4; ++i) {
        for (int alpha = 0; alpha < 3; ++alpha) {
            int row = 3*i + alpha;
            
            for (int k = 0; k < 4; ++k) {
                for (int j = 0; j < 3; ++j) {  // j = direction de la variation
                    int col = 3*k + j;
                    
                    // ∂u_α/∂x_j = ∑_l u_α^l grad[l][j]
                    double grad_u_alpha_j = 0.0;
                    for (int l = 0; l < 4; ++l) {
                        grad_u_alpha_j += Uloc[alpha][l] * grad[l][j];
                    }
                    
                    // Intégrale avec fonction test constante par morceaux
                    A1_jac[row][col] += vol * (1.0/4.0) * grad_u_alpha_j;
                }
            }
        }
    }
    
    // **A2_jac : Terme ∂/∂u[u·∇u] où la dérivée porte sur le ∇u**
    // Formule : ∫_Ω φ_i u_j ∂φ_k/∂x_j dΩ
    for (int i = 0; i < 4; ++i) {
        for (int alpha = 0; alpha < 3; ++alpha) {
            int row = 3*i + alpha;
            
            for (int k = 0; k < 4; ++k) {
                int col = 3*k + alpha;  // même composante
                
                double integral = 0.0;
                for (int j = 0; j < 3; ++j) {  // direction de convection
                    // Vitesse moyenne dans la direction j
                    double u_j_avg = 0.0;
                    for (int l = 0; l < 4; ++l) {
                        u_j_avg += (1.0/4.0) * Uloc[j][l];
                    }
                    integral += u_j_avg * grad[k][j];
                }
                
                A2_jac[row][col] = vol * (1.0/4.0) * integral;
            }
        }
    }
}

void compute_convection_jacobian_exact(double a[4][3], double Uloc[3][4], 
                                      double grad[4][3], double vol,
                                      double A1_jac[12][12], double A2_jac[12][12]) {
    
    // Calcul exact de la jacobienne (utilise grad et vol pré-calculés)
    convection_jacobian(a, Uloc, grad, vol, A1_jac, A2_jac);
}

// Assemblage de la jacobienne (réutilise les matrices pré-calculées)
void assemble_jacobian(Mat J, Vec u, int nv, int ne, DoFOrdering ordering,
                                double dt, int Ndof_v, int Ndof_tot) {

    MatZeroEntries(J);
    
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

        // Vitesses actuelles pour le calcul de la jacobienne
        double Uloc[3][4];
        for (int comp = 0; comp < 3; ++comp) {
            for (int i = 0; i < 4; ++i) {
                PetscScalar val;
                PetscInt idx = local2global_v[3*i + comp];
                VecGetValues(u, 1, &idx, &val);
                Uloc[comp][i] = val;
            }
        }

        // Récupération des matrices pré-calculées
        ElementMatrices *em = &element_matrices[k];
        
        // Matrices dépendantes de la solution (recalculées)
        double A1loc[12][12], A2loc[12][12];
        double A1_jac[12][12], A2_jac[12][12];
        
        convection_matrix1(a, Uloc, A1loc);
        double M4[4][4];
        mass_matrix_tet(em->vol, M4);
        convection_matrix2(Uloc, em->grad, M4, A2loc);
        
        compute_convection_jacobian_exact(a, Uloc, em->grad, em->vol, A1_jac, A2_jac);

        // Assemblage de la jacobienne
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double block4x4[16] = {0};
                
                // Bloc (i,j) de la jacobienne
                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        int ii = 3*i + alpha;
                        int jj = 3*j + beta;
                        // Jacobienne = M/dt + A0 + A1 + A2 + termes jacobiens
                        // (M et A0 sont pré-calculées)
                        block4x4[alpha*4 + beta] = em->A0loc[ii][jj] + 
                                                   A1loc[ii][jj] + A2loc[ii][jj] +
                                                   A1_jac[ii][jj] + A2_jac[ii][jj];
                    }
                    // Terme gradient de pression dans la jacobienne (B pré-calculée)
                    block4x4[alpha*4 + 3] = em->Bloc[j][3*i + alpha];
                }
                
                // Terme divergence dans la jacobienne (B pré-calculée)
                for (int beta = 0; beta < 3; ++beta) {
                    block4x4[3*4 + beta] = -em->Bloc[i][3*j + beta];
                }
                
                // Terme de stabilisation de pression (D pré-calculée)
                block4x4[15] = em->Dloc[i][j];

                PetscInt row = tet[k][i];
                PetscInt col = tet[k][j];
                MatSetValuesBlocked(J, 1, &row, 1, &col, block4x4, ADD_VALUES);
            }
        }
    }
    
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
}

PetscErrorCode assemble_stokes_matrix(Mat S, const ElementMatrices *element_matrices, const PetscInt (*tet)[4],
                                      PetscInt ne, PetscInt nv, PetscInt ORDER_BLOCK_NODE)
{
    PetscErrorCode ierr;

    for (PetscInt k = 0; k < ne; ++k) {
        PetscInt local2global_v[12], local2global_p[4];
        get_local_to_global(nv, tet[k], local2global_v, local2global_p, (DoFOrdering)ORDER_BLOCK_NODE);

        const ElementMatrices *em = &element_matrices[k];

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double block4x4[16] = {0.0};

                // Bloc A0 (diffusion) dans les 3x3 premières composantes
                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        int ii = 3 * i + alpha;
                        int jj = 3 * j + beta;
                        block4x4[alpha * 4 + beta] = em->A0loc[ii][jj]; //+ em->Mloc[ii][jj]; 
                    }
                    // Bloc B^T (pression -> vitesse)
                    block4x4[alpha * 4 + 3] = em->Bloc[j][3 * i + alpha];
                }

                // Bloc B (vitesse -> pression)
                for (int beta = 0; beta < 3; ++beta) {
                    block4x4[3 * 4 + beta] = -em->Bloc[i][3 * j + beta];
                }

                // Bloc D (stabilisation pression)
                block4x4[15] = em->Dloc[i][j];

                // Placement dans la matrice globale (bloc 4x4)
                PetscInt row = tet[k][i];
                PetscInt col = tet[k][j];
                ierr = MatSetValuesBlocked(S, 1, &row, 1, &col, block4x4, ADD_VALUES); CHKERRQ(ierr);
            }
        }
    }

    ierr = MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return 0;
}

int main(int argc, char **args) {
    PetscInitialize(&argc, &args, NULL, NULL);

    char mshname[PETSC_MAX_PATH_LEN];
    PetscBool flg;
    PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

    char full_msh_path[PETSC_MAX_PATH_LEN];
    snprintf(full_msh_path, sizeof(full_msh_path), "/home/kempf/spmv/mesh/%s", mshname);

    int nv, ne, *boundary_nodes = NULL, nb_boundary_nodes = 0;
    read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);

    double dt = 0.01, delta = 0.1, Re = 1;
    int Ndof_v = 3 * nv;
    int Ndof_tot = Ndof_v + nv;

    // Pré-calcul des matrices constantes
    precompute_constant_matrices(ne, dt, delta, Re);

    // Solution actuelle, ancienne et résidu
    Vec u_n, u_old, delta_u, F;
    VecCreateSeq(PETSC_COMM_WORLD, Ndof_tot, &u_n);
    VecDuplicate(u_n, &u_old);
    VecDuplicate(u_n, &delta_u);
    VecDuplicate(u_n, &F);

    // Matrice jacobienne
    Mat J;
    MatCreate(PETSC_COMM_WORLD, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(J, MATSEQBAIJ);
    MatSetBlockSize(J, 4);
    MatSetUp(J);

    // Conditions aux limites
    PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    PetscScalar *values = (PetscScalar*)malloc(nb_boundary_nodes * 3 * sizeof(PetscScalar));

    double y0 = 0.0, z0 = 0.0;   // Centre de l'ellipse
    double a = 1.0, b = 1.0;     // Demi-axes
    double Umax = 1.0;           // Vitesse max
    int count = 0;

    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];

        double ux = 0.0, uy = 0.0, uz = 0.0;

        if (node_surface_tags[node] == 2) {
            // Inlet → profil Poiseuille
            double y = coords[node][1];
            double z = coords[node][2];
            double dy = (y - y0) / a;
            double dz = (z - z0) / b;
            double r2 = dy * dy + dz * dz;

            if (r2 <= 1.0)
                ux = Umax * (1.0 - r2); // Poiseuille
        }

        // Tous les noeuds de bord sont traités → CL de Dirichlet (même pour tag != 2)
        for (int d = 0; d < 3; ++d) {
            rows[count] = 4 * node + d;
            values[count] = (d == 0) ? ux : 0.0;
            count++;
        }
    }
    // === Fin condition Poiseuille ===

    // === Initialisation via Stokes linéaire ===

    // 1. Assemblage de la matrice de Stokes
    Mat S;
    MatCreate(PETSC_COMM_WORLD, &S);
    MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(S, MATSEQBAIJ);
    MatSetBlockSize(S, 4);
    MatSetUp(S);

    assemble_stokes_matrix(S, element_matrices, tet, ne, nv, ORDER_BLOCK_NODE); 

    // 2. Second membre nul (problème homogène), on applique uniquement les CL
    Vec rhs_stokes;
    VecCreateSeq(PETSC_COMM_WORLD, Ndof_tot, &rhs_stokes);
    VecSet(rhs_stokes, 0.0);
    VecSetValues(rhs_stokes, count, rows, values, INSERT_VALUES);
    VecAssemblyBegin(rhs_stokes); VecAssemblyEnd(rhs_stokes);

    // 3. Conditions aux limites sur la matrice
    MatZeroRows(S, count, rows, 1.0, NULL, NULL);

    // 4. Résolution du système
    KSP ksp_stokes;
    KSPCreate(PETSC_COMM_WORLD, &ksp_stokes);
    KSPSetOperators(ksp_stokes, S, S);
    KSPSetType(ksp_stokes, KSPGMRES);
    PC pc;
    KSPGetPC(ksp_stokes, &pc);
    PCSetType(pc, PCILU);
    PCFactorSetLevels(pc, 1);
    KSPSetTolerances(ksp_stokes, 1e-10, 1e-10, PETSC_DEFAULT, 1000);
    KSPSetFromOptions(ksp_stokes);
    KSPSolve(ksp_stokes, rhs_stokes, u_n);
    KSPDestroy(&ksp_stokes);

    // 5. On initialise aussi u_old avec cette solution
    VecCopy(u_n, u_old);

    // Nettoyage
    VecDestroy(&rhs_stokes);
    MatDestroy(&S);

    // === Fin initialisation Stokes ===

    PetscPrintf(PETSC_COMM_WORLD, "Solution initialized with Stokes\n");

    // Boucle de Newton
    PetscInt max_newton = 30;
    PetscReal tol_newton = 1e-8;
    
    for (int iter = 0; iter < max_newton; ++iter) {
        PetscPrintf(PETSC_COMM_WORLD, "[Newton] Iteration %d\n", iter);

        // Application des conditions aux limites à la solution actuelle
        VecSetValues(u_n, count, rows, values, INSERT_VALUES);
        VecAssemblyBegin(u_n); VecAssemblyEnd(u_n);

        // Calcul du résidu F = F(u_n) 
        compute_residual(F, u_n, u_old, nv, ne, ORDER_BLOCK_NODE, dt, Ndof_v, Ndof_tot);
        
        // // Application des conditions aux limites au résidu
        PetscScalar *zero_values = (PetscScalar*)calloc(count, sizeof(PetscScalar));
        VecSetValues(F, count, rows, zero_values, INSERT_VALUES);
        VecAssemblyBegin(F); VecAssemblyEnd(F);
        free(zero_values);

        // Check de convergence du résidu
        PetscReal res_norm;
        VecNorm(F, NORM_2, &res_norm);
        PetscPrintf(PETSC_COMM_WORLD, "  ||F(u_n)|| = %g\n", (double)res_norm);
        if (res_norm < tol_newton) {
            PetscPrintf(PETSC_COMM_WORLD, "Newton converged in %d iterations\n", iter);
            break;
        }

        // Assemblage de la jacobienne J = J(u_n) - VERSION OPTIMISÉE
        assemble_jacobian(J, u_n, nv, ne, ORDER_BLOCK_NODE, dt, Ndof_v, Ndof_tot);
        
        // Application des conditions aux limites à la jacobienne
        MatZeroRows(J, count, rows, 1.0, NULL, NULL);

        // Résolution du système J * delta_u = -F
        VecScale(F, -1.0); // -F 
        
        KSP ksp;
        PC pc;
        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, J, J);
        KSPSetType(ksp, KSPGMRES);
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCILU);
        PCFactorSetLevels(pc, 1);
        KSPSetTolerances(ksp, 1e-12, 1e-12, PETSC_DEFAULT, 2000);
        KSPSetFromOptions(ksp);
        KSPSolve(ksp, F, delta_u);
        KSPDestroy(&ksp);

        // Mise à jour de la solution
        VecAXPY(u_n, 1, delta_u); // u_n = u_n + delta_u
        
        // Application des conditions aux limites à la nouvelle solution
        VecSetValues(u_n, count, rows, values, INSERT_VALUES);
        VecAssemblyBegin(u_n); VecAssemblyEnd(u_n);


    }

    // Sauvegarde de la solution
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.dat", &viewer);
    VecView(u_n, viewer);
    PetscViewerDestroy(&viewer);

    // Nettoyage
    VecDestroy(&u_n);
    VecDestroy(&u_old);
    VecDestroy(&delta_u);
    VecDestroy(&F);
    MatDestroy(&J);
    free(rows);
    free(values);
    free(boundary_nodes);

    PetscFinalize();
    return 0;
}