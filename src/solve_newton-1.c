#include <petscksp.h>
#include <petscdmplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "integration.h"
#include "kernels.h"
#include <omp.h>
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
    // Nouveaux ajouts pour l'optimisation
    double Mloc_dt[12][12];  // M/dt précalculée
    double A0_plus_M_dt[12][12]; // A0 + M/dt précalculée
} ElementMatrices;

// Cache pour les valeurs de solution
typedef struct {
    double Uloc[3][4];
    double Uloc_old[3][4];
    double Ploc[4];
    PetscBool is_cached;
} SolutionCache;

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
            const int base = 4*nodes[i];
            local2global_v[3*i] = base;
            local2global_v[3*i + 1] = base + 1;
            local2global_v[3*i + 2] = base + 2;
            local2global_p[i] = base + 3;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            const int node = nodes[i];
            local2global_v[3*i] = node;
            local2global_v[3*i + 1] = node + nv;
            local2global_v[3*i + 2] = node + 2*nv;
            local2global_p[i] = node + 3*nv;
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

// Pré-calcul optimisé des matrices constantes
ElementMatrices* precompute_constant_matrices(int ne, double dt, double delta, double Re, SolutionCache **solution_cache_out) {
    ElementMatrices *element_matrices = (ElementMatrices*)malloc(ne * sizeof(ElementMatrices));
    SolutionCache *solution_cache = (SolutionCache*)malloc(ne * sizeof(SolutionCache));
    
    const double inv_dt = 1.0 / dt;
    
    #pragma omp parallel for
    for (int k = 0; k < ne; ++k) {
        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            a[i][0] = coords[gi][0];
            a[i][1] = coords[gi][1];
            a[i][2] = coords[gi][2];
        }
        
        ElementMatrices *em = &element_matrices[k];
        
        // Calcul du volume et des gradients
        em->vol = tet_volum(a[0], a[1], a[2], a[3]);
        tet_gradients(a, em->grad);
        
        // Matrices constantes
        mass_matrix(a[0], a[1], a[2], a[3], em->Mloc);
        diffusion_matrix(a, Re, em->A0loc);
        divergence_matrix(em->grad, em->vol, em->Bloc);
        pressure_stabilization_matrix(a, delta, em->Dloc);
        
        // Pré-calcul des combinaisons matricielles
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                em->Mloc_dt[i][j] = em->Mloc[i][j] * inv_dt;
                em->A0_plus_M_dt[i][j] = em->A0loc[i][j] + em->Mloc_dt[i][j];
            }
        }
        
        // Initialisation du cache
        solution_cache[k].is_cached = PETSC_FALSE;
    }
    
    *solution_cache_out = solution_cache;
    return element_matrices;
}

// Extraction optimisée des valeurs de solution avec cache
void extract_solution_values(Vec u, Vec u_old, int k, int nv, DoFOrdering ordering, SolutionCache *solution_cache) {
    SolutionCache *cache = &solution_cache[k];
    
    int local2global_v[12], local2global_p[4];
    get_local_to_global(nv, tet[k], local2global_v, local2global_p, ordering);
    
    // Extraction par blocs pour réduire les appels VecGetValues
    PetscScalar vals_u[16], vals_u_old[12];
    PetscInt indices_u[16], indices_u_old[12];
    
    // Préparer les indices
    for (int i = 0; i < 12; ++i) {
        indices_u[i] = local2global_v[i];
        indices_u_old[i] = local2global_v[i];
    }
    for (int i = 0; i < 4; ++i) {
        indices_u[12 + i] = local2global_p[i];
    }
    
    // Extraction groupée
    VecGetValues(u, 16, indices_u, vals_u);
    VecGetValues(u_old, 12, indices_u_old, vals_u_old);
    
    // Réorganisation des données
    for (int comp = 0; comp < 3; ++comp) {
        for (int i = 0; i < 4; ++i) {
            cache->Uloc[comp][i] = vals_u[3*i + comp];
            cache->Uloc_old[comp][i] = vals_u_old[3*i + comp];
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        cache->Ploc[i] = vals_u[12 + i];
    }
    
    cache->is_cached = PETSC_TRUE;
}

// Calcul du résidu ultra-optimisé
void compute_residual_optimized(Vec F, Vec u, Vec u_old, int nv, int ne, DoFOrdering ordering,
                               double dt, int Ndof_v, int Ndof_tot, 
                               ElementMatrices *element_matrices, SolutionCache *solution_cache) {
    
    VecZeroEntries(F);
    
    // Vecteur pour l'assemblage par blocs
    PetscScalar *global_residual = (PetscScalar*)calloc(Ndof_tot, sizeof(PetscScalar));
    
    #pragma omp parallel for
    for (int k = 0; k < ne; ++k) {
        // Extraction des valeurs de solution (avec cache)
        extract_solution_values(u, u_old, k, nv, ordering, solution_cache);
        
        SolutionCache *cache = &solution_cache[k];
        ElementMatrices *em = &element_matrices[k];
        
        // Récupération rapide des coordonnées
        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            memcpy(a[i], coords[gi], 3 * sizeof(double));
        }
        
        // Calcul des matrices de convection (seules dépendantes de la solution)
        double A1loc[12][12], A2loc[12][12];
        convection_matrix1(a, cache->Uloc, A1loc);
        
        double M4[4][4];
        mass_matrix_tet(em->vol, M4);
        convection_matrix2(cache->Uloc, em->grad, M4, A2loc);

        // Calcul optimisé du résidu local
        double Floc_v[12] = {0}, Floc_p[4] = {0};
        
        // Terme temporel et diffusion combinés : (M/dt + A0) * u - M/dt * u_old
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                int comp_i = i % 3, node_i = i / 3;
                int comp_j = j % 3, node_j = j / 3;
                
                Floc_v[i] += em->A0_plus_M_dt[i][j] * cache->Uloc[comp_j][node_j] 
                           - em->Mloc_dt[i][j] * cache->Uloc_old[comp_j][node_j];
            }
        }
        
        // Terme de convection : (A1 + A2) * u
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                int comp_j = j % 3, node_j = j / 3;
                Floc_v[i] += (A1loc[i][j] + A2loc[i][j]) * cache->Uloc[comp_j][node_j];
            }
        }
        
        // Terme de gradient de pression : B^T * p
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 4; ++j) {
                Floc_v[i] += em->Bloc[j][i] * cache->Ploc[j];
            }
        }
        
        // Terme de divergence et stabilisation de pression
        for (int i = 0; i < 4; ++i) {
            // Divergence : -B * u
            for (int j = 0; j < 12; ++j) {
                int comp_j = j % 3, node_j = j / 3;
                Floc_p[i] -= em->Bloc[i][j] * cache->Uloc[comp_j][node_j];
            }
            
            // Stabilisation : D * p
            for (int j = 0; j < 4; ++j) {
                Floc_p[i] += em->Dloc[i][j] * cache->Ploc[j];
            }
        }

        // Assemblage thread-safe dans le vecteur global
        int local2global_v[12], local2global_p[4];
        get_local_to_global(nv, tet[k], local2global_v, local2global_p, ordering);
        
        #pragma omp critical
        {
            for (int i = 0; i < 12; ++i) {
                global_residual[local2global_v[i]] += Floc_v[i];
            }
            for (int i = 0; i < 4; ++i) {
                global_residual[local2global_p[i]] += Floc_p[i];
            }
        }
    }
    
    // Assemblage final en une seule fois
    PetscInt *indices = (PetscInt*)malloc(Ndof_tot * sizeof(PetscInt));
    for (int i = 0; i < Ndof_tot; ++i) {
        indices[i] = i;
    }
    
    VecSetValues(F, Ndof_tot, indices, global_residual, INSERT_VALUES);
    VecAssemblyBegin(F);
    VecAssemblyEnd(F);
    
    free(global_residual);
    free(indices);
}

void convection_jacobian(double a[4][3], double Uloc[3][4], double grad[4][3], 
                                  double vol, double A1_jac[12][12], double A2_jac[12][12]) {
    
    memset(A1_jac, 0, sizeof(double) * 12 * 12);
    memset(A2_jac, 0, sizeof(double) * 12 * 12);
    
    const double vol_fourth = vol * 0.25;
    
    // **A1_jac optimisé**
    for (int i = 0; i < 4; ++i) {
        for (int alpha = 0; alpha < 3; ++alpha) {
            int row = 3*i + alpha;
            
            // Pré-calcul du gradient
            double grad_u_alpha[3] = {0, 0, 0};
            for (int l = 0; l < 4; ++l) {
                for (int j = 0; j < 3; ++j) {
                    grad_u_alpha[j] += Uloc[alpha][l] * grad[l][j];
                }
            }
            
            for (int k = 0; k < 4; ++k) {
                for (int j = 0; j < 3; ++j) {
                    int col = 3*k + j;
                    A1_jac[row][col] = vol_fourth * grad_u_alpha[j];
                }
            }
        }
    }
    
    // **A2_jac optimisé**
    for (int i = 0; i < 4; ++i) {
        for (int alpha = 0; alpha < 3; ++alpha) {
            int row = 3*i + alpha;
            
            for (int k = 0; k < 4; ++k) {
                int col = 3*k + alpha;
                
                double integral = 0.0;
                for (int j = 0; j < 3; ++j) {
                    double u_j_avg = 0.0;
                    for (int l = 0; l < 4; ++l) {
                        u_j_avg += Uloc[j][l];
                    }
                    integral += (u_j_avg * 0.25) * grad[k][j];
                }
                
                A2_jac[row][col] = vol_fourth * integral;
            }
        }
    }
}

void compute_convection_jacobian_exact(double a[4][3], double Uloc[3][4], 
                                      double grad[4][3], double vol,
                                      double A1_jac[12][12], double A2_jac[12][12]) {
    convection_jacobian(a, Uloc, grad, vol, A1_jac, A2_jac);
}

// Assemblage optimisé de la jacobienne
void assemble_jacobian_optimized(Mat J, Vec u, int nv, int ne, DoFOrdering ordering,
                                double dt, int Ndof_v, int Ndof_tot,
                                ElementMatrices *element_matrices, SolutionCache *solution_cache) {

    MatZeroEntries(J);
    
    #pragma omp parallel for
    for (int k = 0; k < ne; ++k) {
        
        // Utilisation du cache de solution
        SolutionCache *cache = &solution_cache[k];
        ElementMatrices *em = &element_matrices[k];
        
        double a[4][3];
        for (int i = 0; i < 4; ++i) {
            int gi = tet[k][i];
            memcpy(a[i], coords[gi], 3 * sizeof(double));
        }
        
        // Matrices dépendantes de la solution
        double A1loc[12][12], A2loc[12][12];
        double A1_jac[12][12], A2_jac[12][12];
        
        convection_matrix1(a, cache->Uloc, A1loc);
        double M4[4][4];
        mass_matrix_tet(em->vol, M4);
        convection_matrix2(cache->Uloc, em->grad, M4, A2loc);
        
        compute_convection_jacobian_exact(a, cache->Uloc, em->grad, em->vol, A1_jac, A2_jac);

        // Assemblage optimisé par blocs 4x4
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double block4x4[16];
                
                // Bloc vitesse-vitesse (3x3)
                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        int ii = 3*i + alpha;
                        int jj = 3*j + beta;
                        block4x4[alpha*4 + beta] = em->A0_plus_M_dt[ii][jj] + 
                                                   A1loc[ii][jj] + A2loc[ii][jj] +
                                                   A1_jac[ii][jj] + A2_jac[ii][jj];
                    }
                    // Bloc vitesse-pression
                    block4x4[alpha*4 + 3] = em->Bloc[j][3*i + alpha];
                }
                
                // Bloc pression-vitesse
                for (int beta = 0; beta < 3; ++beta) {
                    block4x4[3*4 + beta] = -em->Bloc[i][3*j + beta];
                }
                
                // Bloc pression-pression
                block4x4[15] = em->Dloc[i][j];

                PetscInt row = tet[k][i];
                PetscInt col = tet[k][j];
                
                #pragma omp critical
                {
                    MatSetValuesBlocked(J, 1, &row, 1, &col, block4x4, ADD_VALUES);
                }
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

                // Bloc A0 + M (diffusion + masse)
                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        int ii = 3 * i + alpha;
                        int jj = 3 * j + beta;
                        block4x4[alpha * 4 + beta] = em->A0loc[ii][jj]; //+ em->Mloc_dt[ii][jj]; 
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

    // === Configuration des paramètres ===
    PetscReal dt = 0.001;                  // Pas de temps initial
    PetscReal T = 0.01;                   // Temps final
    PetscInt  N_steps = (PetscInt)(T/dt);// Nombre de pas de temps
    PetscReal Re = 100;                // Nombre de Reynolds
    PetscReal delta = 0.1;               // Paramètre de stabilisation

    // Critères de convergence
    PetscReal rtol = 1e-6;               // Tolérance relative
    PetscReal atol = 1e-8;               // Tolérance absolue
    PetscReal stol = 1e-10;              // Seuil de stagnation
    PetscInt  max_newton = 30;           // Itérations max Newton
    PetscBool monitor = PETSC_TRUE;      // Affichage détaillé

    // === Lecture du maillage ===
    char mshname[PETSC_MAX_PATH_LEN];
    PetscBool flg;
    PetscOptionsGetString(NULL, NULL, "-msh", mshname, sizeof(mshname), &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Please provide -msh <file.msh>");

    char full_msh_path[PETSC_MAX_PATH_LEN];
    snprintf(full_msh_path, sizeof(full_msh_path), "/home/kempf/spmv/mesh/%s", mshname);

    int nv, ne, *boundary_nodes = NULL, nb_boundary_nodes = 0;
    read_mesh(full_msh_path, &nv, &ne, &boundary_nodes, &nb_boundary_nodes);

    // === Déclaration des variables locales pour les matrices ===
    ElementMatrices *element_matrices = NULL;
    SolutionCache *solution_cache = NULL;

    // === Initialisation PETSc ===
    int Ndof_v = 3 * nv;
    int Ndof_tot = Ndof_v + nv;

    // Vecteurs
    Vec u_n, u_old, delta_u, F;
    VecCreateSeq(PETSC_COMM_WORLD, Ndof_tot, &u_n);
    VecDuplicate(u_n, &u_old);
    VecDuplicate(u_n, &delta_u);
    VecDuplicate(u_n, &F);

    // Matrice Jacobienne
    Mat J;
    MatCreate(PETSC_COMM_WORLD, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(J, MATSEQBAIJ);
    MatSetBlockSize(J, 4);
    PetscInt max_nnz_per_row = 64;
    MatSeqBAIJSetPreallocation(J, 4, max_nnz_per_row, NULL);
    MatSetUp(J);

    // === Conditions aux limites ===
    PetscInt *rows = (PetscInt*)malloc(nb_boundary_nodes * 3 * sizeof(PetscInt));
    PetscScalar *values = (PetscScalar*)malloc(nb_boundary_nodes * 3 * sizeof(PetscScalar));
    PetscScalar *zeros = (PetscScalar*)calloc(nb_boundary_nodes * 3, sizeof(PetscScalar));

    double y0 = 0.0, z0 = 0.0, a = 1, b = 1, Umax = 1.0;
    int count = 0;

    for (int i = 0; i < nb_boundary_nodes; ++i) {
        int node = boundary_nodes[i];
        int tag = node_surface_tags[node];
        
        // Appliquer Dirichlet sur les tags 1,2 (no-slip) et slip sur 4,5,6,7
        if (tag == 1 || tag == 2) {
            // === NO-SLIP : ellipsoïde et inlet ===
            double ux = 0.0;
            if (tag == 2) { // Poiseuille sur tag 2
                double y = coords[node][1], z = coords[node][2];
                // double r2 = pow((y-y0)/a,2) * pow((z-z0)/b,2); //revoir
                // if (r2 <= 1.0) ux = Umax * (1.0 - r2);
                ux = (1-y*y)*(1-z*z); // Simplification pour Poiseuille
            }
            // Pour tag 1 (ellipsoïde), ux reste 0.0
            
            // Imposer toutes les composantes
            for (int d = 0; d < 3; ++d) {
                rows[count] = 4*node + d;
                values[count] = (d == 0) ? ux : 0.0;
                count++;
            }
        }
        else if (tag == 4 || tag == 5) {
            // === SLIP : surfaces bas/haut ===
            // Imposer seulement uz = 0 (composante normale)
            rows[count] = 4*node + 1; // uy
            values[count] = 0.0;
            count++;
            // ux et uy restent libres
        }
        else if (tag == 6 || tag == 7) {
            // === SLIP : surfaces gauche/droite ===
            // Imposer seulement uy = 0 (composante normale)
            rows[count] = 4*node + 2; // uz
            values[count] = 0.0;
            count++;
            // ux et uz restent libres
        }
        // Les nœuds avec tag = 3 ne sont PAS ajoutés (sortie libre)
    }

    // === Initialisation par Stokes ===
    element_matrices = precompute_constant_matrices(ne, dt, delta, 1.0, &solution_cache); // Re petit pour Stokes

    Mat S;
    MatCreate(PETSC_COMM_WORLD, &S);
    MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, Ndof_tot, Ndof_tot);
    MatSetType(S, MATSEQBAIJ);
    MatSetBlockSize(S, 4);
    MatSeqBAIJSetPreallocation(S, 4, max_nnz_per_row, NULL);
    MatSetUp(S);

    assemble_stokes_matrix(S, element_matrices, tet, ne, nv, ORDER_BLOCK_NODE);

    Vec rhs_stokes;
    VecCreateSeq(PETSC_COMM_WORLD, Ndof_tot, &rhs_stokes);
    VecSet(rhs_stokes, 0.0);
    VecSetValues(rhs_stokes, count, rows, values, INSERT_VALUES);
    VecAssemblyBegin(rhs_stokes); VecAssemblyEnd(rhs_stokes);

    MatZeroRows(S, count, rows, 1.0, NULL, NULL); //utiliser rhs 

    KSP ksp_stokes;
    KSPCreate(PETSC_COMM_WORLD, &ksp_stokes);
    KSPSetOperators(ksp_stokes, S, S);
    KSPSetType(ksp_stokes, KSPGMRES);
    PC pc_stokes;
    KSPGetPC(ksp_stokes, &pc_stokes);
    PCSetType(pc_stokes, PCILU);
    PCFactorSetLevels(pc_stokes, 3);
    KSPSetTolerances(ksp_stokes, 1e-12, 1e-12, PETSC_DEFAULT, 1000);
    KSPSetFromOptions(ksp_stokes);

    KSPSolve(ksp_stokes, rhs_stokes, u_n);
    VecCopy(u_n, u_old);

    // === Nettoyage Stokes ===
    KSPDestroy(&ksp_stokes);
    VecDestroy(&rhs_stokes);
    MatDestroy(&S);
    free(element_matrices);
    free(solution_cache);

    // === Préparation solveur Newton ===
    element_matrices = precompute_constant_matrices(ne, dt, delta, Re, &solution_cache); // Re réel

    KSP ksp_newton;
    KSPCreate(PETSC_COMM_WORLD, &ksp_newton);
    KSPSetOperators(ksp_newton, J, J);
    KSPSetType(ksp_newton, KSPGMRES);
    KSPGMRESSetRestart(ksp_newton, 30);
    PC pc;
    KSPGetPC(ksp_newton, &pc);
    PCSetType(pc, PCILU);
    PCFactorSetLevels(pc, 0);
    KSPSetTolerances(ksp_newton, 1e-8, 1e-10, PETSC_DEFAULT, 2000);
    KSPSetFromOptions(ksp_newton);

    // === Boucle temporelle principale ===
    VecSet(delta_u, 0.0);

    double start = MPI_Wtime();
    for (int step = 0; step < N_steps; ++step) {
        PetscPrintf(PETSC_COMM_WORLD, "\n=== Time step %d (t=%.3f) ===\n", step+1, (step+1)*dt);
        
        PetscReal initial_res_norm = 0.0;
        PetscInt converged = 0;

        for (int iter = 0; iter < max_newton && !converged; ++iter) {
            // Application CL
            VecSetValues(u_n, count, rows, values, INSERT_VALUES);
            VecAssemblyBegin(u_n); VecAssemblyEnd(u_n);

            // PetscReal norm_u;
            // VecNorm(u_n, NORM_2, &norm_u);
            // PetscPrintf(PETSC_COMM_WORLD, "||u_n|| = %.6e\n", norm_u);

            // Résidu
            compute_residual_optimized(F, u_n, u_old, nv, ne, ORDER_BLOCK_NODE, dt, Ndof_v, Ndof_tot, element_matrices, solution_cache);
            VecSetValues(F, count, rows, zeros, INSERT_VALUES);
            VecAssemblyBegin(F); VecAssemblyEnd(F);

            // Normes
            PetscReal res_norm;
            VecNorm(F, NORM_2, &res_norm);
            if (iter == 0) initial_res_norm = res_norm;

            PetscReal du_norm;
            VecNorm(delta_u, NORM_2, &du_norm);

            converged = (((res_norm < rtol * initial_res_norm) || (res_norm < atol))
                        && (du_norm < atol)) ? 1 : 0;

            // Vérification convergence
            converged = (((res_norm < rtol*initial_res_norm) || (res_norm < atol)) 
                        && (du_norm < atol)) ? 1 : 0;

            if (monitor) {
                PetscPrintf(PETSC_COMM_WORLD, "Newton %02d: |F|=%6.2e (rel %.1e), |du|=%6.2e %s\n",
                          iter, res_norm, res_norm/initial_res_norm, du_norm,
                          converged ? "CONVERGED" : "");
            }

            if (converged) break;

            // Résolution système linéaire
            assemble_jacobian_optimized(J, u_n, nv, ne, ORDER_BLOCK_NODE, dt, Ndof_v, Ndof_tot, element_matrices, solution_cache);
            MatZeroRows(J, count, rows, 1.0, NULL, NULL);
            
            VecScale(F, -1.0);
            KSPSolve(ksp_newton, F, delta_u);
            VecAXPY(u_n, 1.0, delta_u);

            // Détection stagnation
            if (iter > 5 && du_norm < stol) {
                PetscPrintf(PETSC_COMM_WORLD, "Warning: Stagnation (|du|=%.1e)\n", du_norm);
                break;
            }
        }

        if (!converged) {
            PetscPrintf(PETSC_COMM_WORLD, "Warning: Newton failed to converge\n");
            // Stratégie de fallback: réduire dt et réessayer?
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "res/solution_step%04d.dat", step+1);
        PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
        VecView(u_n, viewer);
        PetscViewerDestroy(&viewer);

        VecCopy(u_n, u_old);
    }

    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Total time: %.3f seconds\n", end - start);

    // === Nettoyage final ===
    KSPDestroy(&ksp_newton);
    VecDestroy(&u_n);
    VecDestroy(&u_old);
    VecDestroy(&delta_u);
    VecDestroy(&F);
    MatDestroy(&J);
    free(rows);
    free(values);
    free(zeros);
    free(boundary_nodes);
    free(node_surface_tags);
    free(element_matrices);
    free(solution_cache);
    free(tet);
    free(coords);

    PetscFinalize();
    return 0;
}