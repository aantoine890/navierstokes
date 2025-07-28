#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to compute the volume of a tetrahedron given its vertices
double tet_volum(const double a0[3], const double a1[3], const double a2[3], const double a3[3]) {
    double v1[3] = {a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]};
    double v2[3] = {a2[0] - a0[0], a2[1] - a0[1], a2[2] - a0[2]};
    double v3[3] = {a3[0] - a0[0], a3[1] - a0[1], a3[2] - a0[2]};
     double det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
                  v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
                  v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
    return det / 6.0;
}


// Function to compute the gradients of the basis functions for a tetrahedron
void tet_gradients(const double a[4][3], double grad[4][3]) {
    // Vecteurs des arêtes a1-a0, a2-a0, a3-a0
    double e1[3] = {a[1][0] - a[0][0], a[1][1] - a[0][1], a[1][2] - a[0][2]};
    double e2[3] = {a[2][0] - a[0][0], a[2][1] - a[0][1], a[2][2] - a[0][2]};
    double e3[3] = {a[3][0] - a[0][0], a[3][1] - a[0][1], a[3][2] - a[0][2]};

    // Produit vectoriel e2 × e3 pour le volume
    double n[3];
    n[0] = e2[1] * e3[2] - e2[2] * e3[1];
    n[1] = e2[2] * e3[0] - e2[0] * e3[2];
    n[2] = e2[0] * e3[1] - e2[1] * e3[0];

    // Volume = (e1 · (e2 × e3)) / 6
    double vol6 = e1[0] * n[0] + e1[1] * n[1] + e1[2] * n[2]; // = 6V

    if (vol6 <= 0.0) {
        printf("⚠️  Volume négatif ou nul (det = %.3e)\n", vol6);
    }

    // Gradients des fonctions de forme
    for (int i = 0; i < 4; i++) {
        // Définir les 3 autres nœuds (ordre fixe pour garantir l'orientation)
        int j, k, l;
        switch (i) {
            case 0: j = 1; k = 2; l = 3; break; // Face 1-2-3
            case 1: j = 0; k = 3; l = 2; break; // Face 0-3-2 (ordre inversé pour orientation)
            case 2: j = 0; k = 1; l = 3; break; // Face 0-1-3
            case 3: j = 0; k = 2; l = 1; break; // Face 0-2-1 (ordre inversé pour orientation)
        }

        // Vecteurs de la face opposée à i
        double v1[3] = {a[k][0] - a[j][0], a[k][1] - a[j][1], a[k][2] - a[j][2]};
        double v2[3] = {a[l][0] - a[j][0], a[l][1] - a[j][1], a[l][2] - a[j][2]};

        // Produit vectoriel v1 × v2 (normale sortante)
        grad[i][0] = (v1[1] * v2[2] - v1[2] * v2[1]) / vol6;
        grad[i][1] = (v1[2] * v2[0] - v1[0] * v2[2]) / vol6;
        grad[i][2] = (v1[0] * v2[1] - v1[1] * v2[0]) / vol6;
    }

    // Vérification : somme des gradients doit être (0, 0, 0)
    double check[3] = {0, 0, 0};
    for (int i = 0; i < 4; i++) {
        check[0] += grad[i][0];
        check[1] += grad[i][1];
        check[2] += grad[i][2];
    }
    // printf("Somme des gradients : (%.3e, %.3e, %.3e)\n", check[0], check[1], check[2]);
}

// Function to compute the diameter of a tetrahedron
double tet_diameter(const double a[4][3]) {
    double maxd = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = i+1; j < 4; ++j) {
            double d2 = 0.0;
            for (int k = 0; k < 3; ++k)
                d2 += (a[i][k] - a[j][k]) * (a[i][k] - a[j][k]);
            double d = sqrt(d2);
            if (d > maxd) maxd = d;
        }
    return maxd;
}

// Function to compute the mass matrix for a tetrahedron (P1 basis)
void mass_matrix_tet(double vol, double M[4][4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            M[i][j] = (i == j ? vol/10.0 : vol/20.0);
}

// Function to compute the mass matrix for a tetrahedron (vectorial, 12x12)
void mass_matrix(const double a0[3], const double a1[3], const double a2[3], const double a3[3], double M[12][12]) {
    double vol = tet_volum(a0, a1, a2, a3);

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            M[i][j] = 0.0;
        }
    }

    for(int alpha = 0; alpha < 3; ++alpha){ // inverser avce la lopp i
        for(int i = 0; i < 4; ++i){
            for(int j = 0; j < 4; ++j){
                int I = 3*i + alpha;
                int J = 3*j + alpha;
                M[I][J] = (i == j ? vol/10.0 : vol/20.0);
            }
        }
    }
}

// Function to compute the diffusion matrix for a tetrahedron (12x12)
void diffusion_matrix(const double a[4][3], double Re, double A0[12][12]) {
    // 1. Calcul des gradients et volume
    double grad[4][3];
    tet_gradients(a, grad);
    double vol = tet_volum(a[0], a[1], a[2], a[3]);

    // 2. Matrice de pondération Coef (identique à FreeFEM)
    const double Coef[6] = {1.0, 1.0, 1.0, 0.5, 0.5, 0.5};

    // 3. Initialisation
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            A0[i][j] = 0.0;

    // 4. Boucle sur les fonctions de base vectorielles (4 nœuds × 3 composantes)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int comp_i = 0; comp_i < 3; ++comp_i) {
                for (int comp_j = 0; comp_j < 3; ++comp_j) {

                    int I = 3 * i + comp_i;
                    int J = 3 * j + comp_j;

                    // 5. Construction de EL(u) — 6 composantes du tenseur D(u)
                    double EL_u[6] = {
                        grad[i][0] * (comp_i == 0), // dx(u1)
                        grad[i][1] * (comp_i == 1), // dy(u2)
                        grad[i][2] * (comp_i == 2), // dz(u3)
                        grad[i][1] * (comp_i == 0) + grad[i][0] * (comp_i == 1), // dy(u1)+dx(u2)
                        grad[i][2] * (comp_i == 0) + grad[i][0] * (comp_i == 2), // dz(u1)+dx(u3)
                        grad[i][2] * (comp_i == 1) + grad[i][1] * (comp_i == 2)  // dz(u2)+dy(u3)
                    };

                    double EL_v[6] = {
                        grad[j][0] * (comp_j == 0),
                        grad[j][1] * (comp_j == 1),
                        grad[j][2] * (comp_j == 2),
                        grad[j][1] * (comp_j == 0) + grad[j][0] * (comp_j == 1),
                        grad[j][2] * (comp_j == 0) + grad[j][0] * (comp_j == 2),
                        grad[j][2] * (comp_j == 1) + grad[j][1] * (comp_j == 2)
                    };

                    // 6. Produit scalaire pondéré EL(u)^T * Coef * EL(v)
                    double val = 0.0;
                    for (int k = 0; k < 6; ++k)
                        val += Coef[k] * EL_u[k] * EL_v[k];

                    A0[I][J] += (2.0 / Re) * vol * val;
                }
            }
        }
    }
}

// Function to compute the convection matrix (linearized) for a tetrahedron (12x12)
void convection_matrix1(const double a[4][3], const double U[3][4], double A1[12][12]) {
    double grad[4][3], M[4][4];
    tet_gradients(a, grad);
    double vol = tet_volum(a[0], a[1], a[2], a[3]);
    mass_matrix_tet(vol, M);

    double G[3][3] = {{0}};
    for (int alpha = 0; alpha < 3; ++alpha)
        for (int beta = 0; beta < 3; ++beta)
            for (int m = 0; m < 4; ++m)
                G[alpha][beta] += U[alpha][m] * grad[m][beta];

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int alpha = 0; alpha < 3; ++alpha)
                for (int beta = 0; beta < 3; ++beta) {
                    int I = 3*i + alpha;
                    int J = 3*j + beta;
                    A1[I][J] = G[alpha][beta] * M[i][j];
                }
}

// Function to compute the convection matrix (non-linear) for a tetrahedron (12x12)
void convection_matrix2(const double U[3][4], const double grad[4][3], const double M[4][4], double A2[12][12]) {
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            A2[i][j] = 0.0;

    for (int beta = 0; beta < 3; ++beta) {
        for (int j = 0; j < 4; ++j) {
            double C[4] = {0, 0, 0, 0};
            for (int m = 0; m < 4; ++m)
                for (int d = 0; d < 3; ++d)
                    C[m] += U[d][m] * grad[j][d];
            for (int i = 0; i < 4; ++i) {
                int I = 3*i + beta;
                int J = 3*j + beta;
                for (int m = 0; m < 4; ++m)
                    A2[I][J] += -C[m] * M[i][m];
            }
        }
    }
}

// Function to compute the divergence matrix (4x12)
void divergence_matrix(const double grad[4][3], double vol, double Bk[4][12]) {
    for (int i = 0; i < 4; ++i) {          
        for (int j = 0; j < 4; ++j) {     
            for (int alpha = 0; alpha < 3; ++alpha) { 
                int J = 3*j + alpha;
                Bk[i][J] = (vol / 4.0) * grad[j][alpha];
            }
        }
    }
}

// Function to compute the pressure stabilization matrix (4x4)
void pressure_stabilization_matrix(const double a[4][3], double delta, double D[4][4]){
    double grad[4][3];
    tet_gradients(a, grad);
    double vol = tet_volum(a[0], a[1], a[2], a[3]);
    double d = tet_diameter(a);
    double hk2 = d * d;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double scal = 0.0;
            for (int k = 0; k < 3; ++k)
                scal += grad[i][k] * grad[j][k];
            D[i][j] = delta * hk2 * vol * scal;
        }
}

// Function to assemble the global matrix for the Stokes problem
void global_matrix(int Nnodes, int Nelements, const int elements[][4], const double nodes[][3],const double U[3][4], double dt, double delta, double Re, double *Aglob){
    int Ndof_v = 3 * Nnodes;
    int Ndof_p = Nnodes;
    int Ndof_tot = Ndof_v + Ndof_p;
    memset(Aglob, 0, Ndof_tot * Ndof_tot * sizeof(double));

    // Local matrices
    double Mloc[12][12], A0loc[12][12], A1loc[12][12], A2loc[12][12];
    double Bloc[4][12], Dloc[4][4];
    double Uloc[3][4];

    // Loop over elements
    for (int e = 0; e < Nelements; ++e) {
        int vg[12], pg[4];
        double a[4][3];
        // Mapping local to global indices
        for (int i = 0; i < 4; ++i) {
            int gi = elements[e][i];
            a[i][0] = nodes[gi][0];
            a[i][1] = nodes[gi][1];
            a[i][2] = nodes[gi][2];
            pg[i] = gi;
            for (int comp = 0; comp < 3; ++comp)
                //vg[3*i + comp] = 3*gi + comp; // bloc by coordinate
                vg[3*i + comp] = gi + comp * Nnodes; // bloc by node

        }
        // U values at the vertices 
        if (U != NULL) {
            for (int comp = 0; comp < 3; ++comp)
                for (int i = 0; i < 4; ++i)
                    Uloc[comp][i] = U[comp][elements[e][i]];
        }

        // Mass Matrix
        mass_matrix(a[0], a[1], a[2], a[3], Mloc);

        // Diffusion Matrix
        diffusion_matrix(a, Re, A0loc);

        // Convection Matrices
        if (U != NULL) convection_matrix1(a, Uloc, A1loc); else memset(A1loc, 0, sizeof(A1loc));

        if (U != NULL) {
            double grad[4][3], M4[4][4];
            tet_gradients(a, grad);
            mass_matrix_tet(tet_volum(a[0], a[1], a[2], a[3]), M4);
            convection_matrix2(Uloc, grad, M4, A2loc);
        } else memset(A2loc, 0, sizeof(A2loc));

        // Divergence Matrix
        double grad[4][3];
        tet_gradients(a, grad);
        double vol = tet_volum(a[0], a[1], a[2], a[3]);
        divergence_matrix(grad, vol, Bloc);

        // Pressure Stabilization Matrix
        pressure_stabilization_matrix(a, delta, Dloc);

        // Assembly of the global matrix

        // Bloc velocity/velocity (M + A0 + A1 + A2)
        for (int I = 0; I < 12; ++I)
            for (int J = 0; J < 12; ++J) {
                int gi = vg[I], gj = vg[J];
                Aglob[gi*Ndof_tot + gj] += (Mloc[I][J]/dt) + A0loc[I][J] + A1loc[I][J] + A2loc[I][J];
            }
        // Bloc velocity/pressure (B^T)
        for (int I = 0; I < 12; ++I)
            for (int L = 0; L < 4; ++L) {
                int gi = vg[I], gl = pg[L];
                Aglob[gi*Ndof_tot + (Ndof_v + gl)] += Bloc[L][I];
            }
        // Bloc pressure/velocity (-B)
        for (int L = 0; L < 4; ++L)
            for (int J = 0; J < 12; ++J) {
                int gl = pg[L], gj = vg[J];
                Aglob[(Ndof_v + gl)*Ndof_tot + gj] -= Bloc[L][J];
            }
        // Bloc pressure/pressure (D)
        if (delta != 0.0) {
        for (int L = 0; L < 4; ++L)
            for (int K = 0; K < 4; ++K) {
                int gl = pg[L], gk = pg[K];
                Aglob[(Ndof_v + gl)*Ndof_tot + (Ndof_v + gk)] += Dloc[L][K];
            }
        }
    }
}

// // --- MAIN ---
// int main() {
//     // 1. Maillage : tétraèdre unitaire (4 nœuds, 1 élément)
//     int Nnodes = 4, Nelements = 1;
//     double nodes[4][3] = {
//         {0.0, 0.0, 0.0},
//         {1.0, 0.0, 0.0},
//         {0.0, 1.0, 0.0},
//         {0.0, 0.0, 1.0}
//     };
//     int elements[1][4] = { {0, 1, 2, 3} };

//     // 2. Champ vitesse arbitraire aux sommets (U[3][Nnodes])
//     double U[3][4] = {
//         {1.0, 2.0, 3.0, 4.0},      // Ux
//         {0.0, -1.0, 0.5, 2.0},     // Uy
//         {-2.0, 1.0, 0.0, -1.0}     // Uz
//     };

//     // 3. Paramètres
//     double dt = 0.01; // Pas de temps
//     double delta = 0.1;
//     double Re = 1.0; // Reynolds

//     // 4. Allocation de la matrice globale
//     int Ndof_v = 3*Nnodes, Ndof_p = Nnodes, Ndof_tot = Ndof_v + Ndof_p;
//     double *Aglob = calloc(Ndof_tot*Ndof_tot, sizeof(double));
//     if (!Aglob) { fprintf(stderr, "Erreur alloc Aglob\n"); return 1; }

//     // 5. Assemblage du système global
//     global_matrix(Nnodes, Nelements, elements, nodes, U, dt, delta, Re, Aglob);

//     // Affichage des blocs de la matrice globale
//     printf("Matrice globale (taille %d x %d):\n", Ndof_tot, Ndof_tot);
//     for (int i = 0; i < Ndof_tot; ++i) {
//         for (int j = 0; j < Ndof_tot; ++j)
//             printf("% .6f ", Aglob[i*Ndof_tot + j]);
//         printf("\n");
//     }

//     /*

//     // 6. Affichage du bloc vitesse/vitesse (M + A0 + A1 + A2)
//     printf("\nBloc vitesse/vitesse (12x12):\n");
//     for (int i = 0; i < 12; ++i) {
//         for (int j = 0; j < 12; ++j)
//             printf("% .6f ", Aglob[i*Ndof_tot + j]);
//         printf("\n");
//     }

//     // 7. Affichage du bloc vitesse/pression (B^T) [12 x 4]
//     printf("\nBloc vitesse/pression (12x4):\n");
//     for (int i = 0; i < 12; ++i) {
//         for (int j = 0; j < 4; ++j)
//             printf("% .6f ", Aglob[i*Ndof_tot + (Ndof_v + j)]);
//         printf("\n");
//     }

//     // 8. Affichage du bloc pression/vitesse (-B) [4 x 12]
//     printf("\nBloc pression/vitesse (4x12):\n");
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 12; ++j)
//             printf("% .6f ", Aglob[(Ndof_v + i)*Ndof_tot + j]);
//         printf("\n");
//     }

//     // 9. Affichage du bloc pression/pression (4x4)
//     printf("\nBloc pression/pression (4x4):\n");
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j)
//             printf("% .6f ", Aglob[(Ndof_v + i)*Ndof_tot + (Ndof_v + j)]);
//         printf("\n");
//     }
//     */

//     free(Aglob);
//     return 0;
// }