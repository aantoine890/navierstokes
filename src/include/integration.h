#ifndef INTEGRATION_H
#define INTEGRATION_H

// Volume d'un tétraèdre (coords explicites)
double tet_volum(const double a0[3], const double a1[3], const double a2[3], const double a3[3]);

// Diamètre du tétraèdre (longueur max entre sommets)
double tet_diameter(const double a[4][3]);

// Gradients des fonctions de base P1 sur le tétraèdre
void tet_gradients(const double a[4][3], double grad[4][3]);

// Matrice de masse vectorielle (bloc 12x12)
void mass_matrix(const double a0[3], const double a1[3], const double a2[3], const double a3[3], double M[12][12]);

// Matrice de masse scalaire P1 (bloc 4x4)
void mass_matrix_tet(double vol, double M[4][4]);

// Matrice de diffusion (bloc 12x12)
void diffusion_matrix(const double a[4][3], double Re, double A0[12][12]);

// Matrice convection linéarisée (bloc 12x12)
void convection_matrix1(const double a[4][3], const double U[3][4], double A1[12][12]);

// Matrice convection non-linéaire (bloc 12x12)
void convection_matrix2(const double U[3][4], const double grad[4][3], const double M[4][4], double A2[12][12]);

// Bloc de divergence (bloc 4x12)
void divergence_matrix(const double grad[4][3], double vol, double Bk[4][12]);

// Stabilisation pression (bloc 4x4)
void pressure_stabilization_matrix(const double a[4][3], double delta, double D[4][4]);

// Assemblage global (pour petit maillage, version full dense)
void global_matrix(int Nnodes, int Nelements, const int elements[][4], const double nodes[][3],
                   const double U[3][4], double dt, double delta, double Re, double *Aglob);


#endif // INTEGRATION_H