#include <petscksp.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "integration.h"
#include "kernels.h"

struct ElementMatrices {
    double Mloc[12][12];
    double A0loc[12][12];
    double A1loc[12][12];
    double A2loc[12][12];
    double Bloc[4][12];
    double Dloc[4][4];
    double vol;
    double grad[4][3];
};

static int nv = 0;
static int ne = 0;
static std::vector<int> tet;
static std::vector<double> coord;

static const int DOF_PER_NODE = 4;
static const int VBLOCK = 4;

static void read_mesh(const char *fname) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Cannot open .msh file: %s\n", fname);
        exit(1);
    }

    char buf[256];
    while (fgets(buf, sizeof buf, f)) if (strncmp(buf, "$Nodes", 6) == 0) break;
    fscanf(f, "%d", &nv);
    coord.resize(3*nv);
    for (int i=0; i<nv; ++i) {
        int id; double x,y,z; 
        fscanf(f, "%d %lf %lf %lf", &id,&x,&y,&z);
        coord[3*i+0]=x; coord[3*i+1]=y; coord[3*i+2]=z;
    }

    while (fgets(buf, sizeof buf, f)) if (strncmp(buf, "$Elements", 9) == 0) break;
    int totElem; fscanf(f, "%d", &totElem);
    long pos = ftell(f);
    ne = 0;
    for (int i=0; i<totElem; ++i) {
        int id,tp,t; 
        fscanf(f,"%d %d %d",&id,&tp,&t);
        if(tp==4) ++ne;
        fgets(buf,sizeof buf,f);
    }
    tet.resize(4*ne);

    fseek(f, pos, SEEK_SET);
    int k=0;
    for (int i=0; i<totElem; ++i) {
        int id, type, ntags; 
        fscanf(f, "%d %d %d", &id,&type,&ntags);
        for (int t=0; t<ntags; ++t) {int dummy; fscanf(f,"%d",&dummy);}
        if (type==4) {
            for(int j=0; j<4; ++j) {
                int v; fscanf(f,"%d",&v); 
                tet[4*k+j]=v-1;
            } 
            ++k;
        } else {
            fgets(buf,sizeof buf,f);
        }
    }
    fclose(f);
}

static void assemble_ns_matrix(Mat A, double Re, double delta) {
    const double invRe = 1.0/Re;

    for (int k=0; k<ne; ++k) {
        double a[4][3];
        for (int i=0; i<4; ++i) {
            int gi = tet[4*k+i];
            a[i][0] = coord[3*gi+0];
            a[i][1] = coord[3*gi+1];
            a[i][2] = coord[3*gi+2];
        }

        ElementMatrices em{};
        mass_matrix(a[0],a[1],a[2],a[3],em.Mloc);
        diffusion_matrix(a, Re, em.A0loc);
        double grad[4][3];
        tet_gradients(a, grad);
        double vol = tet_volum(a[0],a[1],a[2],a[3]);
        memcpy(em.grad, grad, sizeof grad);
        em.vol = vol;
        divergence_matrix(grad, vol, em.Bloc);
        pressure_stabilization_matrix(a, delta, em.Dloc);

        PetscInt rows[4];
        for (int i=0; i<4; ++i) rows[i] = tet[4*k+i];

        double block[16];
        for (int i=0; i<4; ++i) {
            for (int j=0; j<4; ++j) {
                memset(block, 0, sizeof block);
                for (int a=0; a<3; ++a)
                    for (int b=0; b<3; ++b)
                        block[a*4+b] = em.A0loc[3*i+a][3*j+b] + em.Mloc[3*i+a][3*j+b];
                for (int a=0; a<3; ++a)
                    block[a*4+3] = em.Bloc[j][3*i+a];
                for (int b=0; b<3; ++b)
                    block[3*4+b] = -em.Bloc[i][3*j+b];
                block[15] = em.Dloc[i][j];

                PetscInt col = rows[j];
                MatSetValuesBlocked(A, 1, &rows[i], 1, &col, block, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

PetscErrorCode MyShellSolve(KSP ksp, Vec b, Vec x) {
    PetscFunctionBeginUser;
    
    PetscPrintf(PETSC_COMM_WORLD, "==========================================\n");
    PetscPrintf(PETSC_COMM_WORLD, "  Custom KSP Shell Solver Activated\n");
    PetscPrintf(PETSC_COMM_WORLD, "==========================================\n");
    
    // Get operator matrix
    Mat A;
    KSPGetOperators(ksp, &A, NULL);
    
    // Get matrix size
    PetscInt m, n;
    MatGetSize(A, &m, &n);
    PetscPrintf(PETSC_COMM_WORLD, " - Solving system of size: %" PetscInt_FMT " x %" PetscInt_FMT "\n", m, n);
    
    // Simple placeholder: x = b (just for testing)
    VecCopy(b, x);
    
    PetscPrintf(PETSC_COMM_WORLD, " - Using trivial solution x = b\n");
    PetscPrintf(PETSC_COMM_WORLD, "==========================================\n");
    
    PetscFunctionReturn(0);
}

typedef struct {
    PetscInt dummy;  // Vous pouvez ajouter des champs ici
} MyKSPContext;

PetscErrorCode MyKSPCreate(PC pc, void **ctx) {
    PetscFunctionBeginUser;
    PetscCall(PetscNew(ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode MyKSPDestroy(void *ctx) {
    PetscFunctionBeginUser;
    PetscCall(PetscFree(ctx));
    PetscFunctionReturn(0);
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    char msh[PETSC_MAX_PATH_LEN] = "";
    PetscOptionsGetString(NULL, NULL, "-msh", msh, sizeof(msh), NULL);
    if (!*msh) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Use -msh <file.msh>");
    read_mesh(msh);

    // Create matrix
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, DOF_PER_NODE*nv, DOF_PER_NODE*nv);
    MatSetType(A, MATSEQBAIJ);
    MatSetBlockSize(A, VBLOCK);
    MatSeqBAIJSetPreallocation(A, VBLOCK, 60, NULL);
    MatSetUp(A);
    assemble_ns_matrix(A, 100.0, 0.05);

    // Create vectors
    Vec b, x;
    VecCreateSeq(PETSC_COMM_WORLD, DOF_PER_NODE*nv, &b);
    VecDuplicate(b, &x);
    VecSet(b, 1.0);  // Constant RHS for testing

    // Create KSP solver
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    
    // Set as shell solver
    KSPSetType(ksp, KSPSHELL);
    KSPSetOperators(ksp, A, A);
    
    // Set our custom solve function
    KSPShellSetContext(ksp, NULL);  // Vous pouvez passer un contexte ici
    KSPShellSetSolve(ksp, MyShellSolve);
    
    // Configure solver
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    // Solve system
    PetscPrintf(PETSC_COMM_WORLD, "Solving system with custom shell solver...\n");
    KSPSolve(ksp, b, x);

    // Cleanup
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
    KSPDestroy(&ksp);

    PetscFinalize();
    return 0;
}