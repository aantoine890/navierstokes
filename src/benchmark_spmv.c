// benchmark_spmv_matrix_only.cpp
// Generates and assembles the global matrix in both AIJ and BAIJ formats
// from a tetrahedral mesh for the time-dependent Navier--Stokes equations.

#include <petscksp.h>
#include <petscdmplex.h>
#include <petscviewer.h>
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

static void read_mesh(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Cannot open .msh file: %s\n", fname);
        exit(1);
    }

    char buf[256];

    while (fgets(buf, sizeof buf, f)) if (strncmp(buf, "$Nodes", 6) == 0) break;
    fscanf(f, "%d", &nv);
    coord.resize(3*nv);
    for (int i=0;i<nv;++i) {
        int id; double x,y,z; fscanf(f, "%d %lf %lf %lf", &id, &x,&y,&z);
        coord[3*i+0]=x; coord[3*i+1]=y; coord[3*i+2]=z;
    }

    while (fgets(buf, sizeof buf, f)) if (strncmp(buf, "$Elements", 9) == 0) break;
    int totElem; fscanf(f, "%d", &totElem);
    long pos = ftell(f);
    ne = 0;
    for (int i=0;i<totElem;++i){int id,t,tp;fscanf(f,"%d %d %d",&id,&tp,&t);if(tp==4)++ne;fgets(buf,sizeof buf,f);}
    tet.resize(4*ne);

    fseek(f, pos, SEEK_SET);
    int k=0;
    for (int i=0;i<totElem;++i) {
        int id, type, ntags; fscanf(f, "%d %d %d", &id,&type,&ntags);
        for (int t=0;t<ntags;++t){int dummy;fscanf(f,"%d",&dummy);}
        if (type==4) {
            for(int j=0;j<4;++j){int v;fscanf(f,"%d",&v); tet[4*k+j]=v-1;} ++k;
        } else {
            fgets(buf,sizeof buf,f);
        }
    }
    fclose(f);
}

static void assemble_ns_matrix(Mat A, double Re, double delta)
{
    const double invRe = 1.0/Re;

    for (int k=0;k<ne;++k) {
        double a[4][3];
        for (int i=0;i<4;++i) {
            int gi=tet[4*k+i];
            a[i][0]=coord[3*gi+0];
            a[i][1]=coord[3*gi+1];
            a[i][2]=coord[3*gi+2];
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
        for (int i=0;i<4;++i) rows[i] = tet[4*k+i];

        double block[16];
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                memset(block,0,sizeof block);
                for (int a=0;a<3;++a)
                    for (int b=0;b<3;++b)
                        block[a*4+b] = em.A0loc[3*i+a][3*j+b] + em.Mloc[3*i+a][3*j+b];
                for (int a=0;a<3;++a)
                    block[a*4+3] = em.Bloc[j][3*i+a];
                for (int b=0;b<3;++b)
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

static double benchmark(Mat A, Vec x, Vec y, PetscErrorCode (*MatMult)(Mat, Vec, Vec), int niter)
{
    PetscLogDouble t1, t2;
    PetscTime(&t1);
    for (int i = 0; i < niter; ++i)
        MatMult(A, x, y);
    PetscTime(&t2);
    return (t2 - t1) / niter;
}

int main(int argc,char**argv)
{
    PetscInitialize(&argc,&argv,NULL,NULL);
    char msh[PETSC_MAX_PATH_LEN] = "";
    PetscOptionsGetString(NULL,NULL,"-msh",msh,sizeof msh,NULL);
    if (!*msh) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Use -msh <file.msh>");
    int niter = 50;
    PetscOptionsGetInt(NULL,NULL,"-niter",&niter,NULL);

    read_mesh(msh);
    PetscPrintf(PETSC_COMM_WORLD,"Mesh: %d nodes, %d tetrahedra\n",nv,ne);

    Mat A_baij, A_aij;

    // BAIJ format
    MatCreate(PETSC_COMM_WORLD,&A_baij);
    MatSetSizes(A_baij,PETSC_DECIDE,PETSC_DECIDE, DOF_PER_NODE*nv, DOF_PER_NODE*nv);
    MatSetType(A_baij,MATSEQBAIJ);
    MatSetBlockSize(A_baij, VBLOCK);
    MatSeqBAIJSetPreallocation(A_baij, VBLOCK, 60, NULL);
    MatSetUp(A_baij);
    assemble_ns_matrix(A_baij, 100.0, 0.05);
    // PetscViewer viewer_baij;
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD, "mat/matrix1_baij4.mtx", &viewer_baij);
    // PetscViewerPushFormat(viewer_baij, PETSC_VIEWER_ASCII_MATRIXMARKET);
    // MatView(A_baij, viewer_baij);
    // PetscViewerDestroy(&viewer_baij);
    MatInfo info_baij;
    PetscInt nrows;
    MatGetInfo(A_baij, MAT_GLOBAL_SUM, &info_baij);
    MatGetSize(A_baij, &nrows, NULL);

    // Paramètres pour le calcul
    PetscInt bs = 4;
    PetscInt nblock_rows = nrows / bs;
    PetscInt nblocks = (PetscInt)(info_baij.nz_used / (bs * bs));

    size_t size_values = nblocks * bs * bs * sizeof(PetscScalar);
    size_t size_colidx = nblocks * sizeof(PetscInt);
    size_t size_rowptr = (nblock_rows + 1) * sizeof(PetscInt);
    size_t size_x = DOF_PER_NODE * nv * sizeof(PetscScalar);
    size_t size_y = DOF_PER_NODE * nv * sizeof(PetscScalar);
    size_t total_size_bytes = size_values + size_colidx + size_rowptr + size_x + size_y;

    PetscPrintf(PETSC_COMM_WORLD,
        "\n[BAIJ 4x4] matrix assembled, nnz= %.0f.\n", info_baij.nz_used);
    PetscPrintf(PETSC_COMM_WORLD,
        "[BAIJ 4x4] Estimated memory = %.2f MiB\n", total_size_bytes / (1024.0 * 1024.0));

    // Convert to AIJ
    MatConvert(A_baij, MATSEQAIJ, MAT_INITIAL_MATRIX, &A_aij);
    PetscViewer viewer_aij;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "mat/matrix10_aij.mtx", &viewer_aij);
    PetscViewerPushFormat(viewer_aij, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(A_aij, viewer_aij);
    PetscViewerDestroy(&viewer_aij);
    MatInfo info_aij;
    PetscInt nrows_aij;
    MatGetInfo(A_aij, MAT_GLOBAL_SUM, &info_aij);
    MatGetSize(A_aij, &nrows_aij, NULL);

    // Taille mémoire estimée (valeurs typiques : Scalar=8, Int=4 ou 8)
    size_values = (size_t)(info_aij.nz_used) * sizeof(PetscScalar);
    size_colidx = (size_t)(info_aij.nz_used) * sizeof(PetscInt);
    size_rowptr = (nrows_aij + 1) * sizeof(PetscInt);
    size_x = DOF_PER_NODE * nv * sizeof(PetscScalar);
    size_y = DOF_PER_NODE * nv * sizeof(PetscScalar);
    total_size_bytes = size_values + size_colidx + size_rowptr + size_x + size_y;

    PetscPrintf(PETSC_COMM_WORLD,
        "[AIJ] matrix generated from BAIJ, nnz= %.0f.\n", info_aij.nz_used);
    PetscPrintf(PETSC_COMM_WORLD,
        "[AIJ] Estimated memory = %.2f MiB\n", total_size_bytes / (1024.0 * 1024.0));

    Vec x, y;
    VecCreateSeq(PETSC_COMM_SELF, DOF_PER_NODE*nv, &x);
    VecDuplicate(x, &y);
    VecSet(x, 1.0);

    MatInfo info;
    MatGetInfo(A_aij, MAT_GLOBAL_SUM, &info);
    PetscInt nnz = (PetscInt)info.nz_used;

    struct Variant {
        const char* name;
        Mat mat;
        PetscErrorCode (*fun)(Mat, Vec, Vec);
    } variants[] = {
        {"aij_mad", A_aij, MatMult_SeqAIJ},
        {"aij_fma", A_aij, MatMult_SeqAIJ_FMA},
        {"baij4_mad", A_baij, MatMult_SeqBAIJ_4},
        {"baij4_fma", A_baij, MatMult_SeqBAIJ_4_FMA},
        {"baij4_avx2", A_baij, MatMult_SeqBAIJ_4_AVX2},
    };

    PetscPrintf(PETSC_COMM_WORLD,"\n[Benchmarking SpMV variants] (%d iterations)\n", niter);
    PetscPrintf(PETSC_COMM_WORLD,"%-16s  Time (ms)  GFLOP/s\n", "Variant");
    for (auto& v : variants) {
        double t = benchmark(v.mat, x, y, v.fun, niter);
        double gflops = 2.0 * nnz / t / 1e9;
        PetscPrintf(PETSC_COMM_WORLD,"%-16s  %8.3f   %8.3f\n", v.name, 1e3 * t, gflops);
    }

    MatDestroy(&A_aij);
    MatDestroy(&A_baij);
    VecDestroy(&x);
    VecDestroy(&y);
    PetscFinalize();
    return 0;
}