// Matrix Powers Kernel for a Sparse matrix to increase local access of
// the coefficient based on finite element connectivity
// Copyright 19 Jun. A. Suzuki

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <list>
#include <immintrin.h>
#include <cmath>
#include <algorithm>

struct csrmatrix {
  int n, nnz;
  std::vector<int> ptrow;
  std::vector<int> indcol;  
  std::vector<double> coef;
};

void Generate1stlayer(std::vector<int> &ptrowend1,
		      csrmatrix &A)
{
  int nrow = A.n;
  std::vector<int> mask(nrow, 0);
  ptrowend1.resize(A.nnz);
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      if (mask[j]) {
	ptrowend1[ia] = A.ptrow[j];
      }
      else {
	ptrowend1[ia] = A.ptrow[j + 1];
	mask[j] = 1;	
      }
    }
  }
}

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V0(double *z, double *y,
	   double *x, csrmatrix &A, std::vector<int> &ptrowend1)
{
  int nrow = A.n;
  for (int i = 0; i < nrow; i++) {
    z[i] = y[i] = 0.0;
  }
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	y[j] += A.coef[jb] * x[k];
      } // loop : jb
      z[i] += A.coef[ia] * y[j];
    } // loop : ia
  }   // loop : i
}

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM2V(double *z, double *y, double *x, csrmatrix &A, std::vector<int> &ptrowend1) {
    const int nrow = A.n;
    std::fill_n(y, nrow, 0.0);
    std::fill_n(z, nrow, 0.0);

    // Précalcul des pointeurs pour un accès plus rapide
    const int* ptrow = A.ptrow.data();
    const int* indcol = A.indcol.data();
    const double* coef = A.coef.data();
    const int* pend = ptrowend1.data();

    for (int i = 0; i < nrow; i++) {
        const int row_start = ptrow[i];
        const int row_end = ptrow[i + 1];
        
        for (int ia = row_start; ia < row_end; ia++) {
            const int j = indcol[ia];
            const double a_ij = coef[ia];
            double sum = 0.0;
            
            const int inner_start = ptrow[j];
            const int inner_end = pend[ia];
            
            // Déroulage de boucle manuel (loop unrolling)
            int jb = inner_start;
            for (; jb + 3 < inner_end; jb += 4) {
                sum += coef[jb] * x[indcol[jb]] +
                       coef[jb+1] * x[indcol[jb+1]] +
                       coef[jb+2] * x[indcol[jb+2]] +
                       coef[jb+3] * x[indcol[jb+3]];
            }
            // for (; jb < inner_end; jb++) {
            //     sum += coef[jb] * x[indcol[jb]];
            // }
            
            y[j] += sum;
            z[i] += a_ij * y[j];
        }
    }
}

void Generate2ndlayer(std::vector<std::vector<int> >&ptrowend2,
		      csrmatrix &A,
		      std::vector<int> &ptrowend1)
{
  int nrow = A.n;
  std::vector<int> mask2(nrow, 0);  
  ptrowend2.resize(A.nnz);  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      ptrowend2[ia].resize(ptrowend1[ia] - A.ptrow[j]);
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	if (mask2[k]) {
	  ptrowend2[ia][jjb] = A.ptrow[k];
	}
	else {
	  ptrowend2[ia][jjb] = A.ptrow[k + 1];
	  mask2[k] = 1;
	}
      } // loop : jb
    } // loop : ia
  }  // loop : i
}

void SpM3V(double *w, double *z, double *y,
	   double *x, csrmatrix &A, std::vector<int> &ptrowend1,
	   std::vector<std::vector<int> > &ptrowend2)	   
{
  int nrow = A.n;  
  for (int i = 0; i < nrow; i++) {
    w[i] = z[i] = y[i] = 0.0;
  }  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  y[k] += A.coef[kc] * x[l];
	}
	z[j] += A.coef[jb] * y[k];	
      } // loop : jb
      w[i] += A.coef[ia] * z[j];
    } // loop : ia
  }   // loop : i
}

void Generate3rdlayer(std::vector<std::vector<std::vector<int> > > &ptrowend3,
		      csrmatrix &A,
		      std::vector<int> &ptrowend1,
		      std::vector<std::vector<int> > &ptrowend2)
{
  int nrow = A.n;
  std::vector<int> mask3(nrow, 0);  
  ptrowend3.resize(A.nnz);  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      ptrowend3[ia].resize(ptrowend1[ia] - A.ptrow[j]);
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  ptrowend3[ia][jjb].resize(ptrowend2[ia][jjb] - A.ptrow[k]);
	  int kkc = kc - A.ptrow[k];
	  if (mask3[l]) {
	    ptrowend3[ia][jjb][kkc] = A.ptrow[l];
	  }
	  else {
	    ptrowend3[ia][jjb][kkc] = A.ptrow[l + 1];	  
	    mask3[l] = 1;
	  }
	}
      } // loop : jb
    } // loop : ia
  }  // loop : i
}

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpM4V(double *v, double *w, double *z, double *y,
	   double *x, csrmatrix &A, std::vector<int> &ptrowend1,
	   std::vector<std::vector<int> > &ptrowend2,
	   std::vector<std::vector<std::vector<int> > > &ptrowend3)
	   
{
  int nrow = A.n;  
  for (int i = 0; i < nrow; i++) {
    v[i] = w[i] = z[i] = y[i] = 0.0;
  }  
  for (int i = 0; i < nrow; i++) {
    for (int ia = A.ptrow[i]; ia < A.ptrow[i + 1]; ia++) {
      int j = A.indcol[ia];
      for (int jb = A.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = A.indcol[jb];
	int jjb = jb - A.ptrow[j];
	for (int kc = A.ptrow[k]; kc < ptrowend2[ia][jjb]; kc++) {
	  int l = A.indcol[kc];
	  int kkc = kc - A.ptrow[k];
	  for (int ld = A.ptrow[l]; ld < ptrowend3[ia][jjb][kkc]; ld++) {
	    int m = A.indcol[ld];	
	    y[l] += A.coef[ld] * x[m];
	  }
	  z[k] += A.coef[kc] * y[l];	
	} // loop : jb
	w[j] += A.coef[jb] * z[k];
      }
      v[i] += A.coef[ia] * w[j];      
    } // loop : ia
  }   // loop : i
}

void generate_CSR(std::list<int>* ind_cols_tmp, std::list<double>* val_tmp, 
		  int nrow, int nnz, 
		  int *irow, int *jcol, double* val)
{
  for (int i = 0; i < nnz; i++) {
    const int ii = irow[i];
    const int jj = jcol[i];
    if (ind_cols_tmp[ii].empty()) {
      ind_cols_tmp[ii].push_back(jj);
      val_tmp[ii].push_back(val[i]);
    }
    else {
      if (ind_cols_tmp[ii].back() < jj) {
	ind_cols_tmp[ii].push_back(jj);
	val_tmp[ii].push_back(val[i]);
      }
      else {
	std::list<double>::iterator iv = val_tmp[ii].begin();
	std::list<int>::iterator it = ind_cols_tmp[ii].begin();
	for ( ; it != ind_cols_tmp[ii].end(); ++it, ++iv) {
	  if (*it == jj) {
	      break;
	  }
	  if (*it > jj) {
	    ind_cols_tmp[ii].insert(it, jj);
	    val_tmp[ii].insert(iv, val[i]);
	    break;
	  }
	}
      }
    }
  }
}

__attribute__((optimize("O3")))
__attribute__((target("no-sse,no-avx2,no-fma")))
void SpMV(double *y, double *x, csrmatrix &a)
{
  int nrow = a.n;
  const double zero(0.0);
  for (int i = 0; i < nrow; i++) {
    y[i] = zero;
    for (int ia = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++) {
      int j = a.indcol[ia];
      y[i] += a.coef[ia] * x[j];
    }
  }
}


void COO2CSR(csrmatrix &a, int nrow, int nnz, int *irow, int *jcol, double *val)
{
  a.n = nrow;
  a.nnz = nnz;
  a.ptrow.resize(nrow + 1);
  a.indcol.resize(nnz);
  a.coef.resize(nnz);
  std::vector<std::list<int> > ind_cols_tmp(nrow);
  std::vector<std::list<double> > val_tmp(nrow);

  // without adding diagonal nor symmetrize for PARDISO mtype = 11
  generate_CSR(&ind_cols_tmp[0], &val_tmp[0], 
               nrow, nnz,
               &irow[0], &jcol[0], &val[0]);
  {
    int k = 0;
    a.ptrow[0] = 0;
    for (int i = 0; i < nrow; i++) {
      std::list<int>::iterator jt = ind_cols_tmp[i].begin();
      std::list<double>::iterator jv = val_tmp[i].begin();     
      for ( ; jt != ind_cols_tmp[i].end(); ++jt, ++jv) {
        a.indcol[k] = (*jt);
        a.coef[k] = (*jv);
        k++;
      }
      a.ptrow[i + 1] = k;
    }
  }
}

double norm2(const std::vector<double> &x) {
    double s = 0.0;
    for (double xi : x) s += xi * xi;
    return std::sqrt(s);
}

double rel_error(const std::vector<double> &ref, const std::vector<double> &test) {
    double s = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        s += (ref[i] - test[i]) * (ref[i] - test[i]);
    return std::sqrt(s) / norm2(ref);
}


int main(int argc, char **argv)
{
  char fname[256];
  char buf[1024];
  int nrow, nnz;

  FILE *fp;
  int itmp, jtmp, ktmp;
  
  strcpy(fname, argv[1]);
  
  if ((fp = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "fail to open %s\n", fname);
  }
  fgets(buf, 256, fp);
  
  while (1) {
    fgets(buf, 256, fp);
    if (buf[0] != '%') {
      sscanf(buf, "%d %d %d", &itmp, &jtmp, &ktmp);
      nrow = itmp;
      nnz = ktmp;
      break;
    }
  }
  std::vector<int> irow(nnz);
  std::vector<int> jcol(nnz);
  std::vector<double>  val(nnz);
  {
    int ii = 0;
    int itmp, jtmp;
    float vtmp;
    for (int i = 0; i < nnz; i++) {
      fscanf(fp, "%d\t%d\t%f", &itmp, &jtmp, &vtmp);
      irow[ii] = itmp - 1; // zero based
      jcol[ii] = jtmp - 1; // zero based
      val[ii] = (double)vtmp;
      ii++;
    }
    fprintf(stderr, "%d\n", ii);
    nnz = ii;
  }
  
  fclose (fp);
  
  csrmatrix a;
  COO2CSR(a, nrow, nnz, &irow[0], &jcol[0], &val[0]);

  std::vector<double> x(nrow, 1.0), b(nrow);
  std::vector<double> x1(nrow), x2(nrow), x3(nrow), x4(nrow);
  std::vector<double> y1(nrow), y2(nrow), y3(nrow), y4(nrow);
  
  SpMV(&b[0], &x[0], a); // b = A * x
  SpMV(&x1[0], &b[0], a); // x1 = A * b
  SpMV(&x2[0], &x1[0], a); // x2 = A * x1 = A * A * b
  SpMV(&x3[0], &x2[0], a); // x3 = A * x2 = A * A * A * b
  SpMV(&x4[0], &x3[0], a); // x4 = A x3 = A * A * A * A * b    

  std::vector<int> ptrowend1(nnz);
  std::vector<std::vector<int> > ptrowend2(nnz);
  std::vector<std::vector<std::vector<int> > > ptrowend3(nnz);    

  Generate1stlayer(ptrowend1, a);
  SpM2V(&y2[0], &y1[0], &b[0], a, ptrowend1);

  for (int i = 0; i < nrow; i++) {
    fprintf(stderr, "%d : %g %g : %g %g \n", i, x1[i], y1[i], x2[i], y2[i]);
  }

  Generate2ndlayer(ptrowend2, a, ptrowend1);
#if 0
  fprintf(stderr, "%s %d\n", __FILE__, __LINE__);
  for (int i = 0; i < nrow; i++) {
    for (int ia  = a.ptrow[i]; ia < a.ptrow[i + 1]; ia++) {
      int j = a.indcol[ia];
      fprintf(stderr, "%d %d : %d %d %d\n", i, j, a.ptrow[j], a.ptrow[j + 1], ptrowend1[ia]);
      for (int jb = a.ptrow[j]; jb < ptrowend1[ia]; jb++) {
	int k = a.indcol[jb];
	int jjb = jb - a.ptrow[j];
	fprintf(stderr, "  %d %d %d : %d %d %d\n", i, j, k, a.ptrow[k], a.ptrow[k + 1], ptrowend2[ia][jjb]);	
      }
    }
  }
#endif
  SpM3V(&y3[0], &y2[0], &y1[0], &b[0], a, ptrowend1, ptrowend2);
  
  for (int i = 0; i < nrow; i++) {
    fprintf(stderr, ": %d : %g %g : %g %g : %g %g\n",
	    i, x1[i], y1[i], x2[i], y2[i], x3[i], y3[i]);
  }

  Generate3rdlayer(ptrowend3, a, ptrowend1, ptrowend2);  
  
  SpM4V(&y4[0], &y3[0], &y2[0], &y1[0], &b[0],
        a, ptrowend1, ptrowend2, ptrowend3);

  for (int i = 0; i < nrow; i++) {
    fprintf(stderr, ":: %d : %g %g : %g %g : %g %g : %g %g\n",
            i, x1[i], y1[i], x2[i], y2[i], x3[i], y3[i], x4[i], y4[i]);
  }

}