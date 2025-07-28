#include "SpMV.h"

/* ----------- Matrix Handler ----------- */

void generate_CSR(std::list<int> *ind_cols_tmp, std::list<double> *val_tmp, int nrow, int nnz, int *irow, int *jcol, double *val)
{
	for (int i = 0; i < nnz; i++)
	{
		const int ii = irow[i];
		const int jj = jcol[i];
		if (ind_cols_tmp[ii].empty())
		{
			ind_cols_tmp[ii].push_back(jj);
			val_tmp[ii].push_back(val[i]);
		}
		else
		{
			if (ind_cols_tmp[ii].back() < jj)
			{
				ind_cols_tmp[ii].push_back(jj);
				val_tmp[ii].push_back(val[i]);
			}
			else
			{
				std::list<double>::iterator iv = val_tmp[ii].begin();
				std::list<int>::iterator it = ind_cols_tmp[ii].begin();
				for (; it != ind_cols_tmp[ii].end(); ++it, ++iv)
				{
					if (*it == jj)
					{
						break;
					}
					if (*it > jj)
					{
						ind_cols_tmp[ii].insert(it, jj);
						val_tmp[ii].insert(iv, val[i]);
						break;
					}
				}
			}
		}
	}
}

void generate_BCSR4(std::list<std::pair<int, std::array<double, 16>>>* block_rows,
                    int nrow, int nnz, const int* irow, const int* jcol, const double* val,
                    bcsr4x4_matrix& A)
{
	const int nblocks = nrow / 4;

	// Remplissage des blocs
	for (int k = 0; k < nnz; ++k)
	{
		int i = irow[k], j = jcol[k];
		double v = val[k];
		int bi = i / 4, bj = j / 4;
		int ii = i % 4, jj = j % 4;
		bool found = false;

		for (auto& pair : block_rows[bi])
		{
			if (pair.first == bj)
			{
				pair.second[4 * ii + jj] = v;
				found = true;
				break;
			}
		}
		if (!found)
		{
			std::array<double, 16> new_block = {};
			new_block[4 * ii + jj] = v;
			block_rows[bi].emplace_back(bj, new_block);
		}
	}

	// Conversion vers format final
	A.nblocks = 0;
	A.ptrow.resize(nblocks + 1, 0);
	A.indcol.clear();
	A.coef.clear();

	for (int bi = 0; bi < nblocks; ++bi)
	{
		A.ptrow[bi + 1] = A.ptrow[bi] + block_rows[bi].size();
		for (const auto& pair : block_rows[bi])
		{
			A.indcol.push_back(pair.first);
			const std::array<double, 16>& block = pair.second;
			A.coef.insert(A.coef.end(), block.begin(), block.end());  // flattening
		}
	}

	A.nrows = nblocks;
}

void COO2CSR(csrmatrix &a, int nrow, int nnz, int *irow, int *jcol, double *val)
{
	a.n = nrow;
	a.nnz = nnz;
	a.ptrow.resize(nrow + 1);
	a.indcol.resize(nnz);
	a.coef.resize(nnz);
	std::vector<std::list<int>> ind_cols_tmp(nrow);
	std::vector<std::list<double>> val_tmp(nrow);

	// without adding diagonal nor symmetrize for PARDISO mtype = 11
	generate_CSR(&ind_cols_tmp[0], &val_tmp[0],
		     nrow, nnz,
		     &irow[0], &jcol[0], &val[0]);
	{
		int k = 0;
		a.ptrow[0] = 0;
		for (int i = 0; i < nrow; i++)
		{
			std::list<int>::iterator jt = ind_cols_tmp[i].begin();
			std::list<double>::iterator jv = val_tmp[i].begin();
			for (; jt != ind_cols_tmp[i].end(); ++jt, ++jv)
			{
				a.indcol[k] = (*jt);
				a.coef[k] = (*jv);
				k++;
			}
			a.ptrow[i + 1] = k;
		}
	}
}

/* ----------- Error Checking ----------- */

double norm2(const std::vector<double> &x) {
	double s = 0.0;
	for (double xi : x)
		s += xi * xi;
	return std::sqrt(s);
}

double rel_error(const std::vector<double> &ref, const std::vector<double> &test) {
	double s = 0.0;
	for (size_t i = 0; i < ref.size(); ++i)
		s += (ref[i] - test[i]) * (ref[i] - test[i]);
	return std::sqrt(s) / norm2(ref);
}


void flush_cache() {
    constexpr size_t CACHE_FLUSH_SIZE = 300 * 1024 * 1024; // 100 MiB > L3 cache
    static std::vector<char> dummy_buffer(CACHE_FLUSH_SIZE);
    volatile char sink = 0;
    for (size_t i = 0; i < dummy_buffer.size(); ++i) {
        dummy_buffer[i] = static_cast<char>(i);
        sink ^= dummy_buffer[i]; 
    }
}