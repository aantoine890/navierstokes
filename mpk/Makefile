# Executables
TARGETS = spmv 2spmv spm2v

# Fichiers source
SPMV_SRC   = SpMV.cpp
2SPMV_SRC  = 2SpMV.cpp
SPM2V_SRC  = SpM2V.cpp
UTILS_SRC  = utils.cpp

# Compilateur et options
CXX = g++
CXXFLAGS = -O3 -march=native -std=c++11
LDFLAGS = 

# Cibles principales
all: $(TARGETS)

spmv: $(SPMV_SRC) $(UTILS_SRC) SpMV.h
	$(CXX) $(CXXFLAGS) -o $@ $(SPMV_SRC) $(UTILS_SRC) $(LDFLAGS)

2spmv: $(2SPMV_SRC) $(SPMV_SRC) $(UTILS_SRC) SpMV.h
	$(CXX) $(CXXFLAGS) -o $@ $(2SPMV_SRC) $(SPMV_SRC) $(UTILS_SRC) $(LDFLAGS)

spm2v: $(SPM2V_SRC) $(UTILS_SRC) SpMV.h
	$(CXX) $(CXXFLAGS) -o $@ $(SPM2V_SRC) $(UTILS_SRC) $(LDFLAGS)

# Nettoyage
clean:
	rm -f $(TARGETS) *.o