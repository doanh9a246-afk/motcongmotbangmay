# Makefile - N-body OpenMP + CUDA
#   make            -> bin/nbody_cpu  (tuan tu + OpenMP)
#   make cuda       -> bin/nbody_cuda (can CUDA Toolkit / nvcc)
#   make all        -> ca hai
#   make ARCH=sm_75 cuda   (chi dinh kien truc GPU neu nvcc cu khong ho tro -arch=native)

ifeq ($(origin CC),default)
  CC := gcc
endif
NVCC   ?= nvcc
ARCH   ?= native
CFLAGS ?= -O3 -march=native -ffast-math -Wall -Wextra
NVFLAGS ?= -O3 -arch=$(ARCH) -use_fast_math -Xcompiler -fopenmp,-O3

ifeq ($(OS),Windows_NT)
  EXE := .exe
  ifeq ($(MSYSTEM),)
    MKDIR_BIN := if not exist bin mkdir bin
    RM_BIN    := if exist bin rmdir /s /q bin
  else
    MKDIR_BIN := mkdir -p bin
    RM_BIN    := rm -rf bin
  endif
else
  MKDIR_BIN := mkdir -p bin
  RM_BIN    := rm -rf bin
endif

CPU_BIN  := bin/nbody_cpu$(EXE)
CUDA_BIN := bin/nbody_cuda$(EXE)
HDRS     := src/nbody_common.h src/nbody_cpu_kernels.h

.PHONY: cpu cuda all clean

cpu: $(CPU_BIN)
cuda: $(CUDA_BIN)
all: cpu cuda

$(CPU_BIN): src/nbody_cpu.c $(HDRS)
	@$(MKDIR_BIN)
	$(CC) $(CFLAGS) -fopenmp $< -o $@ -lm

$(CUDA_BIN): src/nbody_cuda.cu $(HDRS)
	@$(MKDIR_BIN)
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	$(RM_BIN)
