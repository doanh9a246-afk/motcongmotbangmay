# Makefile - N-body OpenMP + CUDA
#   make            -> bin/nbody_cpu  (tuan tu + OpenMP)
#   make cuda       -> bin/nbody_cuda (can CUDA Toolkit / nvcc)
#   make all        -> ca hai
#   make ARCH=sm_75 cuda   (chi dinh kien truc GPU neu nvcc cu khong ho tro -arch=native)

CC     ?= gcc
NVCC   ?= nvcc
ARCH   ?= native
CFLAGS ?= -O3 -march=native -ffast-math -Wall -Wextra
NVFLAGS ?= -O3 -arch=$(ARCH) -use_fast_math -Xcompiler -fopenmp,-O3

ifeq ($(OS),Windows_NT)
  EXE := .exe
endif

CPU_BIN  := bin/nbody_cpu$(EXE)
CUDA_BIN := bin/nbody_cuda$(EXE)
HDRS     := src/nbody_common.h src/nbody_cpu_kernels.h

.PHONY: cpu cuda all clean report slide docs
.DEFAULT_GOAL := cpu

# Bao cao va slide LaTeX (can XeLaTeX: MiKTeX hoac TeX Live, font Times New Roman/Consolas/Arial)
LATEX_DIR := bao_cao/latex
XELATEX   := xelatex -interaction=nonstopmode -halt-on-error

report:
	cd $(LATEX_DIR) && $(XELATEX) bao_cao.tex && $(XELATEX) bao_cao.tex && $(XELATEX) bao_cao.tex
	cp $(LATEX_DIR)/bao_cao.pdf bao_cao/bao_cao_nhom.pdf

slide:
	cd $(LATEX_DIR) && $(XELATEX) slide.tex && $(XELATEX) slide.tex
	cp $(LATEX_DIR)/slide.pdf bao_cao/slide_nhom.pdf

docs: report slide

cpu: $(CPU_BIN)
cuda: $(CUDA_BIN)
all: cpu cuda

$(CPU_BIN): src/nbody_cpu.c $(HDRS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp $< -o $@ -lm

$(CUDA_BIN): src/nbody_cuda.cu $(HDRS)
	@mkdir -p bin
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -rf bin
