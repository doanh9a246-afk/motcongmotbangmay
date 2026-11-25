/*
 * nbody_common.h - Phan dung chung cho ban CPU (tuan tu / OpenMP) va ban CUDA.
 *
 * Mo hinh: N vat the tuong tac hap dan voi nhau (all-pairs, O(N^2)),
 * don vi N-body chuan (G = 1, tong khoi luong = 1), co "softening" eps
 * de tranh ky di khi hai vat the qua gan nhau:
 *
 *      a_i = sum_j  m_j * r_ij / (|r_ij|^2 + eps^2)^(3/2)
 *
 * Tich phan theo thoi gian: Euler doi xung (symplectic Euler)
 *      v <- v + a*dt ;  p <- p + v*dt
 *
 * Header nay viet de bien dich duoc ca bang C (gcc) lan C++ (nvcc host).
 */
#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NB_PI 3.14159265358979323846

/* Bo cuc giong het float4 cua CUDA: (x, y, z, w). Voi mang vi tri, w = khoi luong. */
typedef struct { float x, y, z, w; } vec4f;

typedef struct {
    int n;                 /* so vat the */
    int steps;             /* so buoc mo phong */
    float dt;              /* buoc thoi gian */
    float eps;             /* softening length */
    unsigned long long seed;
    const char *mode;      /* CPU: serial | omp     CUDA: naive | tiled */
    int threads;           /* so luong OpenMP (0 = mac dinh he thong) */
    int block;             /* kich thuoc block CUDA (= kich thuoc tile) */
    const char *out;       /* file CSV ghi quy dao (NULL = khong ghi) */
    int every;             /* ghi 1 frame moi 'every' buoc */
    int verify;            /* so sanh voi ban tham chieu */
    int no_energy;         /* bo qua tinh nang luong (O(N^2) tren CPU) */
    const char *sched;     /* static | dynamic | guided (mac dinh: static) */
    int simd;              /* 1 = co simd, 0 = khong simd (mac dinh: 1) */
} nb_params;

/* ------------------------------------------------------------------ */
/* Dong ho                                                             */
/* ------------------------------------------------------------------ */
#ifdef _WIN32
#include <windows.h>
#endif
static double nb_wtime(void)
{
#if defined(_WIN32)
    /* omp_get_wtime cua MinGW chi chinh xac ~1 ms -> dung QueryPerformanceCounter */
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#elif defined(_OPENMP)
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* ------------------------------------------------------------------ */
/* Bo sinh so ngau nhien tat dinh (splitmix64) - de moi phien ban      */
/* (CPU, OpenMP, CUDA) deu bat dau tu cung mot trang thai ban dau.     */
/* ------------------------------------------------------------------ */
static double nb_rand(unsigned long long *s)
{
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0); /* [0, 1) */
}

static void nb_rand_dir(unsigned long long *s, double *x, double *y, double *z)
{
    double cz = 2.0 * nb_rand(s) - 1.0;
    double phi = 2.0 * NB_PI * nb_rand(s);
    double sz = sqrt(1.0 - cz * cz);
    *x = sz * cos(phi);
    *y = sz * sin(phi);
    *z = cz;
}

/*
 * Khoi tao cum sao theo mo hinh Plummer (Aarseth, Henon & Wielen 1974),
 * dua ve don vi N-body chuan (G = M = 1, E = -1/4).
 */
static void nb_init_plummer(vec4f *pos, vec4f *vel, int n, unsigned long long seed)
{
    unsigned long long s = seed;
    const double rscale = 3.0 * NB_PI / 16.0;
    const double vscale = 1.0 / sqrt(rscale);
    double cp[3] = {0, 0, 0}, cv[3] = {0, 0, 0};
    double *tmp = (double *)malloc(sizeof(double) * 6 * (size_t)n);
    int i;

    for (i = 0; i < n; i++) {
        double r, q, dx, dy, dz, v;
        for (;;) {                       /* ban kinh, cat bo r > 10 */
            double x1 = nb_rand(&s);
            if (x1 < 1e-12) continue;
            r = 1.0 / sqrt(pow(x1, -2.0 / 3.0) - 1.0);
            if (r < 10.0) break;
        }
        for (;;) {                       /* van toc: phuong phap loai bo */
            double x2 = nb_rand(&s);
            double y2 = 0.1 * nb_rand(&s);
            if (y2 < x2 * x2 * pow(1.0 - x2 * x2, 3.5)) { q = x2; break; }
        }
        v = q * sqrt(2.0) * pow(1.0 + r * r, -0.25);

        nb_rand_dir(&s, &dx, &dy, &dz);
        tmp[6 * i + 0] = r * dx * rscale;
        tmp[6 * i + 1] = r * dy * rscale;
        tmp[6 * i + 2] = r * dz * rscale;
        nb_rand_dir(&s, &dx, &dy, &dz);
        tmp[6 * i + 3] = v * dx * vscale;
        tmp[6 * i + 4] = v * dy * vscale;
        tmp[6 * i + 5] = v * dz * vscale;
        cp[0] += tmp[6 * i + 0]; cp[1] += tmp[6 * i + 1]; cp[2] += tmp[6 * i + 2];
        cv[0] += tmp[6 * i + 3]; cv[1] += tmp[6 * i + 4]; cv[2] += tmp[6 * i + 5];
    }
    /* dua khoi tam ve goc toa do va dung yen */
    for (i = 0; i < n; i++) {
        pos[i].x = (float)(tmp[6 * i + 0] - cp[0] / n);
        pos[i].y = (float)(tmp[6 * i + 1] - cp[1] / n);
        pos[i].z = (float)(tmp[6 * i + 2] - cp[2] / n);
        pos[i].w = (float)(1.0 / n);    /* khoi luong */
        vel[i].x = (float)(tmp[6 * i + 3] - cv[0] / n);
        vel[i].y = (float)(tmp[6 * i + 4] - cv[1] / n);
        vel[i].z = (float)(tmp[6 * i + 5] - cv[2] / n);
        vel[i].w = 0.0f;
    }
    free(tmp);
}

/*
 * Tong nang luong (tinh bang double, O(N^2/2)) - dung de kiem tra
 * tinh dung dan cua mo phong (nang luong phai gan nhu bao toan).
 */
static void nb_energy(const vec4f *pos, const vec4f *vel, int n, float eps,
                      double *ke_out, double *pe_out)
{
    double ke = 0.0, pe = 0.0;
    const double eps2 = (double)eps * eps;
    int i;
#pragma omp parallel for reduction(+ : ke, pe) schedule(dynamic, 64)
    for (i = 0; i < n; i++) {
        const double mi = pos[i].w;
        int j;
        ke += 0.5 * mi * ((double)vel[i].x * vel[i].x + (double)vel[i].y * vel[i].y +
                          (double)vel[i].z * vel[i].z);
        for (j = i + 1; j < n; j++) {
            double dx = (double)pos[j].x - pos[i].x;
            double dy = (double)pos[j].y - pos[i].y;
            double dz = (double)pos[j].z - pos[i].z;
            pe -= mi * pos[j].w / sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    }
    *ke_out = ke;
    *pe_out = pe;
}

/* So sanh hai trang thai: max |p_a - p_b| / max |p_b| */
static double nb_max_rel_diff(const vec4f *a, const vec4f *b, int n)
{
    double dmax = 0.0, pmax = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double dx = (double)a[i].x - b[i].x, dy = (double)a[i].y - b[i].y,
               dz = (double)a[i].z - b[i].z;
        double d = sqrt(dx * dx + dy * dy + dz * dz);
        double p = sqrt((double)b[i].x * b[i].x + (double)b[i].y * b[i].y +
                        (double)b[i].z * b[i].z);
        if (d > dmax) dmax = d;
        if (p > pmax) pmax = p;
    }
    return pmax > 0 ? dmax / pmax : dmax;
}

/* ------------------------------------------------------------------ */
/* Ghi quy dao ra CSV: step,id,x,y,z                                  */
/* ------------------------------------------------------------------ */
static void nb_write_header(FILE *f) { fprintf(f, "step,id,x,y,z\n"); }

static void nb_write_frame(FILE *f, int step, const vec4f *pos, int n)
{
    int i;
    for (i = 0; i < n; i++)
        fprintf(f, "%d,%d,%.6f,%.6f,%.6f\n", step, i, pos[i].x, pos[i].y, pos[i].z);
}

/* ------------------------------------------------------------------ */
/* Tham so dong lenh                                                   */
/* ------------------------------------------------------------------ */
static void nb_usage(const char *prog, int is_cuda)
{
    fprintf(stderr,
            "Cach dung: %s [tuy chon]\n"
            "  -n <N>        so vat the                (mac dinh 4096)\n"
            "  -s <steps>    so buoc mo phong          (mac dinh 10)\n"
            "  -dt <dt>      buoc thoi gian            (mac dinh 0.001)\n"
            "  -eps <eps>    softening length          (mac dinh 0.01)\n"
            "  -seed <s>     hat giong ngau nhien      (mac dinh 42)\n",
            prog);
    if (is_cuda)
        fprintf(stderr,
                "  -k <kernel>   naive | tiled             (mac dinh tiled)\n"
                "  -b <block>    so thread / block (tile)  (mac dinh 256)\n"
                "  -verify       so sanh ket qua voi ban CPU (dung N, steps nho)\n");
    else
        fprintf(stderr,
                "  -m <mode>     serial | omp | omp_newton3 (mac dinh omp)\n"
                "  -t <threads>  so luong OpenMP           (mac dinh: tat ca loi)\n"
                "  -sched <sch>  static | dynamic | guided (mac dinh: static)\n"
                "  -simd <0|1>   1: dung SIMD reduction, 0: tat SIMD (mac dinh 1)\n"
                "  -verify       so sanh ket qua omp voi serial\n");
    fprintf(stderr,
            "  -o <file.csv> ghi quy dao (chi nen dung voi N nho)\n"
            "  -every <k>    ghi 1 frame moi k buoc    (mac dinh 1)\n"
            "  -noenergy     bo qua kiem tra nang luong\n");
}

static void nb_parse(int argc, char **argv, nb_params *p, int is_cuda)
{
    int i;
    p->n = 4096;
    p->steps = 10;
    p->dt = 0.001f;
    p->eps = 0.01f;
    p->seed = 42ULL;
    p->mode = is_cuda ? "tiled" : "omp";
    p->threads = 0;
    p->block = 256;
    p->out = NULL;
    p->every = 1;
    p->verify = 0;
    p->no_energy = 0;
    p->sched = "static";
    p->simd = 1;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = (i + 1 < argc);
        if (!strcmp(a, "-n") && has) p->n = atoi(argv[++i]);
        else if (!strcmp(a, "-s") && has) p->steps = atoi(argv[++i]);
        else if (!strcmp(a, "-dt") && has) p->dt = (float)atof(argv[++i]);
        else if (!strcmp(a, "-eps") && has) p->eps = (float)atof(argv[++i]);
        else if (!strcmp(a, "-seed") && has) p->seed = strtoull(argv[++i], NULL, 10);
        else if ((!strcmp(a, "-m") || !strcmp(a, "-k")) && has) p->mode = argv[++i];
        else if (!strcmp(a, "-t") && has) p->threads = atoi(argv[++i]);
        else if (!strcmp(a, "-b") && has) p->block = atoi(argv[++i]);
        else if (!strcmp(a, "-sched") && has) p->sched = argv[++i];
        else if (!strcmp(a, "-simd") && has) p->simd = atoi(argv[++i]);
        else if (!strcmp(a, "-o") && has) p->out = argv[++i];
        else if (!strcmp(a, "-every") && has) p->every = atoi(argv[++i]);
        else if (!strcmp(a, "-verify")) p->verify = 1;
        else if (!strcmp(a, "-noenergy")) p->no_energy = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { nb_usage(argv[0], is_cuda); exit(0); }
        else {
            fprintf(stderr, "Tham so khong hop le: %s\n", a);
            nb_usage(argv[0], is_cuda);
            exit(1);
        }
    }
    if (p->n < 2 || p->steps < 1 || p->every < 1 || p->block < 1) {
        fprintf(stderr, "Gia tri tham so khong hop le.\n");
        exit(1);
    }
}

/*
 * In ket qua. Dong "RESULT,..." dung cho script benchmark doc tu dong.
 * Quy uoc cua GPU Gems 3 (ch.31): moi tuong tac cap = 20 phep toan dau phay dong.
 */
static void nb_report(const char *impl, const nb_params *p, int par, double seconds)
{
    double inter = (double)p->n * (double)p->n * (double)p->steps;
    double gips = inter / seconds / 1e9;
    printf("  Thoi gian tinh toan : %.4f s  (%.3f ms/buoc)\n", seconds,
           1e3 * seconds / p->steps);
    printf("  Tuong tac / giay    : %.3f ty\n", gips);
    printf("  Hieu nang           : %.2f GFLOP/s (20 flop/tuong tac)\n", 20.0 * gips);
    printf("RESULT,%s,%s,%d,%d,%d,%.6f,%.6f,%.4f\n", impl, p->mode, p->n, p->steps, par,
           seconds, gips, 20.0 * gips);
}

#endif /* NBODY_COMMON_H */
