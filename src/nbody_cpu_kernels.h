/*
 * nbody_cpu_kernels.h - Nhan tinh toan tren CPU: tuan tu va OpenMP.
 * Dung chung cho nbody_cpu.c va lam ban tham chieu cho nbody_cuda.cu (-verify).
 */
#ifndef NBODY_CPU_KERNELS_H
#define NBODY_CPU_KERNELS_H

#include "nbody_common.h"

#ifdef __GNUC__
#define NB_UNUSED __attribute__((unused))
#else
#define NB_UNUSED
#endif

/* So luong OpenMP cho cac nhan tinh toan (0 = mac dinh he thong). */
static int nb_omp_threads = 0;

static int nb_num_threads(void)
{
#ifdef _OPENMP
    return nb_omp_threads > 0 ? nb_omp_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

/*
 * Gia toc cua vat the i do toan bo N vat the gay ra.
 * Khong can loai tru j == i: khi do r = 0 nen dong gop bang 0 (nho eps^2 > 0),
 * giong cach lam cua GPU Gems 3 - giup vong lap khong co re nhanh, de vector hoa.
 */
static inline void nb_body_acc(const vec4f *__restrict__ pos, vec4f *__restrict__ acc,
                               int i, int n, float eps2)
{
    const float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int j;
#pragma omp simd reduction(+ : ax, ay, az)
    for (j = 0; j < n; j++) {
        const float dx = pos[j].x - xi;
        const float dy = pos[j].y - yi;
        const float dz = pos[j].z - zi;
        const float d2 = dx * dx + dy * dy + dz * dz + eps2;
        const float inv = 1.0f / sqrtf(d2);
        const float s = pos[j].w * inv * inv * inv;
        ax += dx * s;
        ay += dy * s;
        az += dz * s;
    }
    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
    acc[i].w = 0.0f;
}

/* Gia toc cua vat the i khong dung chi thi SIMD (de so sanh anh huong cua vector hoa). */
static inline void nb_body_acc_nosimd(const vec4f *__restrict__ pos, vec4f *__restrict__ acc,
                                      int i, int n, float eps2)
{
    const float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int j;
    for (j = 0; j < n; j++) {
        const float dx = pos[j].x - xi;
        const float dy = pos[j].y - yi;
        const float dz = pos[j].z - zi;
        const float d2 = dx * dx + dy * dy + dz * dz + eps2;
        const float inv = 1.0f / sqrtf(d2);
        const float s = pos[j].w * inv * inv * inv;
        ax += dx * s;
        ay += dy * s;
        az += dz * s;
    }
    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
    acc[i].w = 0.0f;
}

/* ---------------- Phien ban tuan tu ---------------- */
static void nb_acc_serial(const vec4f *pos, vec4f *acc, int n, float eps2)
{
    int i;
    for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
}

/*
 * ---------------- Phien ban OpenMP ----------------
 * Moi luong phu trach mot day cac vat the i. Moi i chi GHI vao acc[i] cua rieng no
 * va chi DOC pos[] => khong co race condition, khong can critical/atomic.
 * Khoi luong cong viec moi i bang nhau (N tuong tac) nen schedule(static) la toi uu.
 */
static void nb_acc_omp_custom(const vec4f *pos, vec4f *acc, int n, float eps2,
                              const char *sched, int simd)
{
    int i;
    int nth = nb_num_threads();
    int is_dynamic = (sched && !strcmp(sched, "dynamic"));
    int is_guided  = (sched && !strcmp(sched, "guided"));

    if (simd) {
        if (is_dynamic) {
#pragma omp parallel for schedule(dynamic, 64) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
        } else if (is_guided) {
#pragma omp parallel for schedule(guided, 64) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
        } else {
#pragma omp parallel for schedule(static) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
        }
    } else {
        if (is_dynamic) {
#pragma omp parallel for schedule(dynamic, 64) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc_nosimd(pos, acc, i, n, eps2);
        } else if (is_guided) {
#pragma omp parallel for schedule(guided, 64) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc_nosimd(pos, acc, i, n, eps2);
        } else {
#pragma omp parallel for schedule(static) num_threads(nth)
            for (i = 0; i < n; i++) nb_body_acc_nosimd(pos, acc, i, n, eps2);
        }
    }
}

static NB_UNUSED void nb_acc_omp(const vec4f *pos, vec4f *acc, int n, float eps2)
{
    nb_acc_omp_custom(pos, acc, n, eps2, "static", 1);
}

/*
 * ---------------- Bien the Dinh luat 3 Newton (N^2 / 2 cap) ----------------
 * F_ji = -F_ij nen chi can duyet i < j (khoang N^2/2 cap).
 * Do ca hai vat the i va j deu duoc cap nhat dong thoi, de tranh race condition
 * ma khong phai dung critical/atomic lam cham chuong trinh, moi luong duoc cap
 * mot mang gia toc cuc bo (reduction buffer) kich thuoc N.
 * Cuoi cung, cac mang cuc bo duoc cong don vao mang acc tong.
 */
static void nb_acc_omp_newton3(const vec4f *pos, vec4f *acc, vec4f *acc_threads,
                               int n, float eps2, const char *sched)
{
    int p = nb_num_threads();
    int is_dynamic = (sched && !strcmp(sched, "dynamic"));
    int is_static  = (sched && !strcmp(sched, "static"));

#pragma omp parallel num_threads(p)
    {
        int tid = omp_get_thread_num();
        vec4f *my_acc = acc_threads + (size_t)tid * (size_t)n;
        int i;
        memset(my_acc, 0, (size_t)n * sizeof(vec4f));

        if (is_static) {
#pragma omp for schedule(static)
            for (i = 0; i < n; i++) {
                const float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
                const float mi = pos[i].w;
                float aix = 0.0f, aiy = 0.0f, aiz = 0.0f;
                int j;
                for (j = i + 1; j < n; j++) {
                    const float dx = pos[j].x - xi;
                    const float dy = pos[j].y - yi;
                    const float dz = pos[j].z - zi;
                    const float d2 = dx * dx + dy * dy + dz * dz + eps2;
                    const float inv = 1.0f / sqrtf(d2);
                    const float inv3 = inv * inv * inv;
                    const float mj = pos[j].w;

                    const float fx = dx * inv3;
                    const float fy = dy * inv3;
                    const float fz = dz * inv3;

                    aix += fx * mj;
                    aiy += fy * mj;
                    aiz += fz * mj;

                    my_acc[j].x -= fx * mi;
                    my_acc[j].y -= fy * mi;
                    my_acc[j].z -= fz * mi;
                }
                my_acc[i].x += aix;
                my_acc[i].y += aiy;
                my_acc[i].z += aiz;
            }
        } else if (is_dynamic) {
#pragma omp for schedule(dynamic, 16)
            for (i = 0; i < n; i++) {
                const float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
                const float mi = pos[i].w;
                float aix = 0.0f, aiy = 0.0f, aiz = 0.0f;
                int j;
                for (j = i + 1; j < n; j++) {
                    const float dx = pos[j].x - xi;
                    const float dy = pos[j].y - yi;
                    const float dz = pos[j].z - zi;
                    const float d2 = dx * dx + dy * dy + dz * dz + eps2;
                    const float inv = 1.0f / sqrtf(d2);
                    const float inv3 = inv * inv * inv;
                    const float mj = pos[j].w;

                    const float fx = dx * inv3;
                    const float fy = dy * inv3;
                    const float fz = dz * inv3;

                    aix += fx * mj;
                    aiy += fy * mj;
                    aiz += fz * mj;

                    my_acc[j].x -= fx * mi;
                    my_acc[j].y -= fy * mi;
                    my_acc[j].z -= fz * mi;
                }
                my_acc[i].x += aix;
                my_acc[i].y += aiy;
                my_acc[i].z += aiz;
            }
        } else {
#pragma omp for schedule(guided, 16)
            for (i = 0; i < n; i++) {
                const float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
                const float mi = pos[i].w;
                float aix = 0.0f, aiy = 0.0f, aiz = 0.0f;
                int j;
                for (j = i + 1; j < n; j++) {
                    const float dx = pos[j].x - xi;
                    const float dy = pos[j].y - yi;
                    const float dz = pos[j].z - zi;
                    const float d2 = dx * dx + dy * dy + dz * dz + eps2;
                    const float inv = 1.0f / sqrtf(d2);
                    const float inv3 = inv * inv * inv;
                    const float mj = pos[j].w;

                    const float fx = dx * inv3;
                    const float fy = dy * inv3;
                    const float fz = dz * inv3;

                    aix += fx * mj;
                    aiy += fy * mj;
                    aiz += fz * mj;

                    my_acc[j].x -= fx * mi;
                    my_acc[j].y -= fy * mi;
                    my_acc[j].z -= fz * mi;
                }
                my_acc[i].x += aix;
                my_acc[i].y += aiy;
                my_acc[i].z += aiz;
            }
        }

        /* Reduction: cong don tat ca cac ban sao cua tung luong vao acc chung */
#pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            int t;
            for (t = 0; t < p; t++) {
                sx += acc_threads[(size_t)t * (size_t)n + i].x;
                sy += acc_threads[(size_t)t * (size_t)n + i].y;
                sz += acc_threads[(size_t)t * (size_t)n + i].z;
            }
            acc[i].x = sx;
            acc[i].y = sy;
            acc[i].z = sz;
            acc[i].w = 0.0f;
        }
    }
}

/* Cap nhat van toc va vi tri cua vat the i (symplectic Euler). */
static inline void nb_body_update(vec4f *__restrict__ pos, vec4f *__restrict__ vel,
                                  const vec4f *__restrict__ acc, int i, float dt)
{
    vel[i].x += acc[i].x * dt;
    vel[i].y += acc[i].y * dt;
    vel[i].z += acc[i].z * dt;
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}

/*
 * Luu y: khong dung "#pragma omp parallel for if(parallel)" vi libgomp cua MinGW
 * (gcc 12, ban Brecht Sanders) bi loi thoat chuong trinh voi moi vung song song
 * chi co 1 luong. Vi vay tach rieng nhanh tuan tu.
 */
static void nb_integrate(vec4f *pos, vec4f *vel, const vec4f *acc, int n, float dt, int parallel)
{
    int i;
    if (parallel) {
#pragma omp parallel for schedule(static) num_threads(nb_num_threads())
        for (i = 0; i < n; i++) nb_body_update(pos, vel, acc, i, dt);
    } else {
        for (i = 0; i < n; i++) nb_body_update(pos, vel, acc, i, dt);
    }
}

/* Chay 'steps' buoc tren CPU; ho tro ca mode serial (0), omp (1), omp_newton3 (2). */
static double nb_run_cpu_ex(vec4f *pos, vec4f *vel, vec4f *acc, int n, int steps, float dt,
                            float eps, int mode_code, const char *sched, int simd, FILE *out, int every)
{
    const float eps2 = eps * eps;
    double total = 0.0;
    int s;
    int nth = nb_num_threads();
    vec4f *acc_threads = NULL;

    if (mode_code > 0 && nth == 1) {
        if (mode_code == 1) mode_code = 0;
    }

    if (mode_code == 2) {
        acc_threads = (vec4f *)malloc((size_t)nth * (size_t)n * sizeof(vec4f));
        if (!acc_threads) {
            fprintf(stderr, "Khong du bo nho cho Newton 3 reduction buffers\n");
            exit(1);
        }
    }

    /* Warm-up vung song song */
    if (mode_code == 1) {
        nb_acc_omp_custom(pos, acc, n, eps2, sched, simd);
    } else if (mode_code == 2) {
        nb_acc_omp_newton3(pos, acc, acc_threads, n, eps2, sched);
    }

    if (out) nb_write_frame(out, 0, pos, n);
    for (s = 1; s <= steps; s++) {
        double t0 = nb_wtime();
        if (mode_code == 1) {
            nb_acc_omp_custom(pos, acc, n, eps2, sched, simd);
            nb_integrate(pos, vel, acc, n, dt, 1);
        } else if (mode_code == 2) {
            nb_acc_omp_newton3(pos, acc, acc_threads, n, eps2, sched);
            nb_integrate(pos, vel, acc, n, dt, 1);
        } else {
            nb_acc_serial(pos, acc, n, eps2);
            nb_integrate(pos, vel, acc, n, dt, 0);
        }
        total += nb_wtime() - t0;
        if (out && s % every == 0) nb_write_frame(out, s, pos, n);
    }

    if (acc_threads) free(acc_threads);
    return total;
}

static NB_UNUSED double nb_run_cpu(vec4f *pos, vec4f *vel, vec4f *acc, int n, int steps, float dt,
                         float eps, int parallel, FILE *out, int every)
{
    return nb_run_cpu_ex(pos, vel, acc, n, steps, dt, eps, parallel ? 1 : 0, "static", 1, out, every);
}

#endif /* NBODY_CPU_KERNELS_H */
