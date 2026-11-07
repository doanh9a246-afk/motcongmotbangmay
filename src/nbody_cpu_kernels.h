/*
 * nbody_cpu_kernels.h - Nhan tinh toan tren CPU: tuan tu va OpenMP.
 * Dung chung cho nbody_cpu.c va lam ban tham chieu cho nbody_cuda.cu (-verify).
 */
#ifndef NBODY_CPU_KERNELS_H
#define NBODY_CPU_KERNELS_H

#include "nbody_common.h"

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
static void nb_acc_omp(const vec4f *pos, vec4f *acc, int n, float eps2)
{
    int i;
#pragma omp parallel for schedule(static) num_threads(nb_num_threads())
    for (i = 0; i < n; i++) nb_body_acc(pos, acc, i, n, eps2);
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

/* Chay 'steps' buoc tren CPU; tra ve thoi gian tinh toan (khong tinh thoi gian ghi file). */
static double nb_run_cpu(vec4f *pos, vec4f *vel, vec4f *acc, int n, int steps, float dt,
                         float eps, int parallel, FILE *out, int every)
{
    const float eps2 = eps * eps;
    double total = 0.0;
    int s;
    /* OpenMP voi 1 luong == tuan tu (va tranh loi libgomp MinGW noi o tren). */
    if (parallel && nb_num_threads() == 1) parallel = 0;
    /*
     * Warm-up (khong tinh gio): lan dau vao vung song song, runtime OpenMP phai tao
     * cac luong -> neu tinh vao thoi gian se lam speedup bi sai, nhat la khi N nho.
     * Chi ghi vao acc (mang tam) nen khong lam thay doi trang thai mo phong.
     */
    if (parallel) nb_acc_omp(pos, acc, n, eps2);
    if (out) nb_write_frame(out, 0, pos, n);
    for (s = 1; s <= steps; s++) {
        double t0 = nb_wtime();
        if (parallel) nb_acc_omp(pos, acc, n, eps2);
        else          nb_acc_serial(pos, acc, n, eps2);
        nb_integrate(pos, vel, acc, n, dt, parallel);
        total += nb_wtime() - t0;
        if (out && s % every == 0) nb_write_frame(out, s, pos, n);
    }
    return total;
}

#endif /* NBODY_CPU_KERNELS_H */
