/*
 * nbody_cpu.c - Mo phong N-body tren CPU: tuan tu (serial) va song song OpenMP.
 *
 * Bien dich:  gcc -O3 -march=native -ffast-math -fopenmp src/nbody_cpu.c -o bin/nbody_cpu -lm
 * Vi du:      ./bin/nbody_cpu -n 8192 -s 10 -m omp -t 8
 *             ./bin/nbody_cpu -n 1024 -s 5 -verify
 */
#include "nbody_common.h"
#include "nbody_cpu_kernels.h"

int main(int argc, char **argv)
{
    nb_params p;
    vec4f *pos, *vel, *acc;
    int parallel, nthreads = 1;
    double ke0 = 0, pe0 = 0, ke1 = 0, pe1 = 0, sec;
    FILE *out = NULL;

    nb_parse(argc, argv, &p, 0);
    if (strcmp(p.mode, "serial") && strcmp(p.mode, "omp")) {
        fprintf(stderr, "Che do khong hop le: %s (serial | omp)\n", p.mode);
        return 1;
    }
    parallel = !strcmp(p.mode, "omp");

#ifdef _OPENMP
    nb_omp_threads = p.threads;   /* chi ap dung cho nhan tinh toan, khong cho ham nang luong */
    nthreads = parallel ? nb_num_threads() : 1;
#else
    if (parallel) fprintf(stderr, "Canh bao: bien dich khong co -fopenmp, chay 1 luong.\n");
#endif

    pos = (vec4f *)malloc(sizeof(vec4f) * (size_t)p.n);
    vel = (vec4f *)malloc(sizeof(vec4f) * (size_t)p.n);
    acc = (vec4f *)malloc(sizeof(vec4f) * (size_t)p.n);
    if (!pos || !vel || !acc) { fprintf(stderr, "Khong du bo nho\n"); return 1; }
    nb_init_plummer(pos, vel, p.n, p.seed);

    printf("== N-body CPU | che do: %s | N = %d | steps = %d | dt = %g | eps = %g | luong = %d ==\n",
           p.mode, p.n, p.steps, p.dt, p.eps, nthreads);

    /* -------- Kiem tra tinh dung: OpenMP phai cho cung ket qua voi tuan tu -------- */
    if (p.verify) {
        size_t bytes = sizeof(vec4f) * (size_t)p.n;
        vec4f *ps = (vec4f *)malloc(bytes), *vs = (vec4f *)malloc(bytes);
        vec4f *po = (vec4f *)malloc(bytes), *vo = (vec4f *)malloc(bytes);
        double ts, to, err;
        memcpy(ps, pos, bytes); memcpy(vs, vel, bytes);
        memcpy(po, pos, bytes); memcpy(vo, vel, bytes);
        ts = nb_run_cpu(ps, vs, acc, p.n, p.steps, p.dt, p.eps, 0, NULL, 1);
        to = nb_run_cpu(po, vo, acc, p.n, p.steps, p.dt, p.eps, 1, NULL, 1);
        err = nb_max_rel_diff(po, ps, p.n);
        printf("[verify] serial %.4f s | omp %.4f s | speedup %.2fx | sai so tuong doi max = %.3e -> %s\n",
               ts, to, ts / to, err, err < 1e-5 ? "PASS" : "FAIL");
        free(ps); free(vs); free(po); free(vo);
    }

    if (!p.no_energy) nb_energy(pos, vel, p.n, p.eps, &ke0, &pe0);

    if (p.out) {
        out = fopen(p.out, "w");
        if (!out) { fprintf(stderr, "Khong mo duoc %s\n", p.out); return 1; }
        nb_write_header(out);
    }

    sec = nb_run_cpu(pos, vel, acc, p.n, p.steps, p.dt, p.eps, parallel, out, p.every);
    if (out) fclose(out);

    nb_report("cpu", &p, nthreads, sec);

    if (!p.no_energy) {
        double e0, e1;
        nb_energy(pos, vel, p.n, p.eps, &ke1, &pe1);
        e0 = ke0 + pe0;
        e1 = ke1 + pe1;
        printf("  Nang luong E0 = %.6f, E1 = %.6f, |dE/E0| = %.3e\n", e0, e1, fabs((e1 - e0) / e0));
    }

    free(pos); free(vel); free(acc);
    return 0;
}
