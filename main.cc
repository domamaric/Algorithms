#include <cstdio>
#include "matmul.hh"

#define N 1024
float mat_a[N * N], mat_b[N * N], mat_c[N * N];

int main() {
    for (int i = 0; i < N * N; i++) {
        mat_a[i] = (float) rand() / RAND_MAX;
        mat_b[i] = (float) rand() / RAND_MAX;
    }
    
    clock_t start = clock();
    matmul(mat_a, mat_b, mat_c, N);

    float elapsed = (float) (clock() - start) / CLOCKS_PER_SEC;
    printf("Kernel usage with cache blocking: %.4f seconds\n", elapsed);
}
