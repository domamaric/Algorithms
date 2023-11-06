// Micro-kernel
#include <algorithm>
#include <cstring>
#include <ctime>
#include <x86intrin.h>

const int B = 8; // number of elements in a vector
typedef float vector __attribute__ (( vector_size(4 * B) ));

float* alloc(int n) {
    float* ptr = (float*) std::aligned_alloc(64, 4 * n);
    memset(ptr, 0, 4 * n); 
    return ptr;
}

inline int idx_a(int i, int j, int n) {
    return i / 6 * 6 * n + j * 6 + i % 6;
}

inline int idx_b(int i, int j, int n) {
    return j / 16 * 16 * n + i * 16 + j % 16;
}

void kernel(const float *a, const vector *b, vector __restrict__ *c, int x, int y, int l, int r, int n) {
    vector t[6][2] = {0};

    for (int k = l; k < r; k++) {
        __builtin_prefetch(&b[n / 8 * (k + 8) + y / 8]);
        for (int i = 0; i < 6; i++) {
            vector alpha = _mm256_set1_ps(a[x * n + i * n + k]);//a[x * ny + k * 6 + i]);
            for (int j = 0; j < 2; j++)
                t[i][j] += alpha * b[n / 8 * k + y / 8 + j];//b[(y / 16 * 2 * nx + k * 2 + j)];
                // b[idx_b(k, y + 8 * j, nx) / 8];
        }
    }

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 2; j++)
            c[x * n / 8 + i * n / 8 + y / 8 + j] += t[i][j];
}

const int L1 = (1<<16) / 4; // L1 cache is 64K
const int L2 = (1<<19) / 4; // L2 cache is 512K
const int L3 = (1<<21) / 4; // L3 cache is 2M

void matmul(const float *_a, const float *_b, float *_c, int n) {
    int nx = (n + 5) / 6 * 6;
    int ny = (n + 15) / 16 * 16;
    
    const int MAXN = 1920 * 1920; // ~15MB each
    alignas(64) static float a[MAXN], b[MAXN], c[MAXN];

    memset(c, 0, 4 * nx * ny);

    for (int i = 0; i < n; i++) {
        memcpy(&a[i * ny], &_a[i * n], 4 * n);
        memcpy(&b[i * ny], &_b[i * n], 4 * n);
    }

    for (int i = 0; i < n; i += 6)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < std::min(6, n - i); k++)
                a[i * ny + j * 6 + k] = _a[(i+k) * n + j];
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j += 16)
            for (int k = 0; k < std::min(16, n - j); k++)
                b[j * nx + i * 16 + k] = _b[i * n + j + k];

    // how many columns of b fit in L3
    const int s3 = std::min(L3 / nx / 16 * 16, ny);
    // how many rows of a fit in L2
    const int s2 = std::min(L2 / ny / 6 * 6, nx);
    // how tall a (k x s3) block in b can be to fit in L1
    const int s1 = std::min(L1 / s3, nx);
    
    for (int i3 = 0; i3 < ny; i3 += s3)
        // now we are working with b[:][i3:i3+s3]
        for (int i2 = 0; i2 < nx; i2 += s2)
            // now we are working with a[i2:i2+s2][:]
            for (int i1 = 0; i1 < ny; i1 += s1)
                // now we are working with b[i1:i1+s1][i3:i3+s3]
                // this equates to updating c[i2:i2+s2][i3:i3+s3]
                // with [l:r] = [i1:i1+s1]
                for (int x = i2; x < std::min(i2 + s2, nx); x += 6)
                    for (int y = i3; y < std::min(i3 + s3, ny); y += 16)
                        kernel(a, (vector*) b, (vector*) c, x, y, i1, std::min(i1 + s1, n), ny);

    for (int i = 0; i < n; i++)
        memcpy(&_c[i * n], &c[i * ny], 4 * n);
}