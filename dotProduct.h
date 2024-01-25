#pragma once

#include "common.h"

float slow_dotProduct(const float *a, const float *b, int N) {
    float total = 0;
    for (int i = 0; i < N; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float sse_dotProduct(const float *a, const float *b, int N) {
    float total = 0;

    __m128 num1, num2, num3, num4;
    num4 = _mm_setzero_ps();

    for (int i = 0; i < N; i += 4) {
        num1 = _mm_loadu_ps(a + i);
        num2 = _mm_loadu_ps(b + i);
        num3 = _mm_mul_ps(num1, num2);
        num4 = _mm_add_ps(num4, num3);
    }

    num4 = _mm_hadd_ps(num4, num4);
    num4 = _mm_hadd_ps(num4, num4);

    _mm_store_ss(&total, num4);
    return total;
}

float avx_dotProduct(const float *a, const float *b, int N) {
    float total = 0;

    __m256 num1, num2, num3, num4;
    num4 = _mm256_setzero_ps();

    for (int i = 0; i < N; i += 8) {
        num1 = _mm256_loadu_ps(a + i);
        num2 = _mm256_loadu_ps(b + i);
        num3 = _mm256_mul_ps(num1, num2);
        num4 = _mm256_add_ps(num4, num3);
    }

    num4 = _mm256_hadd_ps(num4, num4);
    num4 = _mm256_hadd_ps(num4, num4);

    __m128 lo = _mm256_castps256_ps128(num4);
    __m128 hi = _mm256_extractf128_ps(num4, 1);
    lo = _mm_add_ps(lo, hi);

    _mm_store_ss(&total, lo);
    return total;
}

std::vector<double> getDotProductResult() {
    const int N = 100000;

    float *a = new float[N];
    float *b = new float[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    float *aSlow = new float[N];
    float *bSlow = new float[N];

    float *aSSE = new float[N];
    float *bSSE = new float[N];

    float *aAVX = new float[N];
    float *bAVX = new float[N];

    std::copy(a, a + N, aSlow);
    std::copy(b, b + N, bSlow);

    std::copy(a, a + N, aSSE);
    std::copy(b, b + N, bSSE);

    std::copy(a, a + N, aAVX);
    std::copy(b, b + N, bAVX);

    /*
     * SLOW DOT PRODUCT
     */
    auto startSlow = std::chrono::high_resolution_clock::now();
    float resultSlow = slow_dotProduct(aSlow, bSlow, N);
    auto endSlow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSlow = endSlow - startSlow;

    /*
     * SSE DOT PRODUCT
     */
    auto startSSE = std::chrono::high_resolution_clock::now();
    float resultSSE = sse_dotProduct(aSSE, bSSE, N);
    auto endSSE = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSSE = endSSE - startSSE;

    /*
     * AVX DOT PRODUCT
     */
    auto startAVX = std::chrono::high_resolution_clock::now();
    float resultAVX = avx_dotProduct(aAVX, bAVX, N);
    auto endAVX = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAVX = endAVX - startAVX;

    /*
     * DEALLOCATE
     */
    delete[] a;
    delete[] b;

    delete[] aSlow;
    delete[] bSlow;

    delete[] aSSE;
    delete[] bSSE;

    delete[] aAVX;
    delete[] bAVX;

    return {durationSlow.count(), durationSSE.count(), durationAVX.count()};
}
