#pragma once

#include "common.h"

float beta1 = 0.9f;
float beta2 = 0.999f;
float epsilon = 1e-8f;
float lr = 0.001f;

void slow_adam(float *weights, float *m_weights, float *v_weights, const float *gradients, int numWeights) {
    for (int i = 0; i < numWeights; ++i) {
        float gradient = gradients[i];
        m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * gradient;
        v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * (gradient * gradient);
        weights[i] -= lr * m_weights[i] / (std::sqrt(v_weights[i]) + epsilon);
    }
}

__m128 beta1_sse = _mm_set1_ps(beta1);
__m128 beta2_sse = _mm_set1_ps(beta2);
__m128 one_minus_beta1_sse = _mm_set1_ps(1.0f - beta1);
__m128 one_minus_beta2_sse = _mm_set1_ps(1.0f - beta2);
__m128 epsilon_sse = _mm_set1_ps(epsilon);
__m128 lr_sse = _mm_set1_ps(lr);

void sse_adam(float *weights, float *m_weights, float *v_weights, const float *gradients, int numWeights) {
    for (int i = 0; i < numWeights; i += 4) {
        __m128 gradient = _mm_loadu_ps(gradients + i);
        __m128 m_weight = _mm_loadu_ps(m_weights + i);
        __m128 v_weight = _mm_loadu_ps(v_weights + i);
        __m128 weight = _mm_loadu_ps(weights + i);

        m_weight = _mm_add_ps(_mm_mul_ps(beta1_sse, m_weight), _mm_mul_ps(one_minus_beta1_sse, gradient));
        v_weight = _mm_add_ps(_mm_mul_ps(beta2_sse, v_weight), _mm_mul_ps(one_minus_beta2_sse, _mm_mul_ps(gradient, gradient)));

        __m128 denom = _mm_add_ps(_mm_sqrt_ps(v_weight), epsilon_sse);
        weight = _mm_sub_ps(weight, _mm_div_ps(_mm_mul_ps(lr_sse, m_weight), denom));

        _mm_storeu_ps(weights + i, weight);
        _mm_storeu_ps(m_weights + i, m_weight);
        _mm_storeu_ps(v_weights + i, v_weight);
    }
}

__m256 beta1_avx = _mm256_set1_ps(beta1);
__m256 beta2_avx = _mm256_set1_ps(beta2);
__m256 one_minus_beta1_avx = _mm256_set1_ps(1.0f - beta1);
__m256 one_minus_beta2_avx = _mm256_set1_ps(1.0f - beta2);
__m256 epsilon_avx = _mm256_set1_ps(epsilon);
__m256 lr_avx = _mm256_set1_ps(lr);

void avx_adam(float *weights, float *m_weights, float *v_weights, const float *gradients, int numWeights) {
    for (int i = 0; i < numWeights; i += 8) {
        __m256 gradient = _mm256_loadu_ps(gradients + i);
        __m256 m_weight = _mm256_loadu_ps(m_weights + i);
        __m256 v_weight = _mm256_loadu_ps(v_weights + i);
        __m256 weight = _mm256_loadu_ps(weights + i);

        m_weight = _mm256_add_ps(_mm256_mul_ps(beta1_avx, m_weight), _mm256_mul_ps(one_minus_beta1_avx, gradient));
        v_weight = _mm256_add_ps(_mm256_mul_ps(beta2_avx, v_weight), _mm256_mul_ps(one_minus_beta2_avx, _mm256_mul_ps(gradient, gradient)));

        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_weight), epsilon_avx);
        weight = _mm256_sub_ps(weight, _mm256_div_ps(_mm256_mul_ps(lr_avx, m_weight), denom));

        _mm256_storeu_ps(weights + i, weight);
        _mm256_storeu_ps(m_weights + i, m_weight);
        _mm256_storeu_ps(v_weights + i, v_weight);
    }
}

std::vector<double> getAdamResult() {
    const int numWeights = 768;

    float *weights = new float[numWeights];
    float *m_weights = new float[numWeights];
    float *v_weights = new float[numWeights];
    float *gradients = new float[numWeights];

    for (int i = 0; i < numWeights; ++i) {
        weights[i] = i;
        m_weights[i] = i + 1;
        v_weights[i] = i + 2;
        gradients[i] = i + 3;
    }

    // Slow version
    float *weightsSlow = new float[numWeights];
    float *m_weightsSlow = new float[numWeights];
    float *v_weightsSlow = new float[numWeights];
    std::copy(weights, weights + numWeights, weightsSlow);
    std::copy(m_weights, m_weights + numWeights, m_weightsSlow);
    std::copy(v_weights, v_weights + numWeights, v_weightsSlow);

    // SSE version
    float *weightsSSE = new float[numWeights];
    float *m_weightsSSE = new float[numWeights];
    float *v_weightsSSE = new float[numWeights];
    std::copy(weights, weights + numWeights, weightsSSE);
    std::copy(m_weights, m_weights + numWeights, m_weightsSSE);
    std::copy(v_weights, v_weights + numWeights, v_weightsSSE);

    // AVX version
    float *weightsAVX = new float[numWeights];
    float *m_weightsAVX = new float[numWeights];
    float *v_weightsAVX = new float[numWeights];
    std::copy(weights, weights + numWeights, weightsAVX);
    std::copy(m_weights, m_weights + numWeights, m_weightsAVX);
    std::copy(v_weights, v_weights + numWeights, v_weightsAVX);

    /*
     * SLOW ADAM
     */
    auto startSlow = std::chrono::high_resolution_clock::now();
    slow_adam(weightsSlow, m_weightsSlow, v_weightsSlow, gradients, numWeights);
    auto endSlow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSlow = endSlow - startSlow;

    /*
     * SSE ADAM
     */
    auto startSSE = std::chrono::high_resolution_clock::now();
    sse_adam(weightsSSE, m_weightsSSE, v_weightsSSE, gradients, numWeights);
    auto endSSE = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSSE = endSSE - startSSE;

    /*
     * AVX ADAM
     */
    auto startAVX = std::chrono::high_resolution_clock::now();
    avx_adam(weightsAVX, m_weightsAVX, v_weightsAVX, gradients, numWeights);
    auto endAVX = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAVX = endAVX - startAVX;

    /*
     * DEBUG
     */
    bool sseCorrect = areArraysEqual(weightsSlow, weightsSSE, numWeights) &&
                      areArraysEqual(m_weightsSlow, m_weightsSSE, numWeights) &&
                      areArraysEqual(v_weightsSlow, v_weightsSSE, numWeights);

    bool avxCorrect = areArraysEqual(weightsSlow, weightsAVX, numWeights) &&
                      areArraysEqual(m_weightsSlow, m_weightsAVX, numWeights) &&
                      areArraysEqual(v_weightsSlow, v_weightsAVX, numWeights);

    if (!sseCorrect) {
        std::cerr << "SSE VERSION IS INCORRECT!" << std::endl;
    }

    if (!avxCorrect) {
        std::cerr << "AVX VERSION IS INCORRECT!" << std::endl;
    }

    /*
     * DEALLOCATE
     */
    delete[] weights;
    delete[] m_weights;
    delete[] v_weights;
    delete[] gradients;

    delete[] weightsSlow;
    delete[] m_weightsSlow;
    delete[] v_weightsSlow;

    delete[] weightsSSE;
    delete[] m_weightsSSE;
    delete[] v_weightsSSE;

    delete[] weightsAVX;
    delete[] m_weightsAVX;
    delete[] v_weightsAVX;

    return {durationSlow.count(), durationSSE.count(), durationAVX.count()};
}
