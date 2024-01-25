#pragma once

#include "neuron.h"

void slow_updateGradients(Neuron **neurons, float *input, float *deltas, int numNeurons, int numPrevNeurons) {
    for (int i = 0; i < numNeurons; ++i) {
        for (int j = 0; j < numPrevNeurons; ++j) {
            float inputWeightDer = input[j] * deltas[i];
            neurons[i]->updateGradientWeightAt(j, inputWeightDer);
        }
        neurons[i]->updateGradientBias(deltas[i]);
    }
}

void sse_updateGradients(Neuron** neurons, float* input, float* deltas, int numNeurons, int numPrevNeurons) {
    for (int i = 0; i < numNeurons; ++i) {
        __m128 delta_sse = _mm_set1_ps(deltas[i]);

        for (int j = 0; j < numPrevNeurons; j += 4) {
            __m128 input_sse = _mm_loadu_ps(input + j);
            __m128 inputWeightDer_sse = _mm_mul_ps(input_sse, delta_sse);

            float elements[4];
            _mm_storeu_ps(elements, inputWeightDer_sse);

            neurons[i]->updateGradientWeightAt(j, elements[0]);
            neurons[i]->updateGradientWeightAt(j + 1, elements[1]);
            neurons[i]->updateGradientWeightAt(j + 2, elements[2]);
            neurons[i]->updateGradientWeightAt(j + 3, elements[3]);
        }

        neurons[i]->updateGradientBias(deltas[i]);
    }
}

void avx_updateGradients(Neuron **neurons, float *input, float *deltas, int numNeurons, int numPrevNeurons) {
    for (int i = 0; i < numNeurons; ++i) {
        __m256 delta_avx = _mm256_set1_ps(deltas[i]);

        for (int j = 0; j < numPrevNeurons; j += 8) {
            __m256 input_avx = _mm256_loadu_ps(input + j);
            __m256 inputWeightDer_avx = _mm256_mul_ps(input_avx, delta_avx);

            float elements[8];
            _mm256_storeu_ps(elements, inputWeightDer_avx);

            neurons[i]->updateGradientWeightAt(j, elements[0]);
            neurons[i]->updateGradientWeightAt(j + 1, elements[1]);
            neurons[i]->updateGradientWeightAt(j + 2, elements[2]);
            neurons[i]->updateGradientWeightAt(j + 3, elements[3]);
            neurons[i]->updateGradientWeightAt(j + 4, elements[4]);
            neurons[i]->updateGradientWeightAt(j + 5, elements[5]);
            neurons[i]->updateGradientWeightAt(j + 6, elements[6]);
            neurons[i]->updateGradientWeightAt(j + 7, elements[7]);
        }

        neurons[i]->updateGradientBias(deltas[i]);
    }
}

std::vector<double> getGradientsUpdateResult() {
    const int numNeurons = 128;
    const int numPrevNeurons = 768;

    Neuron **neurons = new Neuron *[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        neurons[i] = new Neuron(numPrevNeurons);
    }

    float *input = new float[numPrevNeurons];
    float *deltas = new float[numNeurons];

    for (int i = 0; i < numPrevNeurons; ++i) {
        input[i] = i * 0.001f;
    }

    for (int i = 0; i < numNeurons; ++i) {
        deltas[i] = i * 0.02f;
    }

    // Slow version
    Neuron **neuronsSlow = new Neuron *[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        neuronsSlow[i] = new Neuron(*neurons[i]);
    }

    // SSE version
    Neuron **neuronsSSE = new Neuron *[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        neuronsSSE[i] = new Neuron(*neurons[i]);
    }

    // AVX version
    Neuron **neuronsAVX = new Neuron *[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        neuronsAVX[i] = new Neuron(*neurons[i]);
    }

    /*
     * SLOW LAYER GRADIENT UPDATE
     */
    auto startSlow = std::chrono::high_resolution_clock::now();
    slow_updateGradients(neuronsSlow, input, deltas, numNeurons, numPrevNeurons);
    auto endSlow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSlow = endSlow - startSlow;

    /*
     * SSE LAYER GRADIENT UPDATE
     */
    auto startSSE = std::chrono::high_resolution_clock::now();
    sse_updateGradients(neuronsSSE, input, deltas, numNeurons, numPrevNeurons);
    auto endSSE = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSSE = endSSE - startSSE;

    /*
     * AVX LAYER GRADIENT UPDATE
     */
    auto startAVX = std::chrono::high_resolution_clock::now();
    avx_updateGradients(neuronsAVX, input, deltas, numNeurons, numPrevNeurons);
    auto endAVX = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationAVX = endAVX - startAVX;

    /*
     * DEBUG
     */
    for(int i = 0; i < numNeurons; ++i) {
        bool sseCorrect = areArraysEqual(neuronsSlow[i]->getGraidentWeights(),
                                         neuronsSSE[i]->getGraidentWeights(),
                                         numPrevNeurons);
        bool avxCorrect = areArraysEqual(neuronsSlow[i]->getGraidentWeights(),
                                         neuronsAVX[i]->getGraidentWeights(),
                                         numPrevNeurons);

        if(!sseCorrect) {
            std::cerr << "SSE VERSION IS INCORRECT!" << std::endl;
        }

        if(!avxCorrect) {
            std::cerr << "AVX VERSION IS INCORRECT!" << std::endl;
        }
    }

    /*
     * DEALLOCATE
     */
    for (int i = 0; i < numNeurons; ++i) {
        delete neurons[i];
        delete neuronsSlow[i];
        delete neuronsSSE[i];
        delete neuronsAVX[i];
    }

    delete[] neurons;
    delete[] neuronsSlow;
    delete[] neuronsSSE;
    delete[] neuronsAVX;

    delete[] input;
    delete[] deltas;

    return {durationSlow.count(), durationSSE.count(), durationAVX.count()};
}
