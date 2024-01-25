#pragma once

#include "common.h"

class Neuron {
public:
    explicit Neuron(int numPrevNeurons) : numWeights(numPrevNeurons) {
        gradientWeights = new float[numWeights];
        weights = new float[numWeights];
        gradientBias = 0;

        for (int i = 0; i < numWeights; ++i) {
            gradientWeights[i] = 0;
            weights[i] = i * 0.003f;
        }
    }

    Neuron(const Neuron& other) : numWeights(other.numWeights) {
        gradientWeights = new float[numWeights];
        weights = new float[numWeights];

        for (int i = 0; i < numWeights; ++i) {
            gradientWeights[i] = other.gradientWeights[i];
            weights[i] = other.weights[i];
        }

        gradientBias = other.gradientBias;
    }

    ~Neuron() {
        delete[] gradientWeights;
        delete[] weights;
    }

    void updateGradientWeightAt(int i, float val) {
        gradientWeights[i] += val;
    }

    void updateGradientBias(float val) {
        gradientBias += val;
    }

    float* getGraidentWeights() {
        return gradientWeights;
    }

    float getGradientBias() {
        return gradientBias;
    }

    float *getWeights() const {
        return weights;
    }

private:
    int numWeights;
    float *gradientWeights;
    float gradientBias;
    float *weights;

};
