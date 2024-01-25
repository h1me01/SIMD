#pragma once

#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <numeric>

double calculateAverage(const std::vector<double>& vec) {
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double average = sum / vec.size();
    return average;
}

bool areArraysEqual(float *original, float *copy, int size) {
    return std::equal(original, original + size, copy);
}

enum ActivationType {
    NONE,
    SIGMOID,
    RELU
};

float activate(float _x, ActivationType _type) {
    switch (_type) {
        case SIGMOID:
            return 1.0f / (1.0f + std::exp(-_x));
        case RELU:
            return (_x > 0) ? _x : 0;
        default:
            return 0;
    }
}

float activateDer(float _x, ActivationType _type) {
    switch (_type) {
        case SIGMOID:
            return activate(_x, SIGMOID) * (1.0f - activate(_x, SIGMOID));
        case RELU:
            return (_x > 0) ? 1.0f : 0;
        default:
            return 0;
    }
}
