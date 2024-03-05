#include "dotProduct.h"
#include "adam.h"
#include "gradientsUpdate.h"

int main() {
    /*
     * TESTING
     */
    const int MAX_ITERATIONS = 1000;
    std::vector<double> slow, sse, avx;

    int count = 0;
    while(count < MAX_ITERATIONS) {
        std::vector<double> tempDot = getDotProductResult();
        std::vector<double> tempAdam = getAdamResult();
        std::vector<double> tempGradientsUpdate = getGradientsUpdateResult();

        slow.insert(slow.end(), {tempDot[0], tempAdam[0], tempGradientsUpdate[0]});
        sse.insert(sse.end(), {tempDot[1], tempAdam[1], tempGradientsUpdate[1]});
        avx.insert(avx.end(), {tempDot[2], tempAdam[2], tempGradientsUpdate[2]});

        count++;
    }

    std::cout << "Slow: ";
    std::cout << calculateAverage(slow) << std::endl;
    std::cout << "SSE: ";
    std::cout << calculateAverage(sse) << std::endl;
    std::cout << "AVX: ";
    std::cout << calculateAverage(avx) << std::endl;

    return 0;
}
