#include "dotProduct.h"
#include "adam.h"
#include "gradientsUpdate.h"

int main() {
    /*
     * TESTING
     */
    const int MAX_ITERATIONS = 1000;

    std::vector<double> slow;
    std::vector<double> sse;
    std::vector<double> avx;

    int count = 0;
    while(count < MAX_ITERATIONS) {
        std::vector<double> tempDot = getDotProductResult();
        std::vector<double> tempAdam = getAdamResult();
        std::vector<double> tempGradientsUpdate = getGradientsUpdateResult();

        slow.push_back(tempDot[0]);
        slow.push_back(tempAdam[0]);
        slow.push_back(tempGradientsUpdate[0]);

        sse.push_back(tempDot[1]);
        sse.push_back(tempAdam[1]);
        sse.push_back(tempGradientsUpdate[1]);

        avx.push_back(tempDot[2]);
        avx.push_back(tempAdam[2]);
        avx.push_back(tempGradientsUpdate[2]);

        count++;
    }

    std::cout << calculateAverage(slow) << std::endl;
    std::cout << calculateAverage(sse) << std::endl;
    std::cout << calculateAverage(avx) << std::endl;

    return 0;
}
