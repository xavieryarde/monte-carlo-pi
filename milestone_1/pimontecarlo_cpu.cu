#include <random>
#include <chrono>
#include "pimontecarlo.h"

namespace
{
    double estimatePIMonteCarloImpl(long long TP)
    {
        long long PIC = 0;

        // std::random_device rd;
        std::mt19937 gen(12345);

        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (long long i = 0; i < TP; ++i)
        {
            double x = dist(gen);
            double y = dist(gen);

            if ((x * x) + (y * y) <= 1.0)
            {
                PIC++;
            }
        }

        double pi_estimate = 4.0 * static_cast<double>(PIC) / static_cast<double>(TP);

        return pi_estimate;
    }
}

double estimatePIMonteCarloCPU(long long TP, double &elapsedSeconds)
{
    auto start = std::chrono::steady_clock::now();
    double pi_estimate = estimatePIMonteCarloImpl(TP);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    elapsedSeconds = elapsed.count();

    return pi_estimate;
}