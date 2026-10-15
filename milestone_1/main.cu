#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <numbers>
#include "pimontecarlo.hpp"

int main()
{
    long long TP = 100;

    std::cout << std::format(
        "{:<14} {:<14} {:<14} {:<14} {:<14} {:<14} {:<14} {:<10}\n",
        "N",
        "Pi (CPU)",
        "Pi (GPU)",
        "Error (CPU)",
        "Error (GPU)",
        "CPU Time (s)",
        "GPU Time (s)",
        "Speedup");

    std::cout << std::format(
        "{:-<14} {:-<14} {:-<14} {:-<14} {:-<14} {:-<14} {:-<14} {:-<10}\n",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "");

    for (int i = 0; i < 8; ++i)
    {
        double cpu_time = 0.0;
        double gpu_time = 0.0;

        double pi_cpu = estimatePIMonteCarloCPU(TP, cpu_time);
        // double pi_gpu = estimatePIMonteCarloGPU(TP, gpu_time);
        double pi_gpu = 0;

        double error_cpu = std::abs(pi_cpu - std::numbers::pi);
        double error_gpu = std::abs(pi_gpu - std::numbers::pi);
        double speedup = cpu_time / gpu_time;

        std::cout << std::format(
            "{:<14} {:<14.10f} {:<14.10f} {:<14.10f} {:<14.10f} {:<14.6f} {:<14.6f} {:<10.1f}\n",
            TP,
            pi_cpu,
            pi_gpu,
            error_cpu,
            error_gpu,
            cpu_time,
            gpu_time,
            speedup);

        TP *= 10;
    }

    return EXIT_SUCCESS;
}