#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <vector>

/**
 * Perform a matrix matrix multiplication.
 */
void stridedAccess()
{
    // dimensions
    constexpr size_t M = 4096 * 4096;
    constexpr int iterations = 1;

    // vector y in R^M
    auto x = Kokkos::View<double *>("A", M * 8);

    for (auto i = 0; i < iterations; ++i)
    {
        auto start = std::chrono::system_clock::now();
        for (auto strideExp = 0; strideExp < 5; ++strideExp)
        {
            auto stride = (1 << strideExp);
            auto policy = Kokkos::RangePolicy<>(0, M);
            auto kernel = KOKKOS_LAMBDA(const int64_t &idx) { x(idx * stride) += 1; };
            Kokkos::parallel_for("stride_" + std::to_string(stride), policy, kernel);
            Kokkos::fence();
        }
        auto stop = std::chrono::system_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        std::cout << "runtime: " << runtime << " ms" << std::endl;
    }
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    stridedAccess();
    Kokkos::finalize();
    return EXIT_SUCCESS;
}