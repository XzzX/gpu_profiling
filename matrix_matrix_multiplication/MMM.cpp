#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <vector>

/**
 * Perform a matrix matrix multiplication.
 */
void MMM()
{
    // dimensions
    constexpr size_t M = 1024;
    constexpr int iterations = 1;

    // vector y in R^M
    auto A = Kokkos::View<double**>("A", M, M);
    // matrix A in R^MxN
    auto B = Kokkos::View<double**>("B", M, M);
    // vector x in R^N
    auto C = Kokkos::View<double**>("C", M, M);

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        double res = 0;
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, M});
        auto kernel = KOKKOS_LAMBDA(const int64_t& idx, const int64_t& jdx)
        {
            C(idx, jdx) = 0;
            for (auto i = 0; i<M; ++i)
            {
                C(idx, jdx) += A(idx, i) * B(i, jdx);
            }
        };
        Kokkos::parallel_for("MMM", policy, kernel);
        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    MMM();
    Kokkos::finalize();
    return EXIT_SUCCESS;
}