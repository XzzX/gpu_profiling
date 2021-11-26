#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <type_traits>

static const std::string LAYOUT_LEFT = "LayoutLeft";
static const std::string LAYOUT_RIGHT = "LayoutRight";

template <typename MEMORY_LAYOUT>
constexpr const std::string& layoutToString()
{
    if constexpr (std::is_same_v<MEMORY_LAYOUT, Kokkos::LayoutLeft>)
        return LAYOUT_LEFT;
    if constexpr (std::is_same_v<MEMORY_LAYOUT, Kokkos::LayoutRight>)
        return LAYOUT_RIGHT;
}

/**
 * Check performance of strided memory access.
 *
 * https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */
template <typename TYPE, typename MEMORY_LAYOUT>
void copy()
{
    constexpr size_t M = 4096 * 2;
    constexpr int iterations = 1;

    auto A = Kokkos::View<TYPE **, MEMORY_LAYOUT>("A", M, M);
    auto B = Kokkos::View<TYPE **, MEMORY_LAYOUT>("B", M, M);

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, M}, {128, 0});
        auto kernel = KOKKOS_LAMBDA(const int64_t &idx, const int64_t &jdx)
        {
            B(idx, jdx) = A(idx, jdx);
        };
        Kokkos::parallel_for("copy_" + layoutToString<MEMORY_LAYOUT>(), policy, kernel);
        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    copy<double, Kokkos::LayoutLeft>();
    copy<double, Kokkos::LayoutRight>();
    Kokkos::finalize();
    return EXIT_SUCCESS;
}