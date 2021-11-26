#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <type_traits>

/**
 * https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */

static const std::string LAYOUT_LEFT = "LayoutLeft";
static const std::string LAYOUT_RIGHT = "LayoutRight";

template <typename MEMORY_LAYOUT>
constexpr const std::string &layoutToString()
{
    if constexpr (std::is_same_v<MEMORY_LAYOUT, Kokkos::LayoutLeft>) return LAYOUT_LEFT;
    if constexpr (std::is_same_v<MEMORY_LAYOUT, Kokkos::LayoutRight>) return LAYOUT_RIGHT;
}

/**
 * Check performance of strided memory access.
 */
template <typename TYPE, typename MEMORY_LAYOUT>
void copy()
{
    constexpr size_t M = 4096;
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

template <typename TYPE, typename MEMORY_LAYOUT>
void transpose()
{
    constexpr size_t M = 4096;
    constexpr int iterations = 1;

    auto A = Kokkos::View<TYPE **, MEMORY_LAYOUT>("A", M, M);
    auto B = Kokkos::View<TYPE **, MEMORY_LAYOUT>("B", M, M);

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, M}, {128, 0});
        auto kernel = KOKKOS_LAMBDA(const int64_t &idx, const int64_t &jdx)
        {
            B(jdx, idx) = A(idx, jdx);
        };
        Kokkos::parallel_for("transpose_" + layoutToString<MEMORY_LAYOUT>(), policy, kernel);
        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

template <typename TYPE, typename MEMORY_LAYOUT>
void transpose_team_policy()
{
    constexpr size_t M = 4096;
    constexpr int iterations = 1;

    auto A = Kokkos::View<TYPE **, MEMORY_LAYOUT>("A", M, M);
    auto B = Kokkos::View<TYPE **, MEMORY_LAYOUT>("B", M, M);

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        auto teamPolicy = Kokkos::TeamPolicy<>(M, Kokkos::AUTO());
        auto teamKernel = KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member)
        {
            auto idx = team_member.league_rank();

            auto policy = Kokkos::TeamThreadRange(team_member, M);
            auto kernel = [=](const int64_t &jdx) { B(idx, jdx) = A(jdx, idx); };
            Kokkos::parallel_for(policy, kernel);
        };
        Kokkos::parallel_for("transpose_teampolicy_" + layoutToString<MEMORY_LAYOUT>(), teamPolicy, teamKernel);

        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

template <typename TYPE, typename MEMORY_LAYOUT>
void transpose_smem()
{
    constexpr size_t M = 4096;
    constexpr size_t TILE = 32;
    constexpr size_t TILES = (M / TILE);
    constexpr int iterations = 1;

    auto A = Kokkos::View<TYPE **, MEMORY_LAYOUT>("A", M, M);
    auto B = Kokkos::View<TYPE **, MEMORY_LAYOUT>("B", M, M);

    //    Kokkos::parallel_for(
    //        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, M}),
    //        KOKKOS_LAMBDA(const int64_t idx, const int64_t jdx) { A(idx, jdx) = idx; });

    //    for (auto i = 0; i < M; ++i)
    //    {
    //        for (auto j = 0; j < M; ++j)
    //        {
    //            std::cout << A(i, j) << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        using shmem_t = Kokkos::View<double **,
                                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        size_t shmem_size = shmem_t::shmem_size(TILE, TILE);

        auto teamPolicy = Kokkos::TeamPolicy<>(TILES * TILES, 256)
                              .set_scratch_size(0, Kokkos::PerTeam(shmem_size));
        auto teamKernel = KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member)
        {
            auto idx = team_member.league_rank();

            const int64_t offsetI = (idx % TILES) * TILE;
            const int64_t offsetJ = (idx / TILES) * TILE;

            auto shmem = shmem_t(team_member.team_scratch(0), TILE, TILE);

            {
                auto threadPolicy = Kokkos::TeamThreadRange(team_member, TILE);
                auto threadKernel = [&](const int64_t &idx)
                {
                    const auto i = offsetI + idx;
                    auto vectorPolicy = Kokkos::ThreadVectorRange(team_member, TILE);
                    auto vectorKernel = [&](const int64_t &jdx)
                    {
                        const auto j = offsetJ + jdx;
                        shmem(idx, jdx) = A(i, j);
                    };
                    Kokkos::parallel_for(vectorPolicy, vectorKernel);
                };
                Kokkos::parallel_for(threadPolicy, threadKernel);
            }

            team_member.team_barrier();

            {
                auto threadPolicy = Kokkos::TeamThreadRange(team_member, TILE);
                auto threadKernel = [&](const int64_t &jdx)
                {
                    const auto j = offsetJ + jdx;
                    auto vectorPolicy = Kokkos::ThreadVectorRange(team_member, TILE);
                    auto vectorKernel = [&](const int64_t &idx)
                    {
                        const auto i = offsetI + idx;
                        B(j, i) = shmem(idx, jdx);
                    };
                    Kokkos::parallel_for(vectorPolicy, vectorKernel);
                };
                Kokkos::parallel_for(threadPolicy, threadKernel);
            }
        };
        Kokkos::parallel_for("transpose_shmem_" + layoutToString<MEMORY_LAYOUT>(), teamPolicy, teamKernel);
        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    //    std::cout << std::endl << "=============================" << std::endl << std::endl;
    //    for (auto i = 0; i < M; ++i)
    //    {
    //        for (auto j = 0; j < M; ++j)
    //        {
    //            std::cout << B(i, j) << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    //    copy<double, Kokkos::LayoutLeft>();
    //    copy<double, Kokkos::LayoutRight>();
    transpose<double, Kokkos::LayoutLeft>();
    transpose_team_policy<double, Kokkos::LayoutLeft>();
    transpose_smem<double, Kokkos::LayoutLeft>();
    Kokkos::finalize();
    return EXIT_SUCCESS;
}