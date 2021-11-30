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
        Kokkos::parallel_for(
            "transpose_teampolicy_" + layoutToString<MEMORY_LAYOUT>(), teamPolicy, teamKernel);

        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << "runtime: " << runtime << " ms" << std::endl;
}

template <typename TYPE, typename MEMORY_LAYOUT>
void transpose_smem()
{
    constexpr size_t MATRIX_SIZE = 4096;
    constexpr size_t TILE_SIZE = 32;
    constexpr size_t SHMEM_TILE_SIZE = 33;
    constexpr size_t NUM_TILES = (MATRIX_SIZE / TILE_SIZE);
    constexpr int iterations = 1;

    auto A = Kokkos::View<TYPE **, MEMORY_LAYOUT>("A", MATRIX_SIZE, MATRIX_SIZE);
    auto B = Kokkos::View<TYPE **, MEMORY_LAYOUT>("B", MATRIX_SIZE, MATRIX_SIZE);

    //        Kokkos::parallel_for(
    //            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {MATRIX_SIZE, MATRIX_SIZE}),
    //            KOKKOS_LAMBDA(const int64_t idx, const int64_t jdx) { A(idx, jdx) = idx; });
    //
    //        for (auto i = 0; i < MATRIX_SIZE; ++i)
    //        {
    //            for (auto j = 0; j < MATRIX_SIZE; ++j)
    //            {
    //                std::cout << A(i, j) << " ";
    //            }
    //            std::cout << std::endl;
    //        }

    auto start = std::chrono::system_clock::now();
    for (auto i = 0; i < iterations; ++i)
    {
        using shmem_t = Kokkos::View<double **,
                                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        size_t shmem_size = shmem_t::shmem_size(SHMEM_TILE_SIZE, SHMEM_TILE_SIZE);

        auto teamPolicy = Kokkos::TeamPolicy<>(NUM_TILES * NUM_TILES, 128)
                              .set_scratch_size(0, Kokkos::PerTeam(shmem_size));
        auto teamKernel = KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member)
        {
            auto tileIndex = team_member.league_rank();

            const int64_t tileOffsetI = (tileIndex % NUM_TILES) * TILE_SIZE;
            const int64_t tileOffsetJ = (tileIndex / NUM_TILES) * TILE_SIZE;

            auto shmem = shmem_t(team_member.team_scratch(0), SHMEM_TILE_SIZE, SHMEM_TILE_SIZE);

            {
                auto threadPolicy = Kokkos::TeamThreadRange(team_member, TILE_SIZE * TILE_SIZE);
                auto threadKernel = [&](const int64_t &index)
                {
                    const int64_t idx = (index % TILE_SIZE);
                    const int64_t jdx = (index / TILE_SIZE);

                    const auto i = tileOffsetI + idx;
                    const auto j = tileOffsetJ + jdx;
                    shmem(idx, jdx) = A(i, j);
                };
                Kokkos::parallel_for(threadPolicy, threadKernel);
            }

            team_member.team_barrier();

            {
                auto threadPolicy = Kokkos::TeamThreadRange(team_member, TILE_SIZE * TILE_SIZE);
                auto threadKernel = [&](const int64_t &index)
                {
                    const int64_t idx = (index / TILE_SIZE);
                    const int64_t jdx = (index % TILE_SIZE);

                    const auto i = tileOffsetI + idx;
                    const auto j = tileOffsetJ + jdx;
                    B(j, i) = shmem(idx, jdx);
                };
                Kokkos::parallel_for(threadPolicy, threadKernel);
            }
        };
        Kokkos::parallel_for(
            "transpose_shmem_" + layoutToString<MEMORY_LAYOUT>(), teamPolicy, teamKernel);
        Kokkos::fence();
    }
    auto stop = std::chrono::system_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    //        std::cout << std::endl << "=============================" << std::endl << std::endl;
    //        for (auto i = 0; i < MATRIX_SIZE; ++i)
    //        {
    //            for (auto j = 0; j < MATRIX_SIZE; ++j)
    //            {
    //                std::cout << B(i, j) << " ";
    //            }
    //            std::cout << std::endl;
    //        }

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