#pragma once

#include <utility>
#include <vector>
#include <atomic>
#include <type_traits>

#include "tools/utils.hpp"

/**
* @brief This namespace contains required functions for
* Nicol's optimal one-dimensional partitioning algorithm.
* described in "David Nicol, Rectilinear Partitioning of Irregular Data Parallel Computations, JPDC, 1994".
*/
#if defined(ENABLE_CPP_PARALLEL)
namespace sarma::nicol1d {
#else
namespace sarma{
    namespace nicol1d {
#endif
    /**
     * @brief Implements 2D probe with 'version' choosing different optimizations.
     * Given 2D prefix sum array (Q by N), number of partitions (P = UB.size() - 1)
     * and the maximum load, this algorithm returns a valid cut vector of [0, N-1]
     * (last elements are >= N),if such partition exists, and
     * an unvalid one otherwise.
     *
     * @param prefixes Q of N prefix sum arrays
     * @param Q number of prefix sum arrays
     * @param L upper bound of load size
     * @param UB upper bound of each cut
     * @return a cut vector
     */
    template <class Ordinal, class Value, bool version = false>
    auto probe(std::vector<Value> const *prefixes, const Ordinal Q, const Value L, std::vector<Ordinal> *UB = nullptr) {
        const auto N = (Ordinal)prefixes[0].size() - 1;
        std::vector<Ordinal> p;
        if (UB)
            p.reserve(UB->size());

        p.push_back(0);
        while ((!UB && p.back() < N) || (UB && p.size() < UB->size())) {
#if defined(ENABLE_CPP_PARALLEL)
            if constexpr (version) {
                std::atomic<Ordinal> r = (UB ? (*UB)[p.size()] : N);
                std::for_each(exec_policy, prefixes, prefixes + Q, [&](const auto &prefix) {
                    auto l = p.back();
                    for (Ordinal n; l < (n = r.load(std::memory_order_acquire));) {
                        const auto m = (l + 1) + (n - (l + 1)) / 2;
                        const auto CL = prefix[m] - prefix[p.back()];
                        if (CL <= L)
                            l = m;
                        else
                            while(n > m - 1 && !r.compare_exchange_weak(n, m - 1, std::memory_order_release));
                    }
                });
                p.push_back(r);   
            }
            else {
                p.push_back(std::transform_reduce(exec_policy, prefixes, prefixes + Q, N, [](const auto &a, const auto &b) {
                    return std::min(a, b);
                }, [&](const auto &prefix) -> Ordinal {
                    auto l = p.back(), r = (UB ? (*UB)[p.size()] : N);
                    while (l < r) {
                        const auto m = (l + 1) + (r - (l + 1)) / 2;
                        const auto CL = prefix[m] - prefix[p.back()];
                        if (CL <= L)
                            l = m;
                        else
                            r = m - 1;
                    }
                    return l;
                }));
            }
#else
            std::atomic<Ordinal> r(0);
            if (UB){ r = (*UB)[p.size()]; }
            else{ r=N; }

            std::for_each(prefixes, prefixes + Q, [&](const auto &prefix) {
                auto l = p.back();
                for (Ordinal n; l < (n = r.load(std::memory_order_acquire));) {
                    const auto m = (l + 1) + (n - (l + 1)) / 2;
                    const auto CL = prefix[m] - prefix[p.back()];
                    if (CL <= L)
                        l = m;
                    else
                        while(n > m - 1 && !r.compare_exchange_weak(n, m - 1, std::memory_order_release));
                }
            });
            p.push_back(r);   
#endif
            // when there is no progress in cuts.
            if (p.back() == p[p.size() - 2]) {
                while(UB && p.size() < UB->size())
                    p.push_back(p.back());
                break;
            }

        }
        return p;
    }

    /**
     * @brief Implements standard 1D probe
     * Given prefix sum array, number of partitions and the maximum load,
     * this algorithm returns true if such partition exists, and false otherwise.
     *
     * @param prefix
     * @param P number of parts
     * @param B upper bound
     * @return whether feasiable cuts exists
     */
    template<class Ordinal, class Value>
    bool probe(const std::vector<Value> &prefix, const Ordinal P, const Value B) {
        std::vector<Ordinal> UB(P + 1, prefix.size() - 1);
        return (std::size_t)probe(&prefix, (Ordinal)1, B, &UB).back() >= prefix.size() - 1;
    }

    template <class Ordinal, class Value>
    auto partition_values(std::vector<Value> const * prefixes, const Ordinal Q, const Ordinal P) {
#if defined(ENABLE_CPP_PARALLEL)
        static_assert(std::is_integral_v<Value>, "Value template parameter has to be an integer!");
#else
        static_assert(std::is_integral<Value>::value, "Value template parameter has to be an integer!");
#endif
        const auto N = (Ordinal)prefixes->size() - 1;
        std::vector<Ordinal> UB(P + 1, N);
        auto l = std::accumulate(prefixes, prefixes + Q, (Value)0, [P](const auto a, const auto &prefix) {
            return std::max(a, (prefix.back() + P - 1) / P);
        }), r = std::accumulate(prefixes, prefixes + Q, (Value)0, [](const auto a, const auto &prefix) {
            return std::max(a, (prefix.back()));
        });
        while (l < r) {
            const auto m = l + (r - l) / 2;
            const auto p = probe(prefixes, Q, m, &UB);
            if (p.back() < N)
                l = m + 1;
            else {
                r = m;
                UB = p;
            }
        }
        return probe(prefixes, Q, l, &UB);
    }

    template <class Ordinal, class Value>
    auto partition_indices(std::vector<Value> const * prefixes, const Ordinal Q, const Ordinal P) {
        const Ordinal N = prefixes->size() - 1;
        std::vector<Ordinal> UB(P + 1, N);
        auto lower_bound = std::accumulate(prefixes, prefixes + Q, (Value)0, [P](const auto a, const auto &prefix) {
            return std::max(a, (prefix.back()) / P);
        }), upper_bound = std::accumulate(prefixes, prefixes + Q, (Value)0, [](const auto a, const auto &prefix) {
            return std::max(a, (prefix.back()));
        });
        auto compute_upper_bound = [&](const auto l, const auto r) {
            return std::accumulate(prefixes, prefixes + Q, (Value)0, [&](const auto a, const auto &prefix) {
                return std::max(a, prefix[r] - prefix[l]);
            });
        };

        Ordinal pback = 0;
        for (Ordinal i = 1; i <= P; i++) {
            auto l = pback, r = UB[i];
            while (l < r) {
                const auto m = (l + 1) + (r - (l + 1)) / 2;
                const auto b = compute_upper_bound(pback, m);
                if (b > upper_bound)
                    r = m - 1;
                else if (b < lower_bound)
                    l = m;
                else {
                    const auto p = probe(prefixes, Q, b, &UB);
                    if (p.back() < N) {
                        l = m;
                        lower_bound = b;
                    }
                    else {
                        r = m - 1;
                        UB = p;
                        upper_bound = b;
                    }
                }
            }
            const auto b = compute_upper_bound(pback, l + 1);
            if (lower_bound < b && b < upper_bound) {
                const auto p = probe(prefixes, Q, b, &UB);
                if (p.back() >= N) {
                    UB = p;
                    upper_bound = b;
                }
                else
                    lower_bound = b;
            }
            pback = l;
        }
        return probe(prefixes, Q, upper_bound, &UB);
    }

    /**
     * @brief Implements the helper function that computes optimum 1D
     * cut for given prefix sum.
     * The algorithm is described in "David Nicol, Rectilinear
     * Partitioning of Irregular Data Parallel Computations, JPDC,
     * 1994".
     *
     * @param prefix prefix sum for 1D
     * @param P number of part
     * @return cut vector and optimal bound
     * 
     */
    template <class Ordinal, class Value, bool use_indices = true>
    auto partition(std::vector<Value> const * prefixes, const Ordinal Q, const Ordinal P) {
#if defined(ENABLE_CPP_PARALLEL)
        if constexpr (use_indices)
#else
        if (use_indices)
#endif
            return partition_indices(prefixes, Q, P);
        else
            return partition_values(prefixes, Q, P);
    }

    /**
     * @brief Implements optimum 1D partitioning
     * The algorithm is described in "David Nicol, Rectilinear
     * Partitioning of Irregular Data Parallel Computations, JPDC,
     * 1994".
     *
     * @param prefix vector
     * @param P number of parts
     * @param on dimension that Matrix will be cut
     * @return cut vector and optimal bound
     *
     */
    template<class Ordinal, class Value>
    std::vector<Ordinal> partition_prefix(const std::vector<Value> &prefix,
                                                       const Ordinal P) {
        return partition(&prefix, (Ordinal)1, P);
    }

    /**
     * @brief Implements optimum 1D partitioning algorithm
     * described in "David Nicol, Rectilinear
     * Partitioning of Irregular Data Parallel Computations, JPDC,
     * 1994".
     *
     * @param array input 1D array
     * @param P number of part
     * @return cut vector and optimal bound
     *
     */
    template<class Ordinal, class Value>
    std::vector<Ordinal> partition(const std::vector<Value> &array,
                                                       const Ordinal P) {
        std::vector<Value> prefix(array.size() + 1, 0);

#if defined(ENABLE_CPP_PARALLEL)
        std::inclusive_scan(exec_policy, array.begin(), array.end(), prefix.begin() + 1);
#else
        for (size_t i=0; i<array.size(); i++)
            prefix[i+1] = prefix[i] + array[i];
#endif
        return partition_prefix(prefix, P);
    }
}
#if !defined(ENABLE_CPP_PARALLEL)
} // nested namespace
#endif