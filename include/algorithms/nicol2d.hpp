#pragma once

#include <vector>
#include <cassert>
#include <utility>
#include <algorithm>
#include <functional>
#include <numeric>

#include "data_structures/csr_matrix.hpp"
#include "tools/utils.hpp"
#include "nicol1d.hpp"

/**
* @brief This namespace contains required functions for
* Nicol's two-dimensional partitioning algorithm
* described in "David Nicol, Rectilinear Partitioning of Irregular Data Parallel Computations, JPDC, 1994".
*/
namespace sarma::nicol2d {

    template <class Ordinal, class Value, bool use_indices = true>
    auto partition(const Matrix<Ordinal, Value> &A, const std::vector<Ordinal> &p, const Ordinal Q, const bool p_is_rows = true, int max_iteration = 0) {
        if (max_iteration == 0)
            max_iteration = std::max((Ordinal)10, (Ordinal)p.size() - 1 + Q);
        else if (max_iteration < 0)
            max_iteration = std::numeric_limits<int>::max();
        std::vector<std::vector<Ordinal>> ps = {p, p};
        const auto AT_ = A.transpose();
        const std::vector<Matrix<Ordinal, Value> const *> As = {&A, &AT_};
        const std::vector<Ordinal> Ps = {p_is_rows ? (Ordinal)p.size() - 1 : Q, p_is_rows ? Q : (Ordinal)p.size() - 1};
        std::vector<std::vector<std::vector<Value>>> prefixess;
        for (int i = 0; i < 2; i++)
            prefixess.emplace_back(Ps[~i & 1], std::vector<Value>(As[i & 1]->indptr.size(), 0));
        for (int i = p_is_rows ? 1 : 0; i <= max_iteration; i++) {
            const auto &A = *As[i & 1];
            auto &p = ps[i & 1];
            const auto P = Ps[i & 1];
            const auto &q = ps[~i & 1];
            auto &prefixes = prefixess[i & 1];
            for (auto &v: prefixes)
                std::fill(exec_policy, v.begin(), v.end(), 0);
            std::for_each(exec_policy, A.indptr.begin(), A.indptr.end() - 1, [&](const auto &indptr_i) {
                const auto i = std::distance(&A.indptr[0], &indptr_i);
                for (auto j = indptr_i; j < A.indptr[i + 1]; j++)
                    prefixes[utils::lowerbound_index(q, A.indices[j])][i + 1] += A.data(j);
            });
            for (auto &v: prefixes)
                std::inclusive_scan(exec_policy, v.begin(), v.end(), v.begin());
            const auto prev_p = p;
            p = nicol1d::partition<Ordinal, Value, use_indices>(&prefixes[0], (Ordinal)prefixes.size(), P);
            #ifdef DEBUG
                const auto o_p = nicol1d::partition<Ordinal, Value, !use_indices>(&prefixes[0], (Ordinal)prefixes.size(), P);
                assert(p == o_p);
                const auto l = A.compute_maxload(p, q);
                std::cerr << i << ' ' << l << std::endl;
                for (auto x: p)
                    std::cerr << x << ' ';
                std::cerr << std::endl;
                for (auto x: o_p)
                    std::cerr << x << ' ';
                std::cerr << std::endl;
            #endif
            if (p == prev_p)
                break;
        }
        return std::make_pair(ps[0], ps[1]);
    }

   /**
    * @brief Implements the rectilinear partitioning algorithm with
    * iterative refinement described in "David Nicol, Rectilinear
    * Partitioning of Irregular Data Parallel Computations, JPDC,
    * 1994".
    *
    * @param A Matrix
    * @param P number of parts
    * @param Q number of parts in the other dimension
    * @param max_iteration limit on refinement iterations
    * @return two cut vectors as a pair
    */
    template <class Ordinal, class Value, bool use_indices = true>
    auto partition(const Matrix<Ordinal, Value> &A, const Ordinal P, const Ordinal Q, const int max_iteration = 0) {
        const auto p = nicol1d::partition<Ordinal, Value, use_indices>(&A.indptr, (Ordinal)1, P);
        return partition<Ordinal, Value, use_indices>(A, p, Q, true, max_iteration);
    }
};
