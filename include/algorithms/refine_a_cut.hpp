#pragma once

#include <fstream>
#include <string>

#include "nicol1d.hpp"
#include "data_structures/csr_matrix.hpp"

/**
* @brief This namespace contains required functions for
* the Refine a Cut (RaC) partitioning algorithm. RaC partitioning
* is one of the algorithms that is used to solve
* mLI problem. Its dual (sarma::refine_a_cut::bal) solves the mNC problem
*/
namespace sarma::refine_a_cut {

    /** An enum type used to define refinement direction. 
     */
    enum LOOKUP {
        DIM_X, /**< column direction */
        DIM_Y /**< row direction */
    };

    /**
    * @brief Implements a simplified refinement technique inspired from Nicol's 
    * refinement precedure. Here instead of running simultaneous binary searches
    * over prefix sum arrays of each interval we first compute the maximum of the
    * each interval than apply an optimal one-dimensional partitioning.
    *
    * @param A a matrix
    * @param prefix prefix sum array
    * @param prev_cuts cuts of the previous iteration
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template <typename Ordinal, typename Value>
    auto refinement(const Matrix<Ordinal, Value> &A, std::vector<Value> &prefix, const std::vector<Ordinal> &prev_cuts) {      

        prefix[0] = 0;
        std::for_each(exec_policy, prefix.begin(), prefix.end() - 1, [&](auto &prefix_i) {
            const auto i = std::distance(&prefix[0], &prefix_i);
            Ordinal max_i = 0, max_k = 0, k = 0;
            for (Ordinal j = A.indptr[i]; j < A.indptr[i + 1]; ++j) {
                while (A.indices[j] >= prev_cuts[k]){
                    max_i = std::max(max_i, max_k);
                    max_k = 0;
                    ++k;
                }
                max_k += A.data(j);
            }
            prefix[i+1] = std::max(max_i, max_k);
        });

        std::inclusive_scan(exec_policy, prefix.begin(), prefix.end(), prefix.begin());
        return nicol1d::partition_prefix<Ordinal, Value>(prefix, prev_cuts.size() - 1);
    }

    /**
    * @brief Implements Nicol's refinement precedure. This procedure runs simultaneous binary
    * searches over prefix sum arrays of each interval.
    *
    * @param A a matrix
    * @param prefix prefix sum array
    * @param prev_cuts cuts of the previous iteration
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template <typename Ordinal, typename Value>
    auto nic_refinement(const Matrix<Ordinal, Value> &A, std::vector<std::vector<Value>> &prefixes, const std::vector<Ordinal> &prev_cuts) {      

        for (auto &v: prefixes)
            std::fill(exec_policy, v.begin(), v.end(), 0);

        std::for_each(exec_policy, A.indptr.begin(), A.indptr.end() - 1, [&](const auto &indptr_i) {
            const auto i = std::distance(&A.indptr[0], &indptr_i);
            for (auto j = indptr_i; j < A.indptr[i + 1]; j++)
                prefixes[utils::lowerbound_index(prev_cuts, A.indices[j])][i + 1] += A.data(j);
        });

        for (auto &v: prefixes)
            std::inclusive_scan(exec_policy, v.begin(), v.end(), v.begin());

        return nicol1d::partition<Ordinal, Value>(&prefixes[0], (Ordinal)prefixes.size(), prev_cuts.size()-1);
    }

   /**
    * @brief Implements the Refine a Cut algorithm.
    * Applies row and column based mapping to map 2D into 1D
    * chooses the direction that gives better load imbalance.
    *
    * @param A Matrix
    * @param p starting cut vector after choosing the direction
    * @param pdf pick direction first if it is true.
    * @param max_iteration limit on refinement iterations
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template<class Ordinal, class Value>
    auto partition(const Matrix<Ordinal, Value> &A, const std::vector<Ordinal> &p, bool pdf=true, int max_iteration = 0, bool nic_ref=false) {
        if (max_iteration == 0)
            max_iteration = std::max(10, (int)p.size()-1);

        const auto AT_ = A.transpose().sort();
        const std::vector<Matrix<Ordinal, Value> const *> As = {&A, &AT_};

        std::vector<std::vector<Value>> prefixes;
        std::vector<Value> prefix;
        
        if(!nic_ref)
            prefix.resize(A.indptr.size());
        else
            prefixes.resize(p.size(), std::vector<Value>(A.indptr.size()));

        std::vector<Ordinal> cuts_x(p), cuts_y(p), cuts_(p);

        auto refine = [&](const LOOKUP on) {
            auto &cuts = (on == DIM_X) ? cuts_x : cuts_y;
            if (nic_ref)
                return nic_refinement<Ordinal>(*As[on], prefixes, cuts);
            return refinement<Ordinal>(*As[on], prefix, cuts);
        };

        auto max_load = [&](const LOOKUP on) {
            const auto &cuts = refine(on);
            return A.compute_maxload(cuts, cuts);
        };

        auto max_x = max_load(DIM_X);
        auto max_y = max_load(DIM_Y);
        auto max_best = A.NNZ();

        const auto first_dir = (max_x<=max_y) ? DIM_X : DIM_Y;
        const auto sec_dir = (max_x>max_y) ? DIM_X : DIM_Y;
        auto &cuts_first = (first_dir == DIM_X) ? cuts_x : cuts_y;
        auto &cuts_sec = (sec_dir == DIM_X) ? cuts_x : cuts_y;

        for (int i = 0; i < max_iteration; i++) {
            cuts_first = refine(first_dir);
            if (!pdf){
                max_x = A.compute_maxload(cuts_first, cuts_first);
                cuts_sec = refine(sec_dir);
                max_y = A.compute_maxload(cuts_sec, cuts_sec);

                if (max_x < max_y)
                    cuts_sec = cuts_first;
                else
                    cuts_first = cuts_sec;

                if (max_best > std::min(max_x, max_y)) {
                    max_best = std::min(max_x, max_y);
                    cuts_ = cuts_first;
                }
            }
        }
        if (pdf) cuts_ = cuts_first;
        return cuts_;
    };

    // bound a load function prototype 
    template <typename Ordinal, typename Value>
    auto bal(const Matrix<Ordinal, Value> &A, const Value Z);

   /**
    * @brief Implements the Refine a Cut (RaC) algorithm.
    * Applies row and column based mapping to map 2D into 1D
    * chooses the direction that gives better load imbalance.
    * Note that if Z is 
    * greater than zero then this function overrides P and calls
    * the dual of the uniform partitioning that solves MNc problem.
    * If pdf is false then in each iteration this algorithm applies
    * row-wise and column-wise refinement and picks the best cuts
    * for the next iteration.
    *
    * @param A Matrix
    * @param P number of cuts
    * @param Z target load
    * @param pdf pick direction first if it is true.
    * @param max_iteration limit on refinement iterations
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template<class Ordinal, class Value>
    std::vector<Ordinal> partition(const Matrix<Ordinal, Value> &A, const Ordinal P, const Value Z=0, bool pdf=true, int max_iteration = 0, bool nic_ref=false) {
        if (Z>0)
            return sarma::refine_a_cut::bal (A, Z);

        auto p = std::vector<Ordinal>(P + 1, A.N());
        p[ 0 ] = 0;
        return partition(A, p, pdf, max_iteration, nic_ref);
    };

   /**
    * @brief Implements Bound a Load (BaL) algorithm using RaC as
    * the mLI algorithm.
    *
    * @param A Matrix
    * @param Z Target load
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template <typename Ordinal, typename Value>
    auto bal(const Matrix<Ordinal, Value> &A, const Value Z) {
        Ordinal l = 1;
        Ordinal r = sarma::uniform::bal(A,Z).size()-1;
        while (l < r) {
            const auto m = (l + r) / 2;
            const auto cuts = sarma::refine_a_cut::partition(A, m);
            const auto ld = A.compute_maxload(cuts, cuts);
            if (ld > Z)
                l = m + 1;
            else
                r = m;
        }
        return sarma::refine_a_cut::partition(A, l);
    }
}
