#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <type_traits>

#include "data_structures/csr_matrix.hpp"
#include "data_structures/sparse_prefix_sum.hpp"

#include "tools/utils.hpp"
#include "tools/timer.hpp"

#include "nicol2d.hpp"
#include "probe_a_load.hpp"
#include "ordered_probe_a_load.hpp"
#include "uniform.hpp"
#include "refine_a_cut.hpp"

#ifdef GUROBI_FOUND
#include "mixed_integer_program.hpp"
#endif

namespace sarma {
    template <typename Ordinal, typename Value>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> uni(const Matrix<Ordinal, Value> &A,
                const Ordinal P, const Ordinal /*Q*/, const Value Z, const int /*seed*/) {
        const auto cuts = uniform::partition(A, P, Z);
        return {cuts, cuts};
    }

    template <typename Ordinal, typename Value, bool use_indices = true>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> nic(const Matrix<Ordinal, Value> &A,
            const Ordinal P, const Ordinal Q, const Value /* Z */, const int /*seed*/) {
        return nicol2d::partition<Ordinal, Value, use_indices>(A, P, Q == 0 ? P : Q);
    }

    template <typename Ordinal, typename Value>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> rac(const Matrix<Ordinal, Value> &A,
            const Ordinal P, const Ordinal /* Q */, const Value Z, const int /*seed*/) {
        const auto p = refine_a_cut::partition<Ordinal, Value>(A, P, Z);
        return {p, p};
    }

    template <typename Ordinal, typename Value>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> pal(const Matrix<Ordinal, Value> &A,
            const Ordinal P, const Ordinal /*Q*/, const Value Z, const int /*seed*/) {
        const auto cuts = Z == 0 ? probe_a_load::partition(A, P) : probe_a_load::probe(A, Z);
        return {cuts, cuts};
    }

    template <typename Ordinal, typename Value>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> opal(const Matrix<Ordinal, Value> &A,
            const Ordinal P, const Ordinal /*Q*/, const Value Z, const int /*seed*/) {
        const auto cuts = Z == 0 ? ordered_probe_a_load::partition(A, P) : ordered_probe_a_load::probe(A, Z);
        return {cuts, cuts};
    }

#ifdef GUROBI_FOUND
    template <typename Ordinal, typename Value>
    std::pair<std::vector<Ordinal>, std::vector<Ordinal>> mip(const Matrix<Ordinal, Value> &A,
            const Ordinal P, const Ordinal Q, const Value /*Z*/, const int /*seed*/) {
        if (Q == 0) {
            const auto cuts = mixed_integer_program::partition<true>(A, P, P);
            return {cuts, cuts};
        }
        else
            return mixed_integer_program::partition<false>(A, P, Q);
    }
#endif

    template <typename Ordinal, typename Value>
    constexpr auto get_algorithm_vec() {

        std::vector<std::pair<std::string, std::pair<std::function<std::pair<std::vector<Ordinal>, std::vector<Ordinal>>(
             const Matrix<Ordinal, Value> &, const Ordinal, const Ordinal, const Value, const int)>, bool>>> alg_vec;

        alg_vec.push_back({"uni", {uni<Ordinal, Value>, false}});
        alg_vec.push_back({"nic", {nic<Ordinal, Value>, false}});
#if defined(ENABLE_CPP_PARALLEL)
        if constexpr (std::is_integral_v<Value>)
#else
        if (std::is_integral<Value>::value)
#endif
            alg_vec.push_back({"nic_v", {nic<Ordinal, Value, false>, false}});

        alg_vec.push_back({"rac", {rac<Ordinal, Value>, false}});
        alg_vec.push_back({"pal", {pal<Ordinal, Value>, true}});
        alg_vec.push_back({"opal", {opal<Ordinal, Value>, false}});
#ifdef GUROBI_FOUND
        alg_vec.push_back({"mip", {mip<Ordinal, Value>, false}});
#endif
        return alg_vec;
    }

    template <typename Ordinal, typename Value>
    constexpr auto get_algorithm_map() {

        std::map<std::string, std::pair<std::function<std::pair<std::vector<Ordinal>, std::vector<Ordinal>>(
             const Matrix<Ordinal, Value> &, const Ordinal, const Ordinal, const Value, const int)>, bool>> alg_list;

#if defined(ENABLE_CPP_PARALLEL)
        for (auto &[str, alg]: get_algorithm_vec<Ordinal, Value>()){
#else
        for (auto &av: get_algorithm_vec<Ordinal, Value>()){
            auto &str = av.first;
            auto &alg = av.second;
#endif
            alg_list.insert({str, alg});
        }

        return alg_list;
    }

    auto get_order_vec() {
        return std::vector<std::pair<std::string, Order>> {
            {"nat", Order::NAT}, {"asc", Order::ASC}, {"dsc", Order::DSC}, {"rcm", Order::RCM}, {"rnd", Order::RND}};
    }

    auto get_order_map() {
        std::map<std::string, Order> order_map;
#if defined(ENABLE_CPP_PARALLEL)
        for (auto &[str, order] : get_order_vec()){
#else
        for (auto &ar : get_order_vec()){
            auto &str = ar.first;
            auto &order = ar.second;
#endif
            order_map.insert({str, order});
        }
        return order_map;
    }

    template <class partitioner_t, class Ordinal, class Value>
    auto AlgorithmPartition(
            partitioner_t partitioner,
            std::shared_ptr<Matrix<Ordinal, Value>> A_ord,
            Ordinal P,
            Ordinal Q,
            Ordinal Z,
            int seed) {
        return partitioner(*A_ord, P, Q, Z, seed);
    }

    template <class partitioner_t, class Ordinal, class Value>
    auto AlgorithmRun(
            partitioner_t partitioner,
            std::ostream &out,
            std::shared_ptr<Matrix<Ordinal, Value>> A_ord,
            std::shared_ptr<Matrix<Ordinal, Value>> A_sp,
            Ordinal P,
            Ordinal Q,
            Ordinal Z,
            bool /*serialize*/,
            bool use_sparse,
            int seed,
            double sp_time,
            double ds_time) {

        timer t("Partitioning");
#if defined(ENABLE_CPP_PARALLEL)
        auto [p, q] = AlgorithmPartition<partitioner_t, Ordinal, Value>(partitioner, A_sp, P, Q, Z, seed);
#else
        auto pq = AlgorithmPartition<partitioner_t, Ordinal, Value>(partitioner, A_sp, P, Q, Z, seed);
        auto p = pq.first;
        auto q = pq.second;
#endif
        auto par_time = t.time();

        A_ord->print(out, p, q,
                    (p == q), sp_time,
                    use_sparse ? ds_time : 0,
                    par_time);

        return std::make_pair(p, q);
    }

    template <class partitioner_t, class Ordinal, class Value>
    std::pair<double, double> AlgorithmBenchmark(
            partitioner_t partitioner,
            std::shared_ptr<Matrix<Ordinal, Value>> A_ord,
            std::shared_ptr<Matrix<Ordinal, Value>> A_sp,
            Ordinal P,
            Ordinal Q,
            Ordinal Z,
            int seed) {

        timer t("Partitioning");
#if defined(ENABLE_CPP_PARALLEL)
        auto [p, q] = AlgorithmPartition<partitioner_t, Ordinal, Value>(partitioner, A_sp, P, Q, Z, seed);
#else
        auto pq = AlgorithmPartition<partitioner_t, Ordinal, Value>(partitioner, A_sp, P, Q, Z, seed);
        auto p = pq.first;
        auto q = pq.second;
#endif
        auto par_time = t.time();
        const auto loads = A_ord->compute_loads(p, q);
        const auto max_load = utils::max(loads);
        const auto avg_num_edges = A_ord->indptr.back() * 1.0 / (p.size() - 1) / (q.size() - 1);
        const auto load_imbalance = max_load * 1.0 / avg_num_edges;
        return {load_imbalance, par_time};
    }

    template <typename Ordinal, typename Value>
    auto AlgorithmOrder(std::string mname, Order order_type, bool triangular, bool use_data) {
        // std::cerr << "# Graph : " << mname << "\n" << "# Order : " << order_type << std::endl;

        auto A_org = std::make_shared<Matrix<Ordinal, Value>>(mname, use_data);
        auto A_ord = std::make_shared<Matrix<Ordinal, Value>>(A_org->order(order_type,
                                                            triangular, true));
        A_ord->use_data(use_data);

        std::vector<std::string> order_info = {"ascending degree", "descending degree",
                                            "reverse cuthill mckee", "natural"};
        // std::cerr << "# Ordering: " << order_info[order_type] << std::endl;
        return A_ord;
    }

    template <typename Ordinal, typename Value>
    auto AlgorithmOrder(std::shared_ptr<Matrix<Ordinal, Value> > mtx, Order order_type, bool triangular, bool use_data) {
        auto A_ord = std::make_shared<Matrix<Ordinal, Value> >(mtx->order(order_type, triangular, true));
        A_ord->use_data(use_data);
        return A_ord;
    }

    template <typename Ordinal, typename Value, typename Real = double>
    auto AlgorithmPlan(std::shared_ptr<Matrix<Ordinal, Value>> A_ord, Real sparsify, int seed, bool use_sparse) {
        auto A_sp = A_ord;
        double sp_time = 0;
        if (sparsify < 1.0) {
            timer t("Sparsification");
            A_sp = std::make_shared<Matrix<Ordinal, Value>>(A_ord->sparsify(sparsify, seed));
            sp_time = t.time();
        }

        timer t("Data-structure Construction");
        if (use_sparse)
            A_sp->get_sps();
        auto ds_time = t.time();

        return make_tuple(A_sp, sp_time, ds_time);
    }

    template <typename Ordinal, typename Value, class partitioner_t, typename Real = double>
    auto Run(
            partitioner_t partitioner,
            std::ostream &out,
            std::string mname,
            Order order_type,
            Ordinal P,
            Ordinal Q,
            Ordinal Z,
            bool triangular,
            bool serialize,
            Real sparsify,
            bool use_sparse,
            bool use_data,
            int seed) {
        auto a_ord = AlgorithmOrder<Ordinal, Value>(mname, order_type, triangular, use_data);
        sparsify = utils::get_prob(a_ord->NNZ(), P, Q == 0 ? P : Q, sparsify);
#if defined(ENABLE_CPP_PARALLEL)
        auto [a_sp, sp_time, ds_time] = AlgorithmPlan<Ordinal, Value>(a_ord, sparsify, seed, use_sparse);
#else
        auto terms = AlgorithmPlan<Ordinal, Value>(a_ord, sparsify, seed, use_sparse);
        auto a_sp = std::get<0>(terms);
        auto sp_time = std::get<1>(terms);
        auto ds_time = std::get<2>(terms);
#endif
        return AlgorithmRun<partitioner_t, Ordinal, Value>(partitioner, out, a_ord, a_sp, P, Q, Z,
                                                    serialize, use_sparse, seed, sp_time, ds_time);
    }

    template <typename Ordinal, typename Value, class partitioner_t, typename Real = double>
    auto Run(partitioner_t partitioner,
             std::ostream &out,
             std::shared_ptr<Matrix<Ordinal, Value> > mtx,
             Order order_type,
             Ordinal P,
             Ordinal Q,
             Ordinal Z,
             bool triangular,
             bool serialize,
             Real sparsify,
             bool use_sparse,
             bool use_data,
             int seed) {
        auto a_ord = AlgorithmOrder<Ordinal, Value>(mtx, order_type, triangular, use_data);
        sparsify = utils::get_prob(a_ord->NNZ(), P, Q == 0 ? P : Q, sparsify);
#if defined(ENABLE_CPP_PARALLEL)
        auto [a_sp, sp_time, ds_time] = AlgorithmPlan<Ordinal, Value>(a_ord, sparsify, seed, use_sparse);
#else
        auto terms = AlgorithmPlan<Ordinal, Value>(a_ord, sparsify, seed, use_sparse);
        auto a_sp = std::get<0>(terms);
        auto sp_time = std::get<1>(terms);
        auto ds_time = std::get<2>(terms);
#endif
        return AlgorithmRun<partitioner_t, Ordinal, Value>(partitioner, out, a_ord, a_sp, P, Q, Z,
                                                           serialize, use_sparse, seed, sp_time, ds_time);
    }
}
