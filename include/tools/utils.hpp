#pragma once

#include <string>
#include <istream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>
#include <cctype>
#include <algorithm>
#include <mutex>
#include <execution>
#include <random>
#include <cstdio>

const auto exec_policy = std::execution::par_unseq;

std::mutex iomutex;

namespace sarma::utils {

    template <class Ordinal, class Real>
    auto get_prob(const Ordinal M, const Ordinal P, const Ordinal Q, const Real prob) {
        if (prob <= 1.0)
            return prob;
        return (P * 1.0 * prob) * (Q * 1.0 * prob) / M;
    };

    template<typename Value>
    auto trans(std::vector<std::vector<Value>> loads, bool symmetric) {
        std::vector<Value> r1(loads.size()), r2_original(loads[0].size());
        auto &r2 = symmetric ? r1 : r2_original;
        for (std::size_t i = 0; i < loads.size(); i++)
            for (std::size_t j = 0; j < loads[i].size(); j++) {
                r1[i] = std::max(r1[i], loads[i][j]);
                r2[j] = std::max(r2[j], loads[i][j]);
            }
        return symmetric ? std::vector<std::vector<Value>>{r1} : std::vector{r1, r2};
    }

    template <typename Iter, typename Int>
    std::size_t lowerbound_index(Iter begin, Iter end, Int v) {
        return std::distance(begin + 1, std::lower_bound(begin, end, v + 1));
    }

    template <typename Int>
    std::size_t lowerbound_index(const std::vector<Int> &v_arr, Int v) {
        return lowerbound_index(v_arr.cbegin(), v_arr.cend(), v);
    }

    template <typename Value>
    auto argmax(const std::vector<std::vector<Value>> &loads) {
        std::size_t i = 0, j = 0;
        for (std::size_t u = 0; u < loads.size(); u++)
            for (std::size_t v = 0; v < loads[u].size(); v++)
                if (loads[u][v] > loads[i][j]) {
                    i = u;
                    j = v;
                }
        return std::make_pair(i, j);
    }

    template <typename Value>
    auto max(const std::vector<std::vector<Value>> &loads) {
        const auto [i, j] = argmax(loads);
        return loads[i][j];
    }

    auto get_matrix_type(std::istream &in) {
        char s[4];
        in.read(s, sizeof s);
        assert(std::isdigit(s[0]));
        assert(std::isdigit(s[2]));
        s[1] = s[3] = '\0';
        return std::make_pair(std::stoi(s), std::stoi(s + 2));
    }

    auto sparse_mask(const size_t N, const double keep_prob = 1.0, const int seed = 1) {
        std::vector<bool> mask(N, false);
        static thread_local std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> rnd;

        if (keep_prob >= 1.0)
            mask.assign(N, true);
        else if (keep_prob > 1.0 / 3)
            std::generate(exec_policy, mask.begin(), mask.end(), [&]() {
                return rnd(gen) <= keep_prob;
            });
        else {
            const auto Q = log(1 - keep_prob);
            auto failure_count = [&]() -> std::size_t {
                return log(1 - rnd(gen)) / Q;
            };
            for (std::size_t i = -1;;) {
                i += std::min(failure_count(), mask.size());
                if (++i < mask.size())
                    mask[i] = true;
                else
                    break;
            }
        }
        return mask;
    }
}