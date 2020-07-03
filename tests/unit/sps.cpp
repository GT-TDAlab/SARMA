#include <vector>
#include <random>
#include <algorithm>

#include "data_structures/csr_matrix.hpp"
#include "data_structures/sparse_prefix_sum.hpp"

#include "test_utils.hpp"

using namespace sarma;

const auto num_iters = 100;

int main() {
    int pass = 0;
    const auto matrices = test_utils::get_test_matrices<Ordinal, Value>();

    std::default_random_engine gen(1);

    for (const auto &A: matrices) {
        const sarma::sparse_prefix_sum sps(A);
        for (int i = 0; i < num_iters; i++) {
            std::vector<std::vector<Ordinal>> pq(2);
            for (auto &p: pq) {
                p.push_back(0);
                auto P = std::uniform_int_distribution<Ordinal>(0, 16)(gen);
                while (P--)
                    p.push_back(std::uniform_int_distribution<Ordinal>(0, A.N())(gen));
                p.push_back(A.N());
                std::sort(p.begin(), p.end());
            }
            pass |= A.compute_loads(pq[0], pq[1]) != sps.compute_loads(pq[0], pq[1]);
        }
    }

    return pass;
}
