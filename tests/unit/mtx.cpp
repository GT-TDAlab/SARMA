#include "tests.h"

#include <vector>

#include "test_utils.hpp"
#include "data_structures/csr_matrix.hpp"

using namespace sarma;

int main() {
    int pass = 0;
    const auto matrices_pair = test_utils::get_test_mtx_bin_matrices<Ordinal, Value>();

    for (const auto &mp: matrices_pair)
        TEST(test_utils::are_same_mtx, mp.first, mp.second);

    const auto matrices = test_utils::get_test_matrices<Ordinal, Value>();
    for (const auto &mp: matrices)
        TEST(test_utils::is_valid_mtx, mp);

    return pass;
}
