#include "tests.h"

#include <vector>

#include "test_utils.hpp"
#include "data_structures/csr_matrix.hpp"
#include "tools/timer.hpp"

using namespace sarma;

int main() {
    int pass = 0;
    const auto matrices = test_utils::get_test_matrices<Ordinal, Value>();
    
    for (const auto &mtx: matrices){
        const auto A_t = mtx.transpose().sort();
        const auto A_t_t = A_t.transpose().sort();
        const auto A = A_t_t.transpose().sort();
        TEST (test_utils::are_same_mtx, mtx, A_t_t);
        TEST (test_utils::are_same_mtx, A_t, A);
    }

    return pass;
}
