#include "tests.h"

#include <vector>

#include "test_utils.hpp"
#include "data_structures/csr_matrix.hpp"
#include "algorithms/refine_a_cut.hpp"

using namespace sarma;

int main() {
    int pass = 0;
    const auto matrices = test_utils::get_test_matrices<Ordinal, Value>();
    
    TEST(test_utils::test_alg_on_mtxs, matrices, "rac", true);

    return pass;
}
