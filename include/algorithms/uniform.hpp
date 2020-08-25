#pragma once

/**
* @brief This namespace contains required functions for
* the uniform (checker-board) partitioning. Uniform partitioning
* is one of the straightforward algorithms that is used to solve
* mLI problem. Its dual (sarma::uniform::bal) solves the mNC problem
*/
#if defined(ENABLE_CPP_PARALLEL)
namespace sarma::uniform {
#else
namespace sarma{
    namespace uniform {
#endif
    /**
    * @brief Implements the uniform (checker-board) partitioning
    * taking number of rows and number of cuts as parameters.
    *
    * @param N number of rows
    * @param P number of parts
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @return a cut vector
    */
    template<class Ordinal>
    std::vector<Ordinal> partition(const Ordinal N, const Ordinal P) {
        std::vector<Ordinal> p;
        p.reserve(P + 1);
        for (Ordinal i = 0; i <= P; i++)
            p.push_back(i * 1ll * N / P);
        return p;
    }

    template <class Ordinal, class Value>
    auto bal(const Matrix<Ordinal, Value> &A, const Value Z);

    /**
    * @brief Implements the Uniform (Uni) checker-board partitioning
    * It basically returns equally spaced cuts. Note that if Z is 
    * greater than zero then this function overrides P and calls
    * the dual of the uniform partitioning that solves MNc problem.
    *
    * @param A Matrix
    * @param P number of parts
    * @param Z Target load
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @return a cut vector
    */
    template<class Ordinal, class Value>
    std::vector<Ordinal> partition(const Matrix<Ordinal, Value> &A, const Ordinal P, const Value Z=0) {
        if (Z>0)
            return bal (A, Z);
        return partition(A.N(), P);
    }

   /**
    * @brief Implements Bound a Load (BaL) algorithm using UNI as
    * the mLI algorithm.
    *
    * @param A Matrix
    * @param Z Target load
    * @tparam Ordinal data-type of indptr and indices structures in sarma::Matrix
    * @tparam Value data-type of data structure in sarma::Matrix
    * @return a cut vector
    */
    template <class Ordinal, class Value>
    auto bal(const Matrix<Ordinal, Value> &A, const Value Z) {
        auto l = (Ordinal)1;
        Ordinal r = A.N() / std::sqrt(Z) + 1;
        while (l < r) {
            const auto m = l + (r - l) / 2;
            const auto cuts = sarma::uniform::partition(A.N(), m);
            const auto ld = A.compute_maxload(cuts, cuts);
            if (ld > Z)
                l = m + 1;
            else
                r = m;
        }
        return sarma::uniform::partition(A.N(), l);
    }
}
#if !defined(ENABLE_CPP_PARALLEL)
} // nested namespace
#endif