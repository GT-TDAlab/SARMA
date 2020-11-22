#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
#include <random>
#include <iterator>
#include <functional>
#include <string>
#include <tuple>
#include <cctype>
#include <map>
#include <set>

#include "data_structures/sparse_prefix_sum.hpp"
#include "data_structures/csr_matrix.hpp"
/**
* @brief This namespace contains required functions for
* the Subgradient Optimization algorithm. This algorithm
* requires the sparse-prefix-sum data-structure.
**/
#if defined(ENABLE_CPP_PARALLEL)
namespace sarma::subgradient_method {
#else
namespace sarma{
    namespace subgradient_method {
#endif
    template <typename Int, typename Value>
    auto uniform_init(const Int P, const Value total_load) {
        std::vector<double> pi(P - 1, 0.0);
        for (Int i = 0; i < pi.size(); i++)
            pi[i] = total_load * (i + 1.0) / P;
        return pi;
    }

    template <typename Int, typename Value, typename Gen>
    auto random_init(const Int P, const Value total_load, Gen &gen) {
        std::uniform_real_distribution<double> rnd(0, total_load);
        std::vector<double> pi(P - 1, 0.0);
        for (auto &x: pi)
            x = rnd(gen);
        std::sort(std::begin(pi), std::end(pi));
        return pi;
    }

    /**
    * @brief Implements the SubGradient Optimization algorithm.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load.
    * 
    * @param pis vector of cut vector parametrizations
    * @param to_rs lambda function to compute gradients given pis
    * @param C number of iterations to wait when objective value flattens
    * @param eps the improvement in the optimization objective expected
    * @return vector of cut vector parametrizations
    */
    template <typename pis_to_rs_t>
    auto partition(std::vector<std::vector<double>> &pis, pis_to_rs_t to_rs, const std::size_t C = 10, const double eps = 0.001) {
        using Value = typename decltype(to_rs(pis))::value_type::value_type;
        const auto t1 = std::vector<double>(pis.size(), 1);
        auto t = [&](std::size_t m, std::size_t i) -> double {
            return t1[i] / std::sqrt(m + 100);
        };
        auto best_pis = std::make_pair(pis, std::numeric_limits<Value>::max());
        std::size_t p_best_m = 1;
        auto p_best_pis = best_pis;
        for (std::size_t m = 1;; m++) {
            const auto rs = to_rs(pis);

            const auto maxLoad = *std::max_element(std::begin(rs[0]), std::end(rs[0]));
            if (best_pis.second > maxLoad)
                best_pis = {pis, maxLoad};
                
            auto minLoad = std::numeric_limits<Value>::max();
            for (std::size_t k = 0; k < rs.size(); k++) {
                std::vector<double> u;
                const auto avg = std::accumulate(std::begin(rs[k]), std::end(rs[k]), (double)0) / rs[k].size();
                std::transform(std::begin(rs[k]), std::end(rs[k]), std::back_inserter(u), [avg] (const auto r_i) {
                    return r_i - avg;
                });
                std::partial_sum(std::begin(u), std::end(u), std::begin(u));
                std::for_each(std::begin(pis[k]), std::end(pis[k]), [&](auto &pi_i) {
                    const auto i = std::distance(pis[k].data(), &pi_i);
                    pi_i -= t(m / (pis[k].size() + 2), k) * u[i];
                });
                std::sort(std::begin(pis[k]), std::end(pis[k]));
                minLoad = std::min(minLoad, *std::min_element(std::begin(rs[k]), std::end(rs[k])));
            }

            const auto err = (maxLoad - minLoad) * 1.0 / minLoad;

            #ifdef DEBUG
                for (auto r: rs) {
                    for (auto x: r)
                        std::cerr << x << ' ';
                    std::cerr << std::endl;
                }
                std::cerr << m << ' ' << maxLoad << ' ' << err << ' ' << eps << '\n';
            #endif

            std::size_t PQ = 0;
            for (const auto &pi: pis)
                PQ += pi.size() + 2;
            if (err < eps)
                break;
            else if (m - p_best_m > C * PQ) {
                if (best_pis.second * (1 + eps) >= p_best_pis.second)
                    break;
                p_best_pis = best_pis;
                p_best_m = m;
            }
        }
        return best_pis.first;
    }

    /**
    * @brief An easy to use wrapper for the the SubGradient Optimization algorithm.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load. The first parameter, the string s describes
    * the optimization objective. Each term in the optimization objective
    * is made up of a digit indicating the index of the matrix, the name of
    * the first dimension of the matrix and the name of the second dimension
    * of the matrix separated by "+" signs inbetween. As examples, sgo_2dr -> "0ij",
    * sgo_2ds -> "0ij,i=j", sgo_3ds -> "0ij+0jk+0ik" and sgo_3dr ->
    * "0ij+1jk". As one can guess, the part before the comma represents
    * the optimization objective and after the comma includes constraints
    * that can't be derived automatically from the optimization objective.
    * As an example, for sgo_2ds, we had to explicitly put the constraint
    * that the two cut vectors in the two dimensions need to be the same via
    * stating "i=j". However, sgo_3ds can do that automatically because the same
    * matrix dimension is named different letters "i" and "j" etc. that that causes
    * all the dimensions to collapse into one removing the need to put the constraints
    * explicitly.
    * 
    * @param s a string describing the optimization objective
    * @param A a vector of references to matrices used in the optimization objective
    * @param P number of parts in each dimension of the resulting cut vectors
    * @param seed
    * @return a vector of cut vectors
    */
    template <typename Int, typename Value>
    auto partition(const std::string s, const std::vector<std::reference_wrapper<const Matrix<Int, Value>>> As, const std::vector<Int> Ps, int seed = 1) {
        std::vector<std::tuple<Int, char, char>> terms;
        std::set<char> cs;
        std::map<char, Int> id;
        std::vector<std::vector<char>> rid;
        std::map<Int, std::set<std::pair<Int, Int>>> rjd;
        Int N = 0;
        {
            std::map<std::pair<Int, Int>, std::set<char>> m;
            std::map<char, std::set<std::pair<Int, Int>>> n;
#if defined(ENABLE_CPP_PARALLEL)
            for (auto [i, s_end, constr] = std::make_tuple(s.c_str(), s.c_str() + s.size(), false); i < s_end; i++) {
#else
            auto i = s.c_str();
            auto s_end = s.c_str() + s.size();
            auto constr = false;
            for (; i < s_end; i++) {
#endif
                const auto j = std::find(i, s_end, ',');
                i = std::find_if(i, s_end, static_cast<int(*)(int)>(std::isdigit));
                if (j < i || constr) {
                    constr = true;
                    i = j;
                    i = std::find_if(i + 1, s_end, static_cast<int(*)(int)>(std::isalpha));
                    assert(i < s_end);
                    const auto c1 = *i;
                    i = std::find(i + 1, s_end, '=');
                    assert(i < s_end);
                    i = std::find_if(i + 1, s_end, static_cast<int(*)(int)>(std::isalpha));
                    assert(i < s_end);
                    const auto c2 = *i;
                    assert(cs.count(c1) > 0 && cs.count(c2) > 0);
                    const auto j = *n[c1].begin();
                    n[c2].insert(j);
                    m[j].insert(c2);
                }
                else {
                    assert(i < s_end);
                    const Int j = *i - '0';
                    i = std::find_if(i + 1, s_end, static_cast<int(*)(int)>(std::isalpha));
                    assert(i < s_end);
                    const auto c1 = *i;
                    i = std::find_if(i + 1, s_end, static_cast<int(*)(int)>(std::isalpha));
                    assert(i < s_end);
                    const auto c2 = *i;
                    assert(j < As.size());
                    terms.emplace_back(j, c1, c2);
                    m[{j, 0}].insert(c1);
                    m[{j, 1}].insert(c2);
                    cs.insert(c1);
                    cs.insert(c2);
                    n[c1].emplace(j, 0);
                    n[c2].emplace(j, 1);
                }
            }
            for (const auto c: cs) {
                if (id.count(c))
                    continue;
                std::vector<char> st;
                st.push_back(c);
                std::vector<char> rid_N;
                while (!st.empty()) {
                    auto c = st.back();
                    st.pop_back();
                    if (id.count(c))
                        continue;
                    id[c] = N;
                    if (std::find(rid_N.begin(), rid_N.end(), c) == rid_N.end())
                        rid_N.push_back(c);
                    for (const auto j: n[c]) {
                        rjd[N].insert(j);
                        for (const auto tc: m[j])
                            st.push_back(tc);
                    }
                }
                if (!rid_N.empty()) {
                    rid.emplace_back(std::move(rid_N));
                    N++;
                }
            }
            assert(N == Ps.size());
        }
        auto get_dim = [&] (auto j) {
            return j.second == 0 ? As[j.first].get().N() : As[j.first].get().M;
        };
        std::vector<Int> sizes;
        for (Int i = 0; i < rid.size(); i++) {
            std::set<Int> szs;
            for (auto j: rjd[i])
                szs.insert(get_dim(j));
            assert(szs.size() == 1);
            sizes.push_back(*szs.begin());
        }
        std::vector<std::vector<double>> pis;
        std::mt19937_64 gen(seed);
        for (Int i = 0; i < N; i++) {
            Value total = 0;
            for (auto j: rjd[i])
                total += As[j.first].get().get_sps().total_load();
            pis.emplace_back(random_init(Ps[i], total, gen));
        }
        auto m_to_nnz = [&](const auto i, const auto m) {
            Value nnz = 0;
            for (auto j: rjd[i])
                nnz += As[j.first].get().get_sps().query(j.second == 0 ? m : sizes[i], j.second == 0 ? sizes[i] : m);
            return nnz;
        };
        auto pi_to_p = [&](const auto i, const double pi) {
            Int l = 0, r = sizes[i];
            while (l < r) {
                const auto m = l + (r - l) / 2;
                if (pi <= m_to_nnz(i, m))
                    r = m;
                else
                    l = m + 1;
            }
            return l;
        };
        auto to_ps = [&](const auto pis) {
            assert(N == pis.size());
            std::vector<std::vector<Int>> ps;
            std::vector<std::tuple<Int, double, std::reference_wrapper<Int>>> temp;
            for (Int i = 0; i < pis.size(); i++) {
                assert(Ps[i] == pis[i].size() + 1);
                std::vector<Int> p(pis[i].size() + 2);
                p.back() = sizes[i];
                ps.push_back(p);
                for (Int j = 0; j < pis[i].size(); j++)
                    temp.emplace_back(i, pis[i][j], std::ref(ps[i][j + 1]));
            }
            std::for_each( std::begin(temp), std::end(temp), [&](const auto item) {
#if defined(ENABLE_CPP_PARALLEL)
                auto [i, pi, p] = item;
#else
                auto i = std::get<0>(item);
                auto pi = std::get<1>(item);
                auto p = std::get<2>(item);
#endif
                p.get() = pi_to_p(i, pi);
            });
            return ps;
        };
        std::vector<std::tuple<Int, Int, Int>> terms_converted;
        std::transform(terms.begin(), terms.end(), std::back_inserter(terms_converted), [&](auto term) {
#if defined(ENABLE_CPP_PARALLEL)
            const auto [A_idx, c1, c2] = term;
#else
            const auto A_idx = std::get<0>(term);
            const auto c1 = std::get<1>(term);
            const auto c2 = std::get<2>(term);
#endif
            return std::make_tuple(A_idx, std::distance(cs.begin(), cs.find(c1)), std::distance(cs.begin(), cs.find(c2)));
        });
        std::vector<std::vector<Int>> rid_converted;
        std::transform(rid.begin(), rid.end(), std::back_inserter(rid_converted), [&](auto rid_i) {
            std::vector<Int> rid_i_converted;
            std::transform(rid_i.begin(), rid_i.end(), std::back_inserter(rid_i_converted), [&](auto c) {
                return std::distance(cs.begin(), cs.find(c));
            });
            return rid_i_converted;
        });
        std::vector<Int> Ps_cs;
        std::transform(cs.begin(), cs.end(), std::back_inserter(Ps_cs), [&](auto c) {
            return Ps[id[c]];
        });
        auto to_rs = [&](const auto pis) {
            const auto ps = to_ps(pis);
            std::map<std::tuple<Int, char, char>, std::vector<std::vector<Value>>> Qs;
#if defined(ENABLE_CPP_PARALLEL)
            for (const auto [A_idx, c1, c2]: terms) {
#else
            for (const auto term: terms) {
                const auto A_idx = std::get<0>(term);
                const auto c1 = std::get<1>(term);
                const auto c2 = std::get<2>(term);
#endif
                if (Qs.count({A_idx, id[c1], id[c2]}))
                    continue;
                Qs[{A_idx, id[c1], id[c2]}] = As[A_idx].get().get_sps().compute_loads(ps[id[c1]], ps[id[c2]]);
            }
            std::vector<std::reference_wrapper<std::vector<std::vector<Value>>>> Qv;
            std::transform(terms.begin(), terms.end(), std::back_inserter(Qv), [&](auto term) {
#if defined(ENABLE_CPP_PARALLEL)
                const auto [A_idx, c1, c2] = term;
#else
                const auto A_idx = std::get<0>(term);
                const auto c1 = std::get<1>(term);
                const auto c2 = std::get<2>(term);
#endif
                return std::ref(Qs[{A_idx, id[c1], id[c2]}]);
            });
            std::vector<std::vector<Value>> rs;
            for (const auto p: ps)
                rs.emplace_back(p.size() - 1);
            std::vector<Int> v(cs.size());
            for (Int i = 0; i < Ps_cs.size();) {
                for (i = 0; i < Ps_cs.size(); i++) {
                    v[i]++;
                    if (v[i] >= Ps_cs[i])
                        v[i] = 0;
                    else
                        break;
                }
                Int k = 0;
                Value sum = 0;
#if defined(ENABLE_CPP_PARALLEL)
                for (const auto [A_idx, i1, i2]: terms_converted){
#else
                for (const auto term: terms_converted){
                    // const auto A_idx = std::get<0>(term);
                    const auto i1 = std::get<1>(term);
                    const auto i2 = std::get<2>(term);
#endif
                    sum += Qv[k++].get()[v[i1]][v[i2]];
                }
                for (Int i = 0; i < rs.size(); i++)
                    for (auto j: rid_converted[i])
                        rs[i][v[j]] = std::max(rs[i][v[j]], sum);
            }
            return rs;
        };
        return to_ps(partition(pis, to_rs));
    }

    template <typename Int, typename Value>
    auto partition_spmv(const Matrix<Int, Value> &A, const Int P, const double alpha, const double beta, const int seed = 1) {
        std::vector<std::vector<double>> pis;
        std::mt19937_64 gen(seed);
        const auto total = A.get_sps().total_load() + alpha * A.N() + beta * A.M;
        pis.emplace_back(random_init(P, total, gen));
        auto m_to_nnz = [&](const auto m) {
            auto nnz = A.get_sps().query(m, A.M) + alpha * m;
            nnz += A.get_sps().query(A.N(), m) + beta * m;
            return nnz;
        };
        auto pi_to_p = [&](const double pi) {
            Int l = 0, r = A.N();
            while (l < r) {
                const auto m = l + (r - l) / 2;
                if (pi <= m_to_nnz(m))
                    r = m;
                else
                    l = m + 1;
            }
            return l;
        };
        auto to_ps = [&](const auto pis) {
            std::vector<std::vector<Int>> ps;
            std::vector<std::tuple<Int, double, std::reference_wrapper<Int>>> temp;
            for (Int i = 0; i < pis.size(); i++) {
                std::vector<Int> p(pis[i].size() + 2);
                p.back() = A.N();
                ps.push_back(p);
                for (Int j = 0; j < pis[i].size(); j++)
                    temp.emplace_back(i, pis[i][j], std::ref(ps[i][j + 1]));
            }
            #ifdef ENABLE_CPP_PARALLEL
            std::for_each(exec_policy, std::begin(temp), std::end(temp), [&](const auto item) {
            #else
            std::for_each(std::begin(temp), std::end(temp), [&](const auto item) {
            #endif
            #ifdef ENABLE_CPP_PARALLEL
                auto [i, pi, p] = item;
            #else
                auto pi = std::get<1>(item);
                auto p = std::get<2>(item);
            #endif
                p.get() = pi_to_p(pi);
            });
            return ps;
        };
        auto to_rs = [&](const auto pis) {
            const auto ps = to_ps(pis);
            auto loads = A.get_sps().compute_loads(ps[0], ps[0]);
            for (Int i = 0; i < P; i++)
                for (Int j = 0; j < P; j++)
                    loads[i][j] += alpha * (ps[0][i + 1] - ps[0][i]) + beta * (ps[0][j + 1] - ps[0][j]);
            std::vector<std::vector<Value>> rs;
            for (const auto p: ps)
                rs.emplace_back(p.size() - 1);
            for (Int i = 0; i < P; i++)
                for (Int j = 0; j < P; j++) {
                    rs[0][i] = std::max(rs[0][i], loads[i][j]);
                    rs[0][j] = std::max(rs[0][j], loads[i][j]);
                }
            return rs;
        };
        return to_ps(partition(pis, to_rs))[0];
    }

    /**
    * @brief Implements the SubGradient Optimization algorithm,
    * where each task is made of 3 tiles and tries to minimize task size.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load.
    *
    * @param A Matrix
    * @param P number of parts
    * @param seed
    * @return a cut vector
    */
    template <typename Int, typename Value>
    auto partition_tri(const Matrix<Int, Value> &A, const Int P, const int seed = 1) {
        return partition("0ij+0jk+0ik", std::vector<std::reference_wrapper<const Matrix<Int, Value>>>{A}, std::vector<Int>{P}, seed)[0];
    }
    
    /**
    * @brief Implements the SubGradient Optimization algorithm,
    * where each task is made up of 2 tiles from each matrix
    * in SpGEMM matrix multiplication and tries to minimize task size.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load.
    *
    * @param A Matrix
    * @param B Matrix
    * @param P number of parts in 1st dimension
    * @param Q number of parts in 2nd dimension
    * @param R number of parts in 3rd dimension
    * @param seed
    * @return a cut vector
    */
    template <typename Int, typename Value>
    auto partition(const Matrix<Int, Value> &A, const Matrix<Int, Value> &B, const Int P, const Int Q, const Int R, int seed = 1) {
        const auto ps = partition("0ij+1jk", std::vector<std::reference_wrapper<const Matrix<Int, Value>>>{A, B}, std::vector<Int>{P, Q, R}, seed);
        return std::make_tuple(ps[0], ps[1], ps[2]);
    }

    /**
    * @brief Implements the SubGradient Optimization algorithm.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load.
    *
    * @param A Matrix
    * @param P number of parts
    * @param seed
    * @return a cut vector
    */
    template <typename Int, typename Value>
    auto partition(const Matrix<Int, Value> &A, const Int P, const int seed = 1) {
        return partition("0ij,i=j", std::vector<std::reference_wrapper<const Matrix<Int, Value>>>{A}, std::vector<Int>{P}, seed)[0];
    }
    
    /**
    * @brief Implements the SubGradient Optimization algorithm.
    * This algorithm initializes the cut vectors randomly and
    * tries to refine them at each iteration by moving all the
    * cuts in the negative direction of the gradient to minimize
    * the maximum load.
    *
    * @param A Matrix
    * @param P number of parts in row dimension
    * @param Q number of parts in column dimension
    * @param seed
    * @return a cut vector
    */
    template <typename Int, typename Value>
    auto partition(const Matrix<Int, Value> &A, const Int P, const Int Q, const int seed = 1) {
        const auto ps = partition("0ij", std::vector<std::reference_wrapper<const Matrix<Int, Value>>>{A}, std::vector<Int>{P, Q}, seed);
        return std::make_pair(ps[0], ps[1]);
    }
}
#if !defined(ENABLE_CPP_PARALLEL)
} // nested namespace
#endif