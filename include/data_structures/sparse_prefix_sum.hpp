#pragma once

#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <thread>
#include <iostream>
#include <random>
#include <cmath>
#include <mutex>
#include <tuple>
#include <utility>
#include <memory>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <atomic>

#include "csr_matrix.hpp"

namespace sarma {

#if defined(ENABLE_CPP_PARALLEL)
    template <typename Int, typename Value, typename Real = Int, auto lowest = std::numeric_limits<Real>::lowest()>
#else
    template <typename Int, typename Value, typename Real = Int, Real lowest = std::numeric_limits<Real>::lowest()>
#endif
    struct coordinate_compressor {

        coordinate_compressor() = default;

        coordinate_compressor(const std::vector<std::pair<Real, Real>> &pts) {
            X.reserve(pts.size() + 1);
            Y.reserve(pts.size() + 1);

#if defined(ENABLE_CPP_PARALLEL)
            for (auto [x, y]: pts) {
                X.push_back(x);
                Y.push_back(y);
            }
#else
            for (auto i: pts) {
                X.push_back(i.first);
                Y.push_back(i.second);
            }
#endif
            compress();
        }

        coordinate_compressor(const Matrix<Int, Value> &A) {
            X.reserve(A.indptr.back() + 1);
            Y.reserve(A.indptr.back() + 1);

            for (Int i = 0; i < A.indptr.size() - 1; i++)
                for (auto j = A.indptr[i]; j < A.indptr[i + 1]; j++) {
                    X.push_back(i);
                    Y.push_back(A.indices[j]);
                }

            compress();
        }

        std::pair<Int, Int> query(const std::pair<Real, Real> p) const {
            return {utils::lowerbound_index<Real>(X, p.first), utils::lowerbound_index<Real>(Y, p.second)};
        }

        auto convert(const std::vector<std::pair<Real, Real>> &pts, const std::vector<Value> &data) {
            std::vector<std::pair<Int, Int>> i_pts(pts.size());

#if defined(ENABLE_CPP_PARALLEL)
            std::transform(exec_policy, std::begin(pts), std::end(pts), std::begin(i_pts), [](const auto &p) {
                return query(p);
            });
#else
            std::transform(std::begin(pts), std::end(pts), std::begin(i_pts), [&](const auto &p) {
                return query(p);
            });
#endif
            return Matrix<Int, Value>(i_pts, data);
        }

        auto convert(const Matrix<Int, Value> &A) {
            std::vector<std::pair<Int, Int>> i_pts(A.indptr.back());

#if defined(ENABLE_CPP_PARALLEL)
            std::transform(exec_policy, std::begin(A.indices), std::end(A.indices), std::begin(i_pts), [&](const auto &j) {
                return query({utils::lowerbound_index(A.indptr, (Int) std::distance(A.indices.data(), &j)), j});
            });
#else
            std::transform( std::begin(A.indices), std::end(A.indices), std::begin(i_pts), [&](const auto &j) {
                return query({utils::lowerbound_index(A.indptr, (Int) std::distance(A.indices.data(), &j)), j});
            });
#endif
            return Matrix<Int, Value>(i_pts, A.get_data(), A.is_using_data());
        }

    private:

        std::vector<Real> X = {lowest}, Y = {lowest};

        void compress() {
#if defined(ENABLE_CPP_PARALLEL)
            std::sort(exec_policy, std::begin(X), std::end(X));
            X.resize(std::distance(std::begin(X), std::unique(exec_policy, std::begin(X), std::end(X))));
#else
            std::sort( std::begin(X), std::end(X));
            X.resize(std::distance(std::begin(X), std::unique( std::begin(X), std::end(X))));
#endif
            X.shrink_to_fit();

#if defined(ENABLE_CPP_PARALLEL)
            std::sort(exec_policy, std::begin(Y), std::end(Y));
            Y.resize(std::distance(std::begin(Y), std::unique(exec_policy, std::begin(Y), std::end(Y))));
#else
            std::sort( std::begin(Y), std::end(Y));
            Y.resize(std::distance(std::begin(Y), std::unique( std::begin(Y), std::end(Y))));
#endif
            Y.shrink_to_fit();
        }
    };

    template <typename Int, typename Value>
    class persistent_BIT {

        static Int down(Int i) {
            return i & (i - 1);
        }

        static Int up(Int i) {
            return (i | (i - 1)) + 1;
        }

        std::shared_ptr<std::size_t[]> xadj;
        std::size_t xadj_size;
        std::shared_ptr<Int[]> v_arr;
        std::shared_ptr<Value[]> arr;
        Int v;

        void updateLastHelper(Int *f, Int i, const Int v, const Value inc) {
            if (v_arr[xadj[i] + f[i] - 1] == v)
                arr[xadj[i] + f[i] - 1] += inc;
            else {
                v_arr[xadj[i] + f[i]] = v;
                arr[xadj[i] + f[i]] = arr[xadj[i] + f[i] - 1] + inc;
                f[i]++;
                assert(xadj[i] + f[i] <= xadj[i + 1]);
            }
        }

        void updateLast(Int *f, Int i, const Int v, const Value inc = 1) {
            for (; i < size(); i = up(i))
                updateLastHelper(f, i, v, inc);
        }

        auto test(const Int offset, const Int * const xadj, const Int * const xadj_end, const Matrix<Int, Value> &A) const {
            const std::size_t N = std::distance(xadj, xadj_end - 1);
            for (Int j = 0; j < N; j++)
                for (auto k = xadj[j]; k < xadj[j + 1]; k++)
                    if (std::abs(1.0 * query(A.indices[k], A.indices[k] + 1, j + offset) - query(A.indices[k], A.indices[k] + 1, j + offset - 1) - A.data(k)) > 0.001)
                        return false;
            return true;
        }

    public:

        auto size() const {
            return xadj_size - 1;
        }

        auto nnz() const {
            return xadj[size()];
        }

        persistent_BIT() = default;

        persistent_BIT(const Int offset, const Int M, const Int * const xadj, const Int * const xadj_end, const Matrix<Int, Value> &A) : xadj_size(M + 2), v(offset) {
            this->xadj = std::shared_ptr<std::size_t[]>(new std::size_t[xadj_size]);
            const std::size_t N = std::distance(xadj, xadj_end - 1);
            std::fill_n(this->xadj.get(), xadj_size, 1);
            std::vector<Int> f(size(), offset);
            for (Int j = 0; j < N; j++)
                for (auto k = xadj[j]; k < xadj[j + 1]; k++) {
                    for (auto i = A.indices[k] + 1; i < size(); i = up(i))
                        if (f[i] < j + offset) {
                            this->xadj[i + 1]++;
                            f[i] = j + offset;
                        }
                }
            this->xadj[0] = 0;
            std::partial_sum(this->xadj.get(), this->xadj.get() + xadj_size, this->xadj.get());
            v_arr = std::shared_ptr<Int[]>(new Int[nnz()]);
            arr = std::shared_ptr<Value[]>(new Value[nnz()]);
            for (std::size_t i = 0; i < size(); i++) {
                v_arr[this->xadj[i]] = offset;
                arr[this->xadj[i]] = 0;
            }
            std::fill(f.begin(), f.end(), 1);
            for (Int j = 0; j < N; j++)
                for (auto k = xadj[j]; k < xadj[j + 1]; k++)
                    updateLast(f.data(), A.indices[k] + 1, j + offset, A.data(k));
            assert(test(offset, xadj, xadj_end, A));
        }

        auto query(Int i, Int v) const {
            Value sum = 0;
            if (v < this->v)
                return sum;
            for (; i > 0; i = down(i))
                sum += arr[xadj[i] + utils::lowerbound_index(v_arr.get() + xadj[i], v_arr.get() + xadj[i + 1], v)];
            return sum;
        }

        auto query(Int i, Int j, Int v) const {
            Value sum = 0;
            if (v < this->v)
                return sum;
            while(i != j) {
                const Value inc = i < j ? 1 : -1;
                auto &idx = i < j ? j : i;
                sum += arr[xadj[idx] + utils::lowerbound_index(v_arr.get() + xadj[idx], v_arr.get() + xadj[idx + 1], v)] * inc;
                idx = down(idx);
            }
            return sum;
        }

        auto getLast() const {
            std::vector<Value> a(size());
            for (Int i = 0; i < size(); i++)
                a[i] = arr[xadj[i + 1] - 1];
            return a;
        }

        void incAll(const std::vector<Value> &a) {
            for (Int i = 0; i < size(); i++)
                for (auto j = xadj[i]; j < xadj[i + 1]; j++)
                    arr[j] += a[i];
        }
    };

    template <typename Int, typename Value>
    class sparse_prefix_sum : public matrix_base<Int, Value> {

        Int N, M;
        std::vector<Int> v_arr;
        std::vector<persistent_BIT<Int, Value>> BITs;
        coordinate_compressor<Int, Value, Int> compressor;
    
        auto compressor_query(std::pair<Int, Int> p) const {
            const Int i = p.first ? 1 : 0;
            const Int j = p.second ? 1 : 0;
            p = compressor.query({p.first - i, p.second - j});
            return std::make_pair(p.first + i, p.second + j);
        }

        auto compute_prefix_loads(const std::vector<Int> &p, const std::vector<Int> &q) const {
            std::vector<std::vector<Value>> loads(p.size(), std::vector<Value>(q.size()));
            std::vector<std::tuple<Int, Int, Value *>> temp;
            for (Int i = 0; i < p.size(); i++)
                for (Int j = 0; j < q.size(); j++)
                    temp.emplace_back(p[i], q[j], &loads[i][j]);

#if defined(ENABLE_CPP_PARALLEL)
            std::for_each(exec_policy, std::begin(temp), std::end(temp), [this](const std::tuple<Int, Int, Value *> item) {
                auto [i, j, L_ptr] = item;
#else
            std::for_each( std::begin(temp), std::end(temp), [this](const std::tuple<Int, Int, Value *> item) {
                auto i = std::get<0>(item);
                auto j = std::get<1>(item);
                auto L_ptr = std::get<2>(item);
#endif
                
                *L_ptr = query(i, j);
            });
            return loads;
        }

    public:

        auto getN() const {
            return N;
        }

        auto getM() const {
            return M;
        }

        /**
         * @brief return the maximum of row or column of the underlying matrix
         **/
        auto size() const {
            return std::max(N, M);
        }

        sparse_prefix_sum() = default;

        /**
         * @brief Constructor to construct sparse prefix sum data structure for a given matrix
         * @param A_ the input matrix
         **/
        sparse_prefix_sum(const Matrix<Int, Value> &A_) : N(A_.N()), M(A_.M), compressor(A_) {

            const auto A = compressor.convert(A_);
            const auto num_structures = std::min((Int)std::thread::hardware_concurrency(), (Int)A.N());
            
            v_arr = nicol1d::partition_prefix(A.indptr, num_structures);
            std::transform(std::begin(v_arr), std::end(v_arr), std::begin(v_arr), [](auto &i) {
                return i + 1;
            });

            std::vector<Int> iota_v(num_structures);
            std::iota(std::begin(iota_v), std::end(iota_v), 0);

            BITs.resize(num_structures);
#if defined(ENABLE_CPP_PARALLEL)
            std::transform(exec_policy, std::begin(iota_v), std::end(iota_v), std::begin(BITs), [&](const auto i) {
                return persistent_BIT<Int, Value>(v_arr[i], A.M, A.indptr.data() + v_arr[i] - 1, A.indptr.data() + v_arr[i + 1], A);
            });
#else
            std::transform( std::begin(iota_v), std::end(iota_v), std::begin(BITs), [&](const auto i) {
                return persistent_BIT<Int, Value>(v_arr[i], A.M, A.indptr.data() + v_arr[i] - 1, A.indptr.data() + v_arr[i + 1], A);
            });
#endif
            #if defined(LOGGING) && defined(ENABLE_CPP_PARALLEL)
                const auto bytes = std::transform_reduce(exec_policy std::begin(BITs), std::end(BITs), num_structures * (A.M + 2) * sizeof(std::size_t), std::plus<>(), [](const auto &BIT_i) {
                    return BIT_i.nnz() * (sizeof(Int) + sizeof(Value));
                });
                std::cerr << "BITs have been constructed & " << bytes << " bytes have been used" << std::endl;
            #endif

            std::vector<std::vector<Value>> lasts(num_structures - 1);
#if defined(ENABLE_CPP_PARALLEL)
            std::transform(exec_policy, std::begin(BITs), std::end(BITs) - 1, std::begin(lasts), [](const auto &BIT_i) {
                return BIT_i.getLast();
            });
#else
            std::transform( std::begin(BITs), std::end(BITs) - 1, std::begin(lasts), [](const auto &BIT_i) {
                return BIT_i.getLast();
            });
#endif

#if defined(ENABLE_CPP_PARALLEL)
            std::for_each(exec_policy, std::begin(iota_v), std::end(iota_v), [&](const auto i) {
#else
            std::for_each( std::begin(iota_v), std::end(iota_v), [&](const auto i) {
#endif
                const Int begin = i * 1ll * A.M / num_structures + 1;
                const Int end = (i + 1) * 1ll * A.M / num_structures + 1;
                for (Int l = 1; l < lasts.size(); l++)
                    for (Int u = begin; u < end; u++)
                        lasts[l][u] += lasts[l - 1][u];
            });


#if defined(ENABLE_CPP_PARALLEL)
            std::for_each(exec_policy, std::begin(iota_v) + 1, std::end(iota_v), [&](const auto i) {
#else
            std::for_each( std::begin(iota_v) + 1, std::end(iota_v), [&](const auto i) {
#endif
                BITs[i].incAll(lasts[i - 1]);
            });
        }


        /**
         * @brief Queries the rectangle whose bottom right corner is given
         * @param i the row of the bottom right corner
         * @param j the column of the bottom right corner
         **/
        Value query(Int i, Int j) const {
            std::tie(i, j) = compressor_query({i, j});
            const auto idx = utils::lowerbound_index<Int>(v_arr, i);
            return idx < BITs.size() ? BITs[idx].query(j, i) : (Value)0;
        }

        /**
         * @brief Queries the rectangle whose upper left and bottom right corners are given
         * @param upper_left holds the row and column in this order
         * @param lower_right holds the row and column in this order
         **/
        Value query(std::pair<Int, Int> upper_left, std::pair<Int, Int> lower_right) const {
            upper_left = compressor_query(upper_left);
            lower_right = compressor_query(lower_right);
            auto idx = utils::lowerbound_index<Int>(v_arr, lower_right.first);
            auto res = idx < BITs.size() ? BITs[idx].query(upper_left.second - 1, lower_right.second, lower_right.first) : (Value)0;
            idx = utils::lowerbound_index<Int>(v_arr, upper_left.first - 1);
            res -= idx < BITs.size() ? BITs[idx].query(upper_left.second - 1, lower_right.second, upper_left.first - 1) : (Value)0;
            return res;
        }

        /**
         * @brief Returns the sum of all nonzeros in the underlying matrix
         **/
        Value total_load() const {
            return query(size(), size());
        }

        /**
         * @brief Given two cut vectors, returns the tile loads implied by the cut vectors
         * @param p the cut vector along rows
         * @param q the cut vector along columns
         **/
        std::vector<std::vector<Value>> compute_loads(const std::vector<Int> &p, const std::vector<Int> &q) const {
            const auto p_loads = compute_prefix_loads(p, q);
            std::vector<std::vector<Value>> loads(p.size() - 1, std::vector<Value>(q.size() - 1));
            for (Int i = 0; i < loads.size(); i++)
                for (Int j = 0; j < loads[i].size(); j++)
                    loads[i][j] = p_loads[i + 1][j + 1] - p_loads[i][j + 1] + p_loads[i][j] - p_loads[i + 1][j];
            return loads;
        }

        /**
         * @brief Given two cut vectors, returns the maximum load implied by the cut vectors
         * @param p the cut vector along rows
         * @param q the cut vector along columns
         **/
        Value compute_maxload(const std::vector<Int> &p, const std::vector<Int> &q) const {
            const auto loads = compute_loads(p, q);
            return utils::max(loads);
        }

        /**
         * @brief Given two cut vectors, returns the maximum load of either idx'th
         * row or column depending on the col parameter implied by the cut vectors
         * @param p the cut vector along rows
         * @param q the cut vector along columns
         * @param idx the row or column index in the tile load matrix
         * @param col whether idx is for rows or columns
         **/
        auto compute_maxload(const std::vector<Int> &p, const std::vector<Int> &q, const Int idx, bool col = false) const {
            if (col) {
                const std::vector<Int> tq = {q[idx - 1], q[idx]};
                return compute_maxload(p, tq);
            }
            else {
                const std::vector<Int> tp = {p[idx - 1], p[idx]};
                return compute_maxload(tp, q);
            }
        }
    };
}
