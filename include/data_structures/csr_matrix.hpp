#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <queue>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include <thread>
#include <atomic>

extern "C" {
#include "mmio.h"
}
#include "tools/utils.hpp"
#include "algorithms/nicol1d.hpp"

namespace sarma {

    /** An enum type used to define row ordering of a given
     * matrix. We provide three simple row ordering strategies.
     */
    enum Order {
        ASC, /**< Order rows in an ascending way st. |A(i,*)| <= |A(i+1,*)| */
        DSC, /**< Order rows in an descending way st. |A(i,*)| >= |A(i+1,*)| */
        RCM, /**< Order rows usin reverse Cuthill-McKee algorithm */
        NAT /**< Don't apply any ordering. Default, natural order */
    };

    template <class Ordinal, class Value>
    class sparse_prefix_sum;

    template <class Ordinal, class Value>
    struct matrix_base {
        
        virtual std::vector<std::vector<Value>> compute_loads(const std::vector<Ordinal> &p, const std::vector<Ordinal> &q) const = 0;
        
        virtual Value compute_maxload(const std::vector<Ordinal> &p, const std::vector<Ordinal> &q) const = 0;

        virtual Value total_load() const = 0;
    };

    template <class Ordinal, class Value>
    class Matrix : public matrix_base<Ordinal, Value> {

        std::vector<Ordinal> indptr_;
        std::vector<Ordinal> indices_;
        std::vector<Value> data_;
        Ordinal M_;
        bool sorted_ = false;
        std::shared_ptr<std::unique_ptr<sparse_prefix_sum<Ordinal, Value>>> sps_
            = std::make_shared<std::unique_ptr<sparse_prefix_sum<Ordinal, Value>>>(nullptr);

        auto use_data_off(std::size_t /*i*/) const {
            return (Value)1;
        }
        
        auto use_data_on(std::size_t i) const {
            return data_[i];
        }

        Value (Matrix::*use_data_)(std::size_t) const = &Matrix::use_data_off;

    public:

        const std::vector<Ordinal> &indptr = indptr_;
        const std::vector<Ordinal> &indices = indices_;
        const Ordinal &M = M_;
        const bool &sorted = sorted_;

        /**
         * @brief returns data[i] if using data, returns 1 otherwise.
         **/
        inline auto data(std::size_t i) const {
            return (this->*use_data_)(i);
        }
        
        const auto & get_data() const {
            return data_;
        }

        Value total_load() const {
            return is_using_data() ? std::accumulate(data_.begin(), data_.end(), (Value)0) : NNZ();
        }

        const auto & get_sps() const;

        /**
         * @brief if data exists, use=true changes matrix mode to use_data
         * @param use specifies whether data should be used
         **/
        void use_data(bool use) {
            use_data_ = use && data_.size() == NNZ() ? &Matrix::use_data_on : &Matrix::use_data_off;
        }

        bool is_using_data() const {
            return use_data_ == &Matrix::use_data_on;
        }

        /**
         * @brief returns whether the matrix has data stored.
         **/
        bool is_pattern() const {
            return data_.size() != NNZ();
        }

        /**
        * @brief please note that the vector indptr is csr representation
        * and size of the vector is N_ + 1, here we add a conventional method for
        * getting the number of indptr. However the size of the vector is actually
        * N + 1, so please keep that in mind.
        **/
        Ordinal N() const { return indptr.size() - 1; }

        Ordinal NNZ() const { return indptr.back(); }

        Matrix() = default;

        /** @brief IO constructor
        * Construct the graph from a file that should be
        * formatted as our csr format
        **/
        Matrix(const std::filesystem::path &path, const bool use_data = false) {
            if (path.filename() == "-") {
                read_binary(std::cin, use_data);
            }
            else if (!std::filesystem::exists(path)) {
                std::cerr << "Failed to open graph file: " << path << std::endl;
                std::exit(1); // UVC: exit if cannot open file; 
                // throw std::runtime_error("Failed to open graph file " + filename);
            } else if (path.extension() == ".bin") {
                std::ifstream in(path, std::ios::binary);
                read_binary(in, use_data);
            }
            else if (path.extension() == ".mtx") {
                auto f = std::fopen(path.c_str(), "r");
                read_matrix_market(f, use_data);
            }
            else {
                std::cerr << "Unrecognized file format: " << path.extension() << std::endl;
                std::exit(1);
            }
            // std::cerr << "N: " << N() << ", NNZ: " << NNZ() << std::endl;
        }

        Matrix(const std::vector<std::pair<Ordinal, Ordinal>> &pts, const std::vector<Value> &data) : sorted_(true) {
            std::vector<Ordinal> idx(pts.size());
            std::for_each(exec_policy, idx.begin(), idx.end(), [&](auto &idx_i) {
                idx_i = std::distance(&idx[0], &idx_i);
            });
            std::sort(exec_policy, std::begin(idx), std::end(idx), [&](const auto i, const auto j) {
                return pts[i] < pts[j];
            });
            indptr_.resize(pts[idx.back()].first + 2);
            std::for_each(exec_policy, std::begin(indptr_), std::end(indptr_) - 1, [&](auto &indptr_i) {
                const auto i = std::distance(&indptr_[0], &indptr_i);
                indptr_i = std::lower_bound(std::begin(idx), std::end(idx), i, [&](const auto &idx_i, const auto &i) {
                    return pts[idx_i].first < i;
                }) - std::begin(idx);
            });
            indptr_.back() = pts.size();
            indices_.resize(NNZ());
            if (data.size() == NNZ())
                data_.resize(NNZ());
            std::for_each(exec_policy, std::begin(indptr), std::end(indptr) - 1, [&](const auto &indptr_i) {
                const auto i = std::distance(&indptr[0], &indptr_i);
                for (auto j = indptr_i; j < indptr[i + 1]; j++) {
                    indices_[j] = pts[idx[j]].second;
                    if (data.size() == pts.size())
                        data_[j] = data[idx[j]];
                }
            });
            M_ = 1 + *std::max_element(exec_policy, std::begin(indices), std::end(indices));
        }

        /**
        * @brief Constructor to create a graph given all the class members
        * @note uses move semantics.
        **/
        Matrix(std::vector<Ordinal> &&indptr, std::vector<Ordinal> &&indices,
            std::vector<Value> &&data, const Ordinal M)
            : indptr_(std::move(indptr)), indices_(std::move(indices)), data_(std::move(data)), M_(M) {}

        /**
        * @brief Constructor to create a graph given all the class members
        * this one is for mostly external usage, uses copy instead of move
        **/
        Matrix(std::vector<Ordinal> &indptr, std::vector<Ordinal> &indices,
            std::vector<Value> &data, const Ordinal M)
            : indptr_(indptr), indices_(indices), data_(data), M_(M) {}

        Matrix(const Matrix &other) : indptr_(other.indptr), indices_(other.indices), data_(other.data_), M_(other.M), sorted_(other.sorted) {}

        Matrix & operator =(const Matrix &other) {
            indptr_ = other.indptr;
            indices_ = other.indices;
            data_ = other.data_;
            M_ = other.M;
            sorted_ = other.sorted;
            return *this;
        }

        Matrix(Matrix &&other) : indptr_(std::move(other.indptr)), indices_(std::move(other.indices)), data_(std::move(other.data_)), M_(other.M), sorted_(other.sorted) {}

        Matrix & operator =(Matrix &&other) {
            indptr_ = std::move(other.indptr);
            indices_ = std::move(other.indices);
            data_ = std::move(other.data_);
            M_ = other.M;
            sorted_ = other.sorted;
            return *this;
        }

        /**
        * @brief Constructor from dense matrix
        * This will be used for testing purposes
        **/
        Matrix(const std::vector<std::vector<Ordinal>> A) {
            indptr_.reserve(A.size() + 1);
            indptr_.push_back(0);
            for (Ordinal i = 0; i < A.size(); i++) {
                Ordinal count = 0;
                for (Ordinal j = 0; j < A[i].size(); j++) {
                    if (A[i][j] != 0) {
                        indices_.push_back(j);
                        data_.push_back(A[i][j]);
                        count++;
                    }
                }
                indptr_.push_back(count);
            }
            std::partial_sum(indptr_.begin(), indptr_.end(), indptr_.begin());
            M_ = A.size();
        }

        void read_binary(std::istream &in, const bool use_data) {
            const auto [v_dtype, e_dtype] = utils::get_matrix_type(in);
            assert(v_dtype == sizeof(Ordinal) && e_dtype == sizeof(Ordinal));
            Ordinal N = v_dtype + e_dtype;
            in.read(reinterpret_cast<char *>(&N), sizeof(Ordinal));
            in.read(reinterpret_cast<char *>(&M_), sizeof(Ordinal));
            indptr_.resize(N + 1);
            in.read(reinterpret_cast<char*>(indptr_.data()), indptr.size() * sizeof(Ordinal));
            indices_.resize(NNZ());
            in.read(reinterpret_cast<char *>(indices_.data()), indices.size() * sizeof(Ordinal));
            if (use_data) {
                data_.resize(NNZ());
                in.read(reinterpret_cast<char *>(data_.data()), data_.size() * sizeof(Value));
                if ((size_t)in.gcount() == data_.size() * sizeof(Value))
                    this->use_data(true);
                else
                    data_.clear();
            }
        }

        void read_matrix_market(std::FILE *f, const bool use_data) {
            MM_typecode matcode;
            if (mm_read_banner(f, &matcode)) {
                std::cerr << "Could not process Matrix Market banner." << std::endl;
                std::exit(1);
            }
            if (mm_is_complex(matcode) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
                std::cerr << "The .mtx file should contain a sparse real matrix." << std::endl;
                std::cerr << "Market Market type: [" << mm_typecode_to_str(matcode) << "] is not yet supported." << std::endl;
                std::exit(1);
            }
            int M, N, nz, retcode;
            if ((retcode = mm_read_mtx_crd_size(f, &M, &N, &nz)))
                std::exit(retcode);
            std::vector<int> I(nz), J(nz);
            std::vector<double> t;
            if (!mm_is_pattern(matcode))
                t.resize(nz);
            if (mm_is_integer(matcode)) // mmio doesn't read integer matrices for some reason
                mm_set_real(&matcode);  // the code logic just doesn't accept integer matrices.
            if ((retcode = mm_read_mtx_crd_data(f, M, N, nz, I.data(), J.data(), t.data(), matcode)))
                std::exit(retcode);
            std::vector<std::pair<Ordinal, Ordinal>> pts;
            pts.reserve(nz);
            for (int i = 0; i < nz; i++)
                pts.emplace_back(I[i] - 1, J[i] - 1);
            if (use_data && !mm_is_pattern(matcode))
                data_.assign(t.begin(), t.end());
            *this = Matrix(pts, data_);
            while (indptr.size() < (std::size_t)M + 1)
                indptr_.push_back(indptr.back());
            M_ = N;
            this->use_data(use_data);
        }

        Value compute_maxload(const std::vector<Ordinal> &p, const std::vector<Ordinal> &q) const {
            const auto loads = compute_loads(p, q);
            return utils::max(loads);
        }

        std::vector<std::vector<Value>> compute_loads(const std::vector<Ordinal> &p, const std::vector<Ordinal> &q) const {
            const auto num_threads = std::thread::hardware_concurrency();
            std::vector<std::vector<std::vector<Value>>> loadss(num_threads, std::vector<std::vector<Value>>(p.size() - 1, std::vector<Value>(q.size() - 1, 0)));
            const auto v_arr = nicol1d::partition_prefix(indptr, num_threads);
            if (is_using_data()) {
                std::for_each(exec_policy, loadss.begin(), loadss.end(), [&](auto &loads) {
                    const auto tid = std::distance(&loadss[0], &loads);
                    const auto begin = v_arr[tid], end = v_arr[tid + 1];
                    for (Ordinal i = begin, u = 0; i < end; i++) {
                        for (; u < p.size() - 1 && i >= p[u]; u++);

                        for (auto j = indptr[i]; j < indptr[i + 1]; j++)
                            loads[u - 1][utils::lowerbound_index(q, indices[j])] += data_[j];
                    }
                });
            } else {
                std::for_each(exec_policy, loadss.begin(), loadss.end(), [&](auto &loads) {
                    const auto tid = std::distance(&loadss[0], &loads);
                    const auto begin = v_arr[tid], end = v_arr[tid + 1];
                    for (Ordinal i = begin, u = 0; i < end; i++) {
                        for (; u < p.size() - 1 && i >= p[u]; u++);

                        for (auto j = indptr[i]; j < indptr[i + 1]; j++)
                            loads[u - 1][utils::lowerbound_index(q, indices[j])] += 1;
                    }
                });
            }
            return std::reduce(exec_policy, loadss.begin(), loadss.end(), std::vector<std::vector<Value>>(p.size() - 1, std::vector<Value>(q.size() - 1, 0)), [](const auto &L1, const auto &L2) {
                std::vector<std::vector<Value>> L(L1.size(), std::vector<Value>(L1[0].size(), 0));
                for (Ordinal i = 0; i < L.size(); i++)
                    for (Ordinal j = 0; j < L[i].size(); j++)
                        L[i][j] += L1[i][j] + L2[i][j];
                return L;
            });
        }

        template <typename Real = double>
        void print(std::ostream &out, const std::vector<Ordinal> &p, const std::vector<Ordinal> &q,
                    bool symmetric, Real sp_time, Real ds_time, Real par_time) const {
            out << "Cuts:\n";
            for (auto i: p)
                out << i << '\t';
            out << '\n';
            if (!symmetric) {
                for (auto i: q)
                    out << i << '\t';
                out << '\n';
            }

            out << "<<<" << '\n';
            auto loads = compute_loads(p, q);
            for (auto row: loads) {
                for (auto tile: row)
                    out << tile << '\t';
                out << '\n';
            }
            out << "<<<" << '\n';

            auto maxLoad = utils::max(loads);
            out << "Max load: " << maxLoad << '\n';
            out << "Sparsification time (s): " << sp_time << '\n';
            out << "Sparse data structure construction time (s): " << ds_time << '\n';
            out << "Partitioning time (s): " << par_time << '\n';

            auto avgNumEdges = indptr.back() * 1.0 / (p.size() - 1) / (q.size() - 1);

            out << "Load imbalance: " << maxLoad * 1.0 / avgNumEdges << '\n';
            out << "Column and row max loads:\n";
            for (auto r: utils::trans(loads, symmetric)) {
                for (auto i: r)
                    out << i << '\t';
                out << '\n';
            }
        }

        /**
        * @brief Get the transpose of the graph
        **/
        Matrix transpose() const {
            Matrix A;
            A.M_ = N();
            
            std::vector<std::atomic<Ordinal>> indptr_(M + 1);
            std::vector<Ordinal> dloc(NNZ());

            const auto num_threads = std::thread::hardware_concurrency();
            const auto v_arr = nicol1d::partition_prefix(indptr, num_threads * 8); //overdecomposing

            std::for_each(exec_policy, v_arr.cbegin(), v_arr.cend() - 1, [&](const auto &begin) {
                const auto tid = std::distance(&v_arr[0], &begin);
                const auto end = v_arr[tid + 1];
                for (Ordinal i = begin; i < end; i++)
                    for (Ordinal j = indptr[i]; j < indptr[i + 1]; j++)
                        dloc[j] = indptr_[indices[j] + 1].fetch_add(1, std::memory_order_relaxed);
            });

            A.indptr_.assign(indptr_.begin(), indptr_.end());
            std::inclusive_scan(exec_policy, A.indptr_.begin(), A.indptr_.end(), A.indptr_.begin());

            A.indices_.assign(NNZ(), 0);
            A.data_.resize(is_pattern() ? 0 : NNZ());
            
            std::for_each(exec_policy, indptr.begin(), indptr.end() - 1, [&](const auto &indptr_i) {
                const auto i = std::distance(&indptr[0], &indptr_i);
                for (Ordinal j = indptr[i]; j < indptr[i + 1]; j++) {
                    const auto loc = A.indptr_[indices[j]] + dloc[j];
                    A.indices_[loc] = i;
                    if (!is_pattern())
                        A.data_[loc] = data_[j];
                }
            });

            return A;
        }

        Matrix & sort() {
            if (!sorted) {
                std::for_each(exec_policy, indptr.begin(), indptr.end() - 1, [&](const auto &indptr_i) {
                    const auto i = std::distance(&indptr[0], &indptr_i);
                    std::sort(indices_.begin() + indptr_i, indices_.begin() + indptr[i + 1]);
                });
                sorted_ = true;
            }
            return *this;
        }
    
        /**
        * @brief Return a sparsified matrix
        * @param prob Sparsification factor
        * @param seed Seed for coin tosses
        **/
        auto sparsify(const double keep_prob, const int seed = 1) const {
            auto mask = utils::sparse_mask(indptr.back(), keep_prob, seed);
            Matrix A;
            A.M_ = M;
            A.sorted_ = sorted;
            A.indptr_.resize(indptr.size());
            A.indptr_[0] = 0;
            for (Ordinal i = 0; i < N(); i++) {
                A.indptr_[i + 1] = A.indptr_[i];
                for (auto j = indptr[i]; j < indptr[i + 1]; j++)
                    if (mask[j]) {
                        A.indices_.push_back(indices[j]);
                        if (!is_pattern())
                            A.data_.push_back(data(j));
                        A.indptr_[i + 1]++;
                    }
            }
            // std::cerr << "NNZ after sparsification: " << A.NNZ() << std::endl;
            return A;
        }

        /**
        * @brief Generalized order function
        * @param type ASC, DSC, RCM or NAT
        * @param triangular Flag to use only upper half of matrix
        * @param reverse decides rcm is reversed
        * @note that reverse is by default true for ASC, and false for DSC
        **/
        Matrix order(Order order = Order::NAT, bool triangular = false, bool reverse = false) const {
            // No order is specified and all matrix is used
            if (order == Order::NAT && !triangular)
                return *this;

            auto get_order_map = [&] () {
                if (order == Order::RCM) 
                    return rcm_helper(reverse);
                std::vector<Ordinal> map(N());
                // If not rcm, it is asc or dsc
                std::vector<std::pair<Ordinal, Ordinal>> degrees;
                degrees.reserve(N());
                for (size_t i = 0; i < N(); i++)
                    degrees.push_back({i, indptr[i + 1] - indptr[i]});

                std::sort(degrees.begin(), degrees.end(), [order](const auto &u, const auto &v) {
                    return order == Order::ASC ? u.second < v.second : u.second > v.second;
                });
                for (size_t i = 0; i < N(); i++)
                    map[degrees[i].first] = i;
                return map;
            };

            auto map = get_order_map();
            std::vector<Ordinal> t_indptr(N()+1);
            for (size_t i = 0; i < N(); i++) {
                Ordinal count = 0;
                for (size_t j = indptr[i]; j < indptr[i + 1]; j++) {
                    // If using only upper triangle
                    if (triangular) { 
                        if (map[indices[j]] > map[i]) { count++; }
                    } else {
                        count++;
                    }
                }
                t_indptr[map[i] + 1] = count;
            }
            std::partial_sum(t_indptr.begin(), t_indptr.end(), t_indptr.begin());

            std::vector<Ordinal> t_indices(t_indptr.back());
            std::vector<Value> t_data(is_pattern() ? 0 : NNZ());
            for (Ordinal i = 0; i < N(); i++) {
                auto st = t_indptr[map[i]];
                for (auto j = indptr[i]; j < indptr[i + 1]; j++) {
                    // If using only upper triangle
                    if (triangular) {
                        if (map[indices[j]] > map[i]) {
                            if (!is_pattern())
                                t_data[st] = map[data(j)];
                            t_indices[st++] = map[indices[j]];
                        }
                    } else {
                        if (!is_pattern())
                            t_data[st] = map[data(j)];
                        t_indices[st++] = map[indices[j]];
                    }
                }
                std::sort(std::next(t_indices.begin(), t_indptr[map[i]]),
                    std::next(t_indices.begin(), st));
            }
            return Matrix(std::move(t_indptr), std::move(t_indices), std::move(t_data), M);
        }

        /**
        * @brief serialize the graph into a file
        * @param filename
        **/
        void serialize(std::string filename) const {
            // std::cerr << "Serializing graph into " << filename << std::endl;
            std::ofstream of(filename, std::ios::out | std::ios::binary);
            of << N() << indptr.back();
            for (auto i: indptr)
                of << i;
            for (auto j: indices)
                of << j;
            for (Ordinal i = 0; i < NNZ(); i++)
                of << data(i);
        }

        /**
        * @brief take a partition vector and serialize as block csr
        * @param cuts partition vector 
        * @param filename output file name
        **/
        void serialize_block(const std::vector<Ordinal> cuts, std::string filename) const {
            const auto p = cuts.size()-1;

            // memory allocation for the block csr
            std::vector<Ordinal> blockptr(p*p+1);
            std::vector<Ordinal> b_indptr((N()+1)*p);
            std::vector<Ordinal> b_indices(M());
            std::vector<Value> b_data(M());

            // block pointer
            for( size_t i=0; i<p; i++ ){
                for( size_t j=0; j<p; j++ ){
                    blockptr[ i*p+j+1 ] = blockptr[ i*p+j ];
                    blockptr[ i*p+j+1 ] += cuts[ i+1 ]-cuts[ i ];
                }
            }

            // block indice pointers
            Ordinal y =0;
            for( size_t u=0; u<N(); u++ ){
                if( u>=cuts[ y+1 ] ){ ++y; }
                Ordinal x = 0;
                for( size_t v=indptr_[ u ]; v<indptr_[ u+1 ]; v++ ){
                    while( indices_[ v ]>=cuts[ x+1 ] ){ ++x; }
                    b_indptr[ blockptr[ y*p+x ]+(u-cuts[ y ])+1 ] += 1;
                }
            }

            std::partial_sum(b_indptr.begin(), b_indptr.end(), b_indptr.begin());

            // filling block csr
            y=0;
            for( size_t u=0; u<N(); u++ ){
                if( cuts[ y+1 ]<=u ){ ++y; }

                size_t i=0, x = 0;
                for( size_t v=indptr_[ u ]; v<indptr_[ u+1 ]; v++ ){
                    while( cuts[ x+1 ]<=indices_[ v ] ){
                        i = b_indptr[ blockptr[ y*p + (++x) ]+(u-cuts[ y ]) ];
                    }
                    b_indices[ i ] = indices[ v ] - cuts[ x ];
                    b_data[ i++ ] = (data_.size()==M()) ? data_[ i-1 ] : 1 ;
                }
            }

            std::ofstream of(filename, std::ios::out | std::ios::binary);
            of << N() << M() << p;
            for (const auto & i: cuts) of << i;
            for (const auto & i: blockptr) of << i;
            for (const auto & i: b_indptr) of << i;
            for (const auto & i: b_indices) of << i;
            for (const auto & i: b_data) of << i;
        }

        /**
        * @brief simple override of graph ostream
        **/
        friend std::ostream &operator<<(std::ostream &os, const Matrix &g) {
            for (size_t i = 0; i < g.indptr.size() - 1; i++) {
                os << i << " : ";
                for (size_t j = g.indptr[i]; j < g.indptr[i + 1]; j++)
                    os << g.indices[j] << " ";
                os << std::endl;
            }
            return os;
        }

    private:
        /**
        * @brief Helper function to return the mapping
        * Just a revisited version of Apo's implementation of
        * CuthillMcKee algorithm
        **/
        std::vector<Ordinal> rcm_helper(bool reverse) const {
            const auto INF = std::numeric_limits<Ordinal>::max();

            std::vector<std::pair<Ordinal, Ordinal>> not_visited;
            not_visited.reserve(N());
            for (Ordinal i = 0; i < N(); i++)
                not_visited.push_back({i, indptr[i + 1] - indptr[i]});

            std::sort(not_visited.begin(), not_visited.end(), [] (const auto & u, const auto & v) {
                return u.second < v.second;
            });

            std::vector<Ordinal> t_indices(N());
            for (size_t i = 0; i < N(); i++)
                t_indices[not_visited[i].first] = i;

            Ordinal map_index = 0;
            Ordinal min_index = 0;
            std::queue<Ordinal> q;
            std::vector<Ordinal> map(N());
            while (min_index < N()) {
                q.push(not_visited[min_index].first);
                t_indices[q.front()] = INF;
                while (!q.empty()) {
                    std::vector<std::pair<Ordinal, Ordinal>> sorted;
                    for (size_t i = indptr[q.front()]; i < indptr[q.front() + 1]; i++) {
                        auto j = t_indices[indices[i]];
                        if (j != INF && indices[i] != q.front()) {
                            sorted.push_back({not_visited[j].first, not_visited[j].second});
                            t_indices[indices[i]] = INF;
                        }
                    }
                    std::sort(sorted.begin(), sorted.end(), [] (const auto & u, const auto & v) {
                        return u.second < v.second;
                    });
                    for (const auto & ele: sorted)
                        q.push(ele.first);
                    
                    map[q.front()] = map_index++;
                    q.pop();
                }

                while (min_index<N() && t_indices[not_visited[min_index].first]==INF) {
                    min_index++;
                }

            }
            
            /** @note This is where CuthillMcKee ends, now we adjust the map and return **/
            if (reverse) 
                for (size_t i = 0; i < N(); i++)
                    map[i] = N() - map[i];

            return map;
        }
    };
}

#include "sparse_prefix_sum.hpp"

template <class Ordinal, class Value>
const auto & sarma::Matrix<Ordinal, Value>::get_sps() const {
    return *sps_ ? **sps_ : *(*sps_ = std::make_unique<sparse_prefix_sum<Ordinal, Value>>(*this));
}