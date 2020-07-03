#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "algorithms/algorithm.hpp"
#include "data_structures/csr_matrix.hpp"
#include "data_structures/sparse_prefix_sum.hpp"

#include "tests.h"

using Ordinal = unsigned;
using Value = unsigned;

namespace sarma::test_utils {

    template<typename Ordinal, typename Value>
    auto get_test_matrices(bool use_data=false) {
        const auto gdir = std::getenv("TEST_MTX_DIR");
        std::vector<sarma::Matrix<Ordinal, Value>> matrices;

        if (!std::filesystem::exists(gdir))
            return matrices;

        for (const auto &g: std::filesystem::directory_iterator(gdir))
            if (g.path().extension() == ".mtx" || g.path().extension() == ".bin")
                matrices.emplace_back(g, use_data);

        return matrices;
    }

    template<typename Ordinal, typename Value>
    auto get_test_mtx_bin_matrices() {
        const auto gdir = std::getenv("TEST_MTX_DIR");
        std::vector<std::pair<sarma::Matrix<Ordinal, Value>, sarma::Matrix<Ordinal, Value>>> matrices;

        if (!std::filesystem::exists(gdir))
            return matrices;

        for (auto &g: std::filesystem::directory_iterator(gdir))
            if (g.path().extension() == ".mtx") {
                auto bin_path = g.path();
                bin_path.replace_extension(".bin");
                if (std::filesystem::exists(bin_path))
                    matrices.emplace_back(std::make_pair(
                        sarma::Matrix<Ordinal, Value>(g, false),
                        sarma::Matrix<Ordinal, Value>(bin_path, false)
                    ));
            }
        
        return matrices;
    }

    bool endswith(const std::string & str, const std::string & suffix) {
        auto ss = str.size();
        auto fs = suffix.size();
        return (ss >= fs) && str.compare(ss - fs, fs, suffix) == 0;
    }

    template<typename T>
    auto read_vector_bin(std::istream & in_stream) {
        std::vector<T> data;

        try{
            long n_elements = 0;
            in_stream.read((char*)&n_elements, sizeof(n_elements));

            for (long i=0; i<n_elements && i < 10; i++) {
                T t;
                in_stream.read((char*)&t, sizeof(T));
                data.push_back(t);
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << e.what() << '\n';
            std::exit(1);
        }

        return data;
    }
    
    template<typename T>
    auto read_vector(std::istream & in_stream) {
        std::vector<T> data;

        try{
            long n_elements = 0;
            in_stream >> n_elements;

            for (long i=0; i<n_elements && i < 10; i++) {
                T t;
                in_stream >> t;
                data.push_back(t);
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << e.what() << '\n';
            std::exit(1);
        }

        return data;
    }

    template<typename T>
    auto read_vector(const std::string & fname) {

        std::streambuf * buf;
        std::ifstream inf;

        try{
            if (fname == "-") {
                buf = std::cin.rdbuf();
            } else {
                if (endswith(fname, ".bin"))
                    inf.open(fname, std::ios::binary);
                else
                    inf.open(fname, std::ios::in);

                if (!inf.good()) {
                    throw std::ios_base::failure("opening files to read failing");
                }
                buf = inf.rdbuf();
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << "Error on reading " << fname << std::endl;
            std::cerr << e.what() << '\n';
            std::exit(1);
        }

        std::istream in(buf);
        if (endswith(fname, ".bin"))
            return read_vector_bin<T>(in);
        else
            return read_vector<T>(in);
    }

    template<typename T>
    bool write_vector_bin(std::ostream & out_stream, const std::vector<T> & data) {

        try{
            long n_elements = data.size();
            out_stream.write((char *)&n_elements, sizeof(n_elements));

            for (long i=0; i<n_elements; i++) {
                T t = data[i];
                out_stream.write((char *)&t, sizeof(t));
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << e.what() << '\n';
            return false;
        }

        return true;
    }

    template<typename T>
    bool write_vector(std::ostream & out_stream, const std::vector<T> & data) {

        try{
            long n_elements = data.size();
            out_stream << (long)n_elements << std::endl;

            for (long i=0; i<n_elements; i++) {
                out_stream << data[i] << std::endl;
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << e.what() << '\n';
            return false;
        }

        return true;
    }

    template<typename T>
    auto write_vector(const std::string & fname, const std::vector<T> & data) {

        std::streambuf * buf;
        std::ofstream of;

        try{
            if (fname == "-")
                buf = std::cout.rdbuf();
            else {
                if (endswith(fname, ".bin"))
                    of.open(fname, std::ios::binary);
                else
                    of.open(fname, std::ios::out);

                if (!of.good()) {
                    throw std::ios_base::failure("opening files to write failing");
                }

                buf = of.rdbuf();
            }

        } catch (std::ios_base::failure & e) {
            std::cerr << "Error on writing " << fname << std::endl;
            std::cerr << e.what() << '\n';
            return false;
        }

        std::ostream out(buf);
        if (endswith(fname, ".bin"))
            return write_vector_bin<T>(out, data);
        else
            return write_vector<T>(out, data);
    }

    template<typename Ordinal>
    bool is_valid(const std::vector<Ordinal> &p, const Ordinal N){
        if (p[0] != 0 || p.back() != N)
            return false;
        for (size_t i = 0; i < p.size() - 1; i++)
            if (p[i] > p[i + 1])
                return false;
        return true;
    }

    template<typename Ordinal>
    int is_same(const std::vector<Ordinal> &a, const std::vector<Ordinal> &b){
        int pass = 0;
        EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size() - 1; i++)
            EQ(a[i], b[i]);
        return pass;
    }

    template <typename Ordinal, typename Value, typename alg_t>
    int test_alg(alg_t alg, const sarma::Matrix<Ordinal, Value> &A, const Ordinal P, const Ordinal Q, const Ordinal Z) {
        const auto [ans, _] = alg(A, P, Q, Z, 1);
        return is_valid(ans, A.N()) ? 0 : -1;
    }

    template <typename Ordinal, typename Value>
    int test_alg_on_mtxs(const std::vector<sarma::Matrix<Ordinal, Value>> &mtxs, const std::string alg_str, const bool mnc=false) {
        int pass = 0;
        
        const auto alg = sarma::get_algorithm_map<Ordinal, Value>().at(alg_str).first;

        for (const auto &mtx: mtxs) {
            const auto nnz = mtx.NNZ();
            for (Ordinal i = 2; i < 33; i *= 2){
                if (mnc){
                    TEST(test_alg, alg, mtx, (Ordinal)0, (Ordinal)0, nnz / i);
                } else {
                    TEST(test_alg, alg, mtx, i, i, (Ordinal)0);
                }
            }
        }

        return pass;
    }

    template<typename Ordinal>
    auto parti_vect(const Ordinal p, const Ordinal N){
        std::vector<Ordinal> ans(p+1);
        for (Ordinal i=0; i<p; i++)
            ans[i] = i;
        ans[p]=N;

        return ans;
    }

    template<typename Ordinal, typename Value>
    auto are_same_mtx(const sarma::Matrix<Ordinal, Value> &A_, const sarma::Matrix<Ordinal, Value> &B_) {
        auto A = A_, B = B_;
        A.sort(); B.sort();
        if (A.indptr != B.indptr)
            return -1;
        if (A.indices != B.indices)
            return -1;
        if (A.get_data() != B.get_data())
            return -1;
        if (A.M != B.M)
            return -1;
        return 0;
    }

    template<typename Ordinal, typename Value>
    auto is_valid_mtx(const sarma::Matrix<Ordinal, Value> &A) {
        for (Ordinal i=0; i<A.N(); i++)
            if (A.indptr[i+1]<A.indptr[i])
                return -1;

        for (Ordinal i=0; i<A.NNZ(); i++)
            if (A.indices[i]<0 || A.indices[i]>=A.N())
                return -1;

        if (A.indptr[A.N()]!=A.indices.size())
            return -1;
        
        if (!A.is_pattern() && A.get_data().size()!=A.indices.size())
            return -1;

        return 0;
    }   
}
