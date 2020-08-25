#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include <exception>
#include <functional>

#include "sarma.hpp"

/**
 * 
 * Arguments are given as #graphs graphs, #partition type, partition type, #, tile size ...
 * 
 * */

using namespace sarma;

int main(int /*argc*/, const char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    const std::string GRAPH_DIR = std::getenv("GRAPH_DIR");

    unsigned ap = 1;

    std::size_t count = std::stoi(argv[ap++]);
    std::vector<std::string> graphs;
    for (size_t i = 0; i < count; i++) graphs.push_back(argv[ap++]);

    count = std::stoi(argv[ap++]);
    std::vector<std::string> orders;
    for (size_t i = 0; i < count; i++) orders.push_back(argv[ap++]);

    count = std::stoi(argv[ap++]);
    std::vector<std::string> algorithms;
    for (size_t i = 0; i < count; i++) algorithms.push_back(argv[ap++]);

    count = std::stoi(argv[ap++]);
    std::vector<Ordinal> cuts;
    for (size_t i = 0; i < count; i++) cuts.push_back(std::stoi(argv[ap++]));

    count = std::stoi(argv[ap++]);
    std::vector<double> probs;
    for (size_t i = 0; i < count; i++) probs.push_back(std::stod(argv[ap++]));

    count = std::stoi(argv[ap++]);
    std::vector<int> seeds;
    for (size_t i = 0; i < count; i++) seeds.push_back(std::stoi(argv[ap++]));

    ns_filesystem::path outpath(argv[ap++]);
    const auto triangular = false;
    const auto serialize = false;
    const auto use_data = false;

    const auto algs = get_algorithm_map<Ordinal, Value>();
    const auto ords = get_order_map();

    for (auto graph: graphs) {
        for (auto order: orders) {
            auto a_ord = AlgorithmOrder<Ordinal, Value>(GRAPH_DIR + graph, ords.at(order), triangular, use_data);
            for (auto seed : seeds) {
                for (auto prob : probs) {
                    for (auto cut : cuts) {
#if defined(ENABLE_CPP_PARALLEL)
                        auto[a_sp, sp_time, ds_time] = AlgorithmPlan(a_ord, utils::get_prob(a_ord->NNZ(), cut, cut, prob), seed, true);
#else
                        auto terms = AlgorithmPlan(a_ord, utils::get_prob(a_ord->NNZ(), cut, cut, prob), seed, true);
                        auto a_sp = std::get<0>(terms);
                        auto sp_time = std::get<1>(terms);
                        auto ds_time = std::get<2>(terms);
#endif
                        std::for_each(std::begin(algorithms), std::end(algorithms), [&](const auto &algorithm) {
#if defined(ENABLE_CPP_PARALLEL)
                            const auto &[alg, use_sparse] = algs.at(algorithm);
#else
                            const auto &terms = algs.at(algorithm);
                            const auto &alg = terms.first;
                            const auto &use_sparse = terms.second;
#endif
                            auto filename = outpath / order / graph / ns_filesystem::path(
                                    algorithm + "$" + std::to_string(cut) + "$" + std::to_string(prob) +
                                    "$" + (use_sparse ? "w" : "wo") + "$" +
                                    std::to_string(seed) + "$out");
                            ns_filesystem::create_directories(filename.parent_path());
                            if (!ns_filesystem::exists(filename)) {
                                try {
                                    std::ofstream out(filename, std::ofstream::out);
                                    AlgorithmRun(alg, out, a_ord, a_sp, cut, cut, (Ordinal)0, serialize,
                                                        use_sparse, seed, sp_time,
                                                        ds_time);
                                } catch (std::exception &e) {
                                    std::cerr << "Error on " << filename << std::endl;
                                    std::cerr << e.what() << std::endl;
                                }
                            }
                        });
                    }
                }

            }
        }
    }

    return 0;
}