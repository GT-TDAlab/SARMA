#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <memory>
#include <map>

#include "sarma.hpp"
#include "tools/progress_bar.hpp"

/**
 * 
 * Benchmark takes a configuration file, and creates a csv file
 * If previous results are present in the file, the previous results are
 * taken as baseline.
 * 
 * */

using namespace sarma;

const auto HASHTAG = '#';
const auto DELIM = '\t';
const auto PRECISION = 3;
const auto EPSILON = 1e-7;

struct Parameters {
    std::filesystem::path graph_dir = "";  // Graph locations
    std::filesystem::path input_path = ""; // Config file
    std::filesystem::path output_dir = ""; // Default is std::cout
};

struct noop {
    void operator()(...) const {}
};

/**
 * 
 * Simple helper structure to initalize config
 * This struct might be useful if we want to sort by 
 * some field and group the runs
 * */
struct Config {
    Config() = default;
    Config(std::istream& is) { is >> *this; }

    std::string graph;
    std::string order;
    std::string algorithm;
    Ordinal p;
    double prob;
    int seed;
    double load_imbalance;
    double time;
    double norm_load_imbalance;
    double norm_time;

    friend std::istream& operator>>(std::istream &in, Config& c) {
        std::getline(in, c.graph, DELIM);
        std::getline(in, c.order, DELIM);
        std::getline(in, c.algorithm, DELIM);
        std::string holder = "";
        std::getline(in, holder, DELIM);
        c.p = std::stoi(holder);
        std::getline(in, holder, DELIM);
        c.prob = std::stod(holder);
        std::getline(in, holder, DELIM);
        c.seed = std::stoi(holder);
        std::getline(in, holder, DELIM);
        c.load_imbalance = std::stod(holder);
        std::getline(in, holder, DELIM);
        c.time = std::stod(holder);
        std::getline(in, holder, DELIM);
        c.norm_load_imbalance = std::stod(holder);
        std::getline(in, holder); // Last line should be newline
        c.norm_time = std::stod(holder); 
        return in;
    }

    friend std::ostream& operator<<(std::ostream &out, const Config& c) {
        return out << std::fixed << std::setprecision(PRECISION)
                << c.graph << DELIM << c.order << DELIM << c.algorithm << DELIM << c.p << DELIM
                << c.prob << DELIM << c.seed << DELIM 
                << c.load_imbalance << DELIM << c.time << DELIM << c.norm_load_imbalance << DELIM << c.norm_time << '\n';
    }
};

void print_usage();
int parse_parameters(Parameters &params, int argc, const char **argv);

int main(int argc, const char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    Parameters params;
    if (parse_parameters(params, argc, argv)) {
        print_usage();
        return -1;
    }

    // Use cin if graph_dir is not specified
    std::shared_ptr<std::istream> config_stream;
    if (params.input_path == "") {
        config_stream.reset(&std::cin, noop());
    } else {
        config_stream.reset(new std::ifstream(params.input_path, std::ios::in));
    }
    // Then there is no config file
    if (!config_stream->good()) {
        std::cerr << "Failed to open config." << std::endl;
        return -1;
    }

    /**
     * 
     * Format is 10 columns:
     * 
     * graph, order, algorithm, p, prob, seed, load_imbalance, time, norm_imbalance, norm_time
     * 
     * If load_imbalance and time is marked as -1, it means the previous results
     * are not available, so we are not going to try to compare.
     * 
     * */
    std::map<std::pair<std::string, std::string>, std::map<std::tuple<Ordinal, double, int>, std::vector<std::pair<size_t, Config> >>> base_configs;
    // Peek at the end of every iteration to understand eof
    size_t num_config = 0;
    for (auto lino = 0; config_stream->good(); lino++, config_stream->peek()) {
        try {
            if (config_stream->peek() == '\n') {config_stream->get();} // Ignore newline for simplicity
            // Skip comments
            if (config_stream->peek() == HASHTAG) {
                std::string holder = "";
                std::getline(*config_stream, holder);
                continue;
            }
            auto config = Config(*config_stream);
            num_config++;
            base_configs[{config.graph, config.order}][{config.p, config.prob, config.seed}].push_back({lino, config});
        } catch (std::exception& e) {
            std::cerr << "Bad config at line: " << lino << std::endl;
            std::cerr << e.what() << std::endl;
            print_usage();
            return -1;
        }
    }
    
    const auto triangular = false;
    const auto use_data = false;
    const auto algs = get_algorithm_map<Ordinal, Value>();
    const auto ords = get_order_map();
    

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream time_stream;
    time_stream << std::put_time(&tm, "sarma-bench_%Y-%m-%d_%H-%M-%S.csv");
    int rc = 0;
    try {
        std::shared_ptr<std::ostream> out;
        if (params.output_dir != "") {
            const auto out_file = params.output_dir / time_stream.str();
            std::filesystem::create_directories(out_file.parent_path());
            out.reset(new std::ofstream(out_file, std::ofstream::out));
        } else {
            out.reset(&std::cout, noop());
        }
        // Header for the file
        *out << "# " << time_stream.str() << "\n"
            << "# Graph\tgraph\torder\talgorithm\tp\tprob\tseed\tload_imbalance\ttime\tnorm_imbalance\tnorm_time\n";
        
        std::vector<std::pair<size_t, Config>> run_configs;
        std::vector<double> run_geomean = {0, 0, 0, 0};
        bool is_first = base_configs.begin()->second.begin()->second.begin()->second.time < 0; // Check if the previous iteration is first run

        utils::progress_bar pb(num_config);
        for (auto& [go, pps] : base_configs) {
            auto a_ord = AlgorithmOrder<Ordinal, Value>(params.graph_dir / go.first, ords.at(go.second), triangular, use_data);
            for (auto& [k, configs] : pps) {
                auto [a_sp, sp_time, ds_time] = AlgorithmPlan(a_ord, utils::get_prob(a_ord->NNZ(), std::get<0>(k), std::get<0>(k), std::get<1>(k)), std::get<2>(k), true);
                for (auto& [index, config] : configs) {
                    try {
                        const auto &[alg, use_sparse] = algs.at(config.algorithm);
                        auto [load_imbalance, par_time] = AlgorithmBenchmark(alg, a_ord, a_sp, config.p, config.p, (Ordinal)0, config.seed);
                        const auto total_time = sp_time + (use_sparse ? ds_time : 0) + par_time;
                        config.norm_load_imbalance = is_first ? 1 : load_imbalance / config.load_imbalance;
                        config.norm_time = is_first ? 1 : total_time / config.time;
                        config.load_imbalance = load_imbalance;
                        config.time = total_time;
                        run_configs.push_back({index, config});

                        run_geomean[0] += std::log(config.load_imbalance);
                        run_geomean[1] += std::log(config.time);
                        run_geomean[2] += std::log(config.norm_load_imbalance);
                        run_geomean[3] += std::log(config.norm_time);
                        pb.update();
                    } catch (std::exception& e) {
                        std::cerr << "Error while running config. Config info" << std::endl;
                        std::cerr << config;
                        std::cerr << e.what() << std::endl;
                        return -1;
                    }
                }
            }
        }
        // Print in the same order received
        std::sort(run_configs.begin(), run_configs.end(), [](const auto& u, const auto& v) {
            return u.first < v.first;
        });
        for (auto& [index, config]: run_configs) {
            *out << config;
        }

        // Write geo-mean to the file
        for (auto& geo : run_geomean) {
            geo = std::exp(geo / run_configs.size());
        }
        *out << std::fixed << std::setprecision(PRECISION) << "# Geomean\tLoad imbalance: " << run_geomean[0] << "\tTime: "
            << run_geomean[1] << "\tNormalized Load Imbalance: " << run_geomean[2] 
            << "\tNormalized time: " << run_geomean[3] << std::endl;
        rc = run_geomean[2] > (1.0 + EPSILON);
    } catch (std::exception &e){
        std::cerr << "Error on while writing output file" << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return rc;
}

void print_usage() {
    std::cout << "SARMA: SpatiAl Rectilinear Matrix pArtitioning"                                 << std::endl;
    std::cout << "SARMA 1.0 - Copyright (c) 2020 by GT-TDAlab"                                    << std::endl;
    std::cout << "Usage: sarma-bench [OPTION].. [--help | -h]"                                    << std::endl;
    std::cout << "Options:                                                        (Defaults)"     << std::endl;
    std::cout << "  -d, --dir    <path>            Directory of the matrices.     ()"             << std::endl; 
    std::cout << "  -i, --input  <path>            Input config file.             (stdin)"        << std::endl;
    std::cout << "  -o, --output <path>            Output directory               (stdout)"       << std::endl << std::endl; 

    std::cout << "Examples:"                                                                      << std::endl;
    std::cout << " sarma-bench -d matrices/ -i matrices/config_file.txt -o matrices/"             << std::endl;
    std::cout << " cat matrices/config_file.txt | sarma-bench -d matrices/ -o matrices/"          << std::endl;
    std::cout << " cat matrices/config_file.txt | sarma-bench"                                    << std::endl << std::endl;

    std::cout << "Output format: "                                                                << std::endl;
    std::cout << "  If output has specified as file, it will be available with a time-stamp."     << std::endl;
    std::cout << "  Otherwise results will be printed to stdout."                                 << std::endl;
    std::cout << "  Each line will have config line format (See below)."                          << std::endl;
    std::cout << "  Last line of the output will contain geomeans of the results."                << std::endl << std::endl;

    std::cout << "Input format:"                                                                  << std::endl;
    std::cout << "  Each line is one configuration to run with proper line format."               << std::endl << std::endl;

    std::cout << "Config Line Format:"                                                            << std::endl;   
    std::cout << "  graph order alg p sparsify seed load_imbalance time norm_imbalance norm_time" << std::endl << std::endl;

    std::cout << "Note:"                                                                          << std::endl;
    std::cout << "  Each column in config line format is tab(\\t)seperated."                      << std::endl;
    std::cout << "  Lines starting with #(Hashtag) are considered as comments and ignored."       << std::endl;
    std::cout << "  Non-negative output columns are considered as baselines."                     << std::endl << std::endl;

    std::cout << "Config Parameter Columns:"                                                      << std::endl;
    std::cout << "  name       type        Description"                                           << std::endl;
    std::cout << "  graph      str         Name of the graph"                                     << std::endl;
    std::cout << "                         If dir is not specified. Name will be takes as path"   << std::endl;
    std::cout << "  order      str         Row ordering algorithm"                                << std::endl;
    std::cout << "                         Orderings: [nat | asc | dsc | rcm]"                    << std::endl;
    std::cout << "                         nat: Natural order of the matrix"                      << std::endl;
    std::cout << "                         asc: Ascending nnz ordering"                           << std::endl;
    std::cout << "                         dsc: Descending nnz ordering"                          << std::endl;
    std::cout << "                         rcm: Reverse Cuthill-McKee ordering"                   << std::endl;
    std::cout << "  alg        str         Name of the partitionin algorithm."                    << std::endl;
    std::cout << "                         Algorithms: [uni | nic | rac | pal | opal]"            << std::endl;
    std::cout << "                         uni: Uniform partitioning"                             << std::endl;
    std::cout << "                         nic: Nicol's 2D rectilinear partitioning"              << std::endl;
    std::cout << "                         rac: Refine a Cut partitioning"                        << std::endl;
    std::cout << "                         pal: Probe a Load partitioning"                        << std::endl;
    std::cout << "                         opal: Ordered Probe a Load partitioning"               << std::endl;
    std::cout << "  p          int         Number of partitions in the first dimension"           << std::endl;
    std::cout << "  sparsify   real        Sparsification factor"                                 << std::endl;
    std::cout << "  seed       int         Seed for random number generator"                      << std::endl;
    std::cout << "                         Used when '--sparsify' != 1.0"                         << std::endl << std::endl;

    std::cout << "Config Output Columns:"                                                         << std::endl;
    std::cout << "  name                  type   Description"                                     << std::endl;
    std::cout << "  load_imbalance        real   Load imbalance of the configuration"             << std::endl;  
    std::cout << "  time                  real   Total run-time of the configuration"             << std::endl;
    std::cout << "  norm_load_imbalance   real   Normalized load imbalance w.r.t. initial config" << std::endl;
    std::cout << "                               Note: If previous results are not available"     << std::endl;
    std::cout << "                               This field will be 1"                            << std::endl;
    std::cout << "  norm_load_time        real   Normalized run-time w.r.t. initial config"       << std::endl;
    std::cout << "                               Note: If previous results are not available"     << std::endl;
    std::cout << "                               This field will be 1"                            << std::endl;     
}

std::string tolower(const std::string &s) {
    std::string r;
    for (const auto c: s)
        r.push_back(std::tolower(c));
    return r;
}

auto strcasecmp(const std::string &a, const std::string &b) {
    return std::strcmp(tolower(a).c_str(), tolower(b).c_str());
}

int parse_parameters(Parameters &params, int argc, const char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcasecmp(argv[i], "--dir") || !strcasecmp(argv[i], "-d")) {
            params.graph_dir = std::string(argv[++i]);
            if (!std::filesystem::exists(params.graph_dir)) {
                std::cerr << "Graph directory does not exist." << std::endl;
                return -1;
            }
        } else if (!strcasecmp(argv[i], "--input") || !strcasecmp(argv[i], "-i")) {
            params.input_path = std::string(argv[++i]);
            if (!std::filesystem::exists(params.input_path)) {
                std::cerr << "Input path does not exist." << std::endl;
                return -1;
            } 
        } else if (!strcasecmp(argv[i], "--output") || !strcasecmp(argv[i], "-o")) {
            params.output_dir = std::string(argv[++i]);
        } else if (!strcasecmp(argv[i], "--help") || !strcasecmp(argv[i], "-h")) {
            print_usage();
            std::exit(0);
        }
    }
    return 0;
}