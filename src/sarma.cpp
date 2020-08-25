#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <functional>
#include <iomanip>

#include "sarma.hpp"

using namespace sarma;

struct Parameters {
    std::string file_name = "-";
    ns_filesystem::path dir_name = "";
    ns_filesystem::path res_file = "";
    std::string alg = "pal";
    Order order_type = Order::NAT;
    Ordinal p = 8;
    Ordinal q = 0;
    Value z = 0;
    int seed = 2147483647;
    double sparsify = 1.0;
    bool triangular = false;
    bool serialize = false;
    bool use_data = false;
    std::string order_str = "nat";
};

void print_usage();
int parse_parameters(Parameters &params, int argc, const char **argv );

int main(int argc, const char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    Parameters params;

    if (parse_parameters(params, argc, argv)) {
        print_usage();
        return -1;
    }

    std::cout << "=========================" << std::endl;
    std::cout << "Graph     : " << params.file_name << std::endl;
    std::cout << "Algorithm : " << params.alg << std::endl;
    std::cout << "Order     : " << params.order_str << std::endl;
    std::cout << "P         : " << params.p << std::endl;
    std::cout << "Q         : " << params.q << std::endl;
    std::cout << "Z         : " << params.z << std::endl;
    std::cout << "=========================" << std::endl;
    
    const auto algs = get_algorithm_map<Ordinal, Value>();

#if defined(ENABLE_CPP_PARALLEL)    
    const auto [p, q] = Run<Ordinal, Value>(algs.at(params.alg).first, std::cout, params.file_name, params.order_type, params.p, params.q, params.z, params.triangular,
                            params.serialize, params.sparsify, algs.at(params.alg).second, params.use_data, params.seed);
#else
    const auto pq = Run<Ordinal, Value>(algs.at(params.alg).first, std::cout, params.file_name, params.order_type, params.p, params.q, params.z, params.triangular,
                            params.serialize, params.sparsify, algs.at(params.alg).second, params.use_data, params.seed);
    const auto p = pq.first;
    const auto q = pq.second;
#endif    
    
    if (!params.res_file.empty()) {
        std::ofstream out(params.res_file);
        out << p.size() - 1 << '\n';
        for (auto x: p)
            out << x << ' ';
        out << '\n';
        if (p != q) {
            out << q.size() - 1 << '\n';
            for (auto x: q)
                out << x << ' ';
            out << '\n';
        }
    }

    return 0;
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

void print_usage(){
    std::cout << "SARMA: SpatiAl Rectilinear Matrix pArtitioning" << std::endl;
    std::cout << "SARMA 1.0 - Copyright (c) 2020 by GT-TDAlab" << std::endl;
    std::cout << "syntax: sarma --graph <path> [OPTION].. [--help | -h]" << std::endl;
    std::cout << "Options:                                                           (Defaults:)" << std::endl;
    std::cout << "  --dir <path>       Path for the dataset directory.               ()" << std::endl;
    std::cout << "  --alg <str>        Name of the partitionin algorithm.            (pal)" << std::endl;
    std::cout << "                     Algorithms: [uni | nic | rac | pal | opal]" << std::endl;
    std::cout << "                     uni: Uniform partitioning" << std::endl;
    std::cout << "                     nic: Nicol's 2D rectilinear partitioning" << std::endl;
    std::cout << "                     rac: Refine a Cut partitioning" << std::endl;
    std::cout << "                     pal: Probe a Load partitioning" << std::endl;
    std::cout << "                     opal: Ordered Probe a Load partitioning" << std::endl;
    std::cout << "  --use-data         Use weight of nonzero elements in the matrix  (false)" << std::endl;
    std::cout << "                     Note: Otherwise matrix is considered binary." << std::endl;
    std::cout << "  --order <str>      Row ordering algorithm.                       (nat)" << std::endl;
    std::cout << "                     Orderings: [nat | asc | dsc | rcm]" << std::endl;
    std::cout << "                     nat: Natural order of the matrix" << std::endl;
    std::cout << "                     asc: Ascending nnz ordering" << std::endl;
    std::cout << "                     dsc: Descending nnz ordering" << std::endl;
    std::cout << "                     rcm: Reverse Cuthill-McKee ordering" << std::endl;
    std::cout << "  --res-file <path>  Filename for the partition vector(s).         ()" << std::endl;
    std::cout << "  --use-upper        Use only upper half of the matrix.            (false)" << std::endl;
    std::cout << "  --z <int>          Target load. Available on [uni | rac | pal].  (0)" << std::endl;
    std::cout << "  --p <int>          Number of partitions in the first dimension.  (8)" << std::endl;
    std::cout << "  --q <int>          Number of partitions in the second dimension. (0)" << std::endl;
    std::cout << "                     Note: If the algorithm is symmetric; q=p." << std::endl;
    std::cout << "  --sparsify <real>  Sparsification factor.                        (1.0)" << std::endl;
    std::cout << "  --seed <int>       Seed for random number generator.             (2147483647)" << std::endl;
    std::cout << "                     Note: used when '--sparsify' != 1.0" << std::endl;
    std::cout << "  --serialize        Serialize the output matrix.                  (false)" << std::endl;
    std::cout << "  --list             List matrices under '--dir'                   ()" << std::endl << std::endl;;

    std::cout << "Examples:" << std::endl;
    std::cout << "  sarma --graph email-Eu-core.mtx --alg pal -p 4" << std::endl;
    std::cout << "  sarma --graph email-Eu-core.mtx --alg pal -z 1000" << std::endl;
    std::cout << "  sarma --graph email-Eu-core.mtx --alg pal -p 4 --sparsify 0.1" << std::endl;

    std::cout << "Note:" << std::endl;
    std::cout << "  A sparsification value, s, greater than 1.0 means that the load imbalance, L" <<std::endl;
    std::cout << "  will be off on the order of s." << std::endl;
    std::cout << "  If s=100 then load imbalance is not going to be worse than 1%." << std::endl;
    std::cout << "  sarma --graph email-Eu-core.mtx --alg pal -p 4 --sparsify 100" << std::endl;
}

int parse_parameters(Parameters &params, int argc, const char **argv ){
    const auto algs = get_algorithm_map<Ordinal, Value>();
    const auto order_types = get_order_map();

    if (argc == 1)
        return -1;

    for (int i = 1; i < argc; i++) {
        if (!strcasecmp(argv[i], "--graph") || !strcasecmp(argv[i], "-g")) {
            params.file_name = std::string(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--dir") || !strcasecmp(argv[i], "-d")) {
            params.dir_name = std::string(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--alg") || !strcasecmp(argv[i], "-a")) {
            params.alg = argv[++i];
            if (algs.find(params.alg) == algs.end()) {
                std::cout << "Wrong algorithm type." << std::endl;
                return -1;
            }
        }
        else if (!strcasecmp(argv[i], "--order") || !strcasecmp(argv[i], "-o")) {
            std::string s = argv[++i];
            if (order_types.find(s) == order_types.end()) {
                std::cout << "Wrong order type." << std::endl;
                return -1;
            }
            params.order_type = order_types.at(s);
            params.order_str = s;
        }
        else if (!strcasecmp(argv[i], "--p") || !strcasecmp(argv[i], "-p")) {
            params.p = std::stoi(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--q") || !strcasecmp(argv[i], "-q")) {
            params.q = std::stoi(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--z") || !strcasecmp(argv[i], "-z")) {
            params.z = std::stoi(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--sparsify") || !strcasecmp(argv[i], "-s")) {
            params.sparsify = std::stod(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--res-file")) {
            params.res_file = std::string(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--seed")) {
            params.seed = std::stoi(argv[++i]);
        }
        else if (!strcasecmp(argv[i], "--use-upper")) {
            params.triangular = true;
        }
        else if (!strcasecmp(argv[i], "--serialize")) {
            params.serialize = true;
        }
        else if (!strcasecmp(argv[i], "--use-data")) {
            params.use_data = true;
        }
        else if (!strcasecmp(argv[i], "--list") || !strcasecmp(argv[i], "-l")) {
            const auto gdir = params.dir_name;
            std::vector<std::string> graphs;

            for (const auto &g: ns_filesystem::directory_iterator(gdir))
                if (g.path().extension() == ".mtx" || g.path().extension() == ".bin")
                    graphs.push_back(g.path().filename());

            std::sort(graphs.begin(), graphs.end());

            const std::size_t cw = 4;
            const auto rcnt = (graphs.size() + cw - 1) / cw;
            std::vector<size_t> maxlen(cw);
            for (std::size_t i = 0; i < graphs.size(); i++)
                maxlen[i / rcnt] = std::max(maxlen[i / rcnt], graphs[i].size());

            std::cout << "======================================================================" << std::endl;
            std::cout << "There are " << graphs.size() << " graphs under " << gdir << std::endl;
            std::cout << "======================================================================" << std::endl;

            std::cout << std::left;
            for (size_t r = 0; r < rcnt; r++) {
                for (std::size_t i = r; i < graphs.size(); i += rcnt)
                    std::cout << std::setw(maxlen[i / rcnt] + 2) << graphs[i];
                std::cout << '\n';
            }

            std::cout << std::endl << std::endl;

            std::exit(0);
        }
        else if (!strcasecmp(argv[i], "--help") || !strcasecmp(argv[i], "-h")) {
            print_usage();
            std::exit(0);
        } else {
            std::cout << "Wrong argument '" << argv[i] << "' given." << std::endl;
            return -1;
        }
    }
    if (params.file_name != "-")
        params.file_name = params.dir_name / params.file_name;
    return 0;
}
