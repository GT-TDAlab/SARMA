#include <iostream>
#include <vector>
#include <filesystem>
#include <map>
#include <cstdlib>
#include <iostream>
#include <exception>
#include <execution>
#include <functional>
#include <cassert>

#include "data_structures/csr_matrix.hpp"

using Ordinal = unsigned int;
using Value = Ordinal;

using namespace sarma;

/**
 * 
 * Arguments are given as #graphs graphs, #partition type, partition type, #, tile size ...
 * 
 * */

int main(int /*argc*/, const char *argv[]) {

    std::string graph_dir;
    const std::string SUFFIX = ".bin";

    unsigned ap = 1;
    std::vector<std::string> graphs;
    ns_filesystem::path outpath;

    // Reading args
    try {
        std::size_t count = std::stoi(argv[ap++]);
        for (size_t i = 0; i < count; i++) graphs.push_back(argv[ap++]);

        graph_dir = argv[ap++];
        outpath = argv[ap++];

        if (!ns_filesystem::exists(outpath)) {
            throw std::runtime_error("Output path doesn't exist");
        }
    } catch (std::exception &e) {
        std::cerr <<"Input format error with:" << std::endl;
        std::cerr << e.what() << std::endl;

        std::cerr<<"Usage : "<<argv[0]<<" <N> <N matrix names> <gpath> <opath>"<<std::endl;
        std::cerr<<"    <N> is the number of matrixes to convert"<<std::endl;
        std::cerr<<"    <N matrix names> is the matrixes' names chosen to be converted." << std::endl;
        std::cerr<<"    Matrix should be located at :" << std::endl;
        std::cerr<<"        " << graph_dir << std::endl;
        std::cerr<<"    <opath> is the directory of converted matrixes to be stored at." << std::endl;
        std::cerr<<std::endl;
        std::cerr<<"Usage example:" << std::endl;
        std::cerr<<"    bintospart 2 add20 494_bus ~/matrixes/" << std::endl;
        exit(0);
    }

    for (auto graph: graphs) {

        auto A = std::make_shared<Matrix<Ordinal, Value>>(graph_dir + graph + SUFFIX);
        auto filename = outpath / ns_filesystem::path( graph + ".spart");

        try {
            std::ofstream out(filename);
            out << A->N() << " " << A->N() << std::endl;
            for (Ordinal i=0; i<A->N(); i++) {
                Ordinal k=0;
                for (Ordinal j=A->indptr[i]; j<A->indptr[i+1]; j++) {
                    assert(k <= A->indices[j]);

                    while (k<A->indices[j]) {
                        out << 0 << std::endl;
                        k++;
                    }
                    out << 1 << std::endl;
                    k++;
                }
                while (k<A->N()) {
                    out << 0 << std::endl;
                    k++;
                }
            }
        } catch (std::exception &e) {
            std::cerr << "Error on " << filename << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }

    return 0;
}
