#include <fstream>
#include <vector>
#include <cassert>
#include <iostream>
#include <string>

int main(int /*argc*/, char *argv[]) {
    std::ifstream in(argv[1], std::ios::binary);

    if (!in.is_open())
        std::cerr << "failed to open " << argv[1] << std::endl;
    else {
        std::string fline;
        unsigned n, m;

        std::getline(in, fline);
        in.read(reinterpret_cast<char*>(&n), sizeof n);
        in.read(reinterpret_cast<char*>(&m), sizeof m);

        unsigned prev;
        unsigned cur;
        in.read(reinterpret_cast<char*>(&prev), sizeof prev);

        assert(prev == 0);
        for (unsigned i = 0; i < n; i++) {
            in.read(reinterpret_cast<char*>(&cur), sizeof cur);
            assert(prev <= cur);
            prev = cur;
        }
        
        for (unsigned i = 0; i < cur; i++) {
            unsigned e;
            in.read(reinterpret_cast<char*>(&e), sizeof e);
            assert(e < m);
        }

        assert(!(in >> fline));
    }
}