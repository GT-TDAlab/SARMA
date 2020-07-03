#pragma once

#include <iostream>
#include <string>
#include <iomanip>

namespace sarma::utils {

    class progress_bar {
        std::ostream &os; 
        std::size_t current = 0;
    public:
        const std::size_t capacity;
        const std::size_t width;
        progress_bar(std::size_t capacity, std::ostream &os = std::cerr, std::size_t width = 100) 
            : os(os), capacity(capacity), width(width) {
            os << '\n';
        };

        void update() {
            if (current >= capacity) 
                return;
            const auto pos = current * width / capacity;
            os << std::fixed << std::setprecision(2) << "\033[A\033[2K\r[" << std::string(pos, '=') 
                    << ">" << std::string(width - pos - 1, ' ') << "] " << ++current * 100.0 / capacity << "%" << std::endl;
        }
    };
}