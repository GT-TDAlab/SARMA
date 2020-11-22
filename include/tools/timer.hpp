#pragma once

#include <chrono>
#include <string>
#include <mutex>
#include <iostream>

namespace sarma {

    static std::mutex iomutex;
    class timer {
        const std::string name;
        const std::chrono::high_resolution_clock::time_point start;
        std::mutex &mtx;
        std::ostream &os;
    public:
        timer(std::string s, std::mutex &_mtx = iomutex, std::ostream &_os = std::cerr) : name(s), start(std::chrono::high_resolution_clock::now()), mtx(_mtx), os(_os) {
#if defined(ENABLE_CPP_PARALLEL)
            std::lock_guard lock(mtx);
#else
            std::lock_guard<std::mutex> lock(mtx);
#endif
            #ifdef DEBUG
                os << name << " has started" << std::endl;
            #endif
        }
        ~timer() {
#if defined(ENABLE_CPP_PARALLEL)
            std::lock_guard lock(mtx);
#else
            std::lock_guard<std::mutex> lock(mtx);
#endif
            #ifdef DEBUG
                os << name << " took " << time() << "s" << std::endl;
            #endif
        }
        double time() const {
            return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
        }
    };
}