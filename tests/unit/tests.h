#pragma once

#include <iostream>
#include <stdlib.h>

/**
 * This file contains helper macros for running simple unit tests
 */

#define EQ(x,y)                                                                         \
    do {                                                                                \
        if ((x) != (y)) {                                                               \
            std::cerr << "FAILURE: Expected " << (y) << " instead got " << (x)          \
                << " for " << #x << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                                   \
            }                                                                           \
    } while(0);

#define TEST(f, ...)                                             \
    do {                                                         \
        int pass_ = f(__VA_ARGS__);                              \
        if (pass_ != 0) {                                        \
                std::cerr << "TEST FAILED: " << #f << std::endl; \
            } else {                                             \
                std::cout << "TEST PASSED: " << #f << std::endl; \
            }                                                    \
            pass |= pass_;                                       \
    } while(0);

#define COMMA ,
