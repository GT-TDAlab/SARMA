#pragma once

#define GCC_VERSION (__GNUC__ * 10000 + \
                     __GNUC_MINOR__ * 100 + \
                     __GNUC_PATCHLEVEL__)

#if GCC_VERSION < 90000
#include <experimental/filesystem>
namespace ns_filesystem = std::experimental::filesystem;

#define DISABLE_CPP_PARALLEL
#else
#include <filesystem>
#include <execution>

#define ENABLE_CPP_PARALLEL

constexpr auto exec_policy = std::execution::par_unseq;
namespace ns_filesystem = std::filesystem;
#endif

#include "algorithms/algorithm.hpp"
#include "data_structures/csr_matrix.hpp"

using Ordinal = unsigned;
using Value = Ordinal;
