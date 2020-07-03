# Crated by mfbalin on May 29, 2020.
# -------------------------------------------------------------

find_package(pybind11 QUIET)

if (pybind11_FOUND)
        pybind11_add_module(_sarma sarma.cpp)
else ()
    message(WARNING "pybind11 is needed to build python bindings!")
endif()
