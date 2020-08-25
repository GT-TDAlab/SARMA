add_library(mmio
        STATIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/external>
        mmio.c)
set_target_properties(mmio PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/mmio.h")

include_directories(include)
