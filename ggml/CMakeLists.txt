cmake_minimum_required(VERSION 3.10)
project(mel_band_roformer_ggml C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Find or build GGML
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(GGML ggml)
endif()

if(NOT GGML_FOUND)
    # Download GGML if not found
    include(FetchContent)
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggerganov/ggml.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(ggml)
    set(GGML_INCLUDE_DIRS ${ggml_SOURCE_DIR}/include)
    set(GGML_LIBRARIES ggml)
endif()

# Source files
set(SOURCES
    mel_band_roformer.c
    mel_band_roformer.h
)

# Library
add_library(mel_band_roformer ${SOURCES})
target_include_directories(mel_band_roformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GGML_INCLUDE_DIRS}
)
target_link_libraries(mel_band_roformer PUBLIC ${GGML_LIBRARIES} m)

# Executable
add_executable(mel_band_roformer_inference main.c)
target_link_libraries(mel_band_roformer_inference mel_band_roformer)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mel_band_roformer PRIVATE -Wall -Wextra -O3 -march=native)
    target_compile_options(mel_band_roformer_inference PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Install
install(TARGETS mel_band_roformer mel_band_roformer_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES mel_band_roformer.h DESTINATION include)
