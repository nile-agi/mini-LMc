cmake_minimum_required(VERSION 3.16)
project(mini-LMc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Werror)
endif()

add_library(minilmc_core STATIC
  src/platform.c
  src/ggml_types.c
  src/gguf.c
)
target_include_directories(minilmc_core PUBLIC include)

add_executable(minilmc src/main.c)
target_link_libraries(minilmc PRIVATE minilmc_core)

enable_testing()

add_executable(test_platform tests/test_platform.c)
target_link_libraries(test_platform PRIVATE minilmc_core)
add_test(NAME platform COMMAND test_platform)

add_executable(test_ggml_types tests/test_ggml_types.c)
target_link_libraries(test_ggml_types PRIVATE minilmc_core)
add_test(NAME ggml_types COMMAND test_ggml_types)

add_executable(test_gguf_loader tests/test_gguf_loader.c)
target_link_libraries(test_gguf_loader PRIVATE minilmc_core)
add_test(NAME gguf_loader COMMAND test_gguf_loader)