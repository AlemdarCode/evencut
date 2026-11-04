cmake_minimum_required(VERSION 3.20)
project(evencut
    VERSION 1.0.0
    DESCRIPTION "Exact balanced partitioning of ordered workloads"
    LANGUAGES C CXX)

option(EVENCUT_BUILD_SHARED "Build the C ABI shared library" ON)
option(EVENCUT_BUILD_CLI "Build the evencut command-line tool" ON)
option(EVENCUT_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(EVENCUT_BUILD_BENCH "Build the benchmark" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  # executables and DLLs side by side, so Windows finds the DLL when tests run
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

include(GNUInstallDirs)

# Header-only C++ library ---------------------------------------------------------------
add_library(evencut_headers INTERFACE)
add_library(evencut::headers ALIAS evencut_headers)
target_include_directories(evencut_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(evencut_headers INTERFACE cxx_std_20)
# installed as evencut::headers, the same name the build tree uses
set_target_properties(evencut_headers PROPERTIES EXPORT_NAME headers)

if(MSVC)
  set(EVENCUT_WARNINGS /W4 /permissive-)
else()
  set(EVENCUT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
endif()
if(MINGW)
  # self-contained binaries: no libstdc++ / libgcc / winpthread DLLs needed at run time
  set(EVENCUT_STATIC_RUNTIME -static -static-libgcc -static-libstdc++)
endif()

# C ABI shared library --------------------------------------------------------------------
if(EVENCUT_BUILD_SHARED)
  add_library(evencut SHARED src/evencut_c.cpp)
  add_library(evencut::evencut ALIAS evencut)
  target_link_libraries(evencut PRIVATE evencut_headers)
  target_include_directories(evencut PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_definitions(evencut PRIVATE EVENCUT_BUILDING_DLL INTERFACE EVENCUT_USING_DLL)
  target_compile_options(evencut PRIVATE ${EVENCUT_WARNINGS})
  target_link_options(evencut PRIVATE ${EVENCUT_STATIC_RUNTIME})
  set_target_properties(evencut PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
      PDB_NAME evencut_c)
  if(WIN32)
    set_target_properties(evencut PROPERTIES PREFIX "")  # evencut.dll with MSVC and MinGW alike
  endif()
  if(NOT SKBUILD)
    set_target_properties(evencut PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  endif()
endif()

# Command-line tool ---------------------------------------------------------------------
if(EVENCUT_BUILD_CLI)
  add_executable(evencut_cli cli/evencut_cli.cpp)
  target_link_libraries(evencut_cli PRIVATE evencut_headers)
  target_compile_options(evencut_cli PRIVATE ${EVENCUT_WARNINGS})
  target_link_options(evencut_cli PRIVATE ${EVENCUT_STATIC_RUNTIME})
  set_target_properties(evencut_cli PROPERTIES OUTPUT_NAME evencut)
endif()

# Tests and benchmark -------------------------------------------------------------------
if(EVENCUT_BUILD_TESTS)
  enable_testing()
  add_executable(evencut_tests tests/cpp/test_evencut.cpp)
  target_link_libraries(evencut_tests PRIVATE evencut_headers)
  target_compile_options(evencut_tests PRIVATE ${EVENCUT_WARNINGS})
  target_link_options(evencut_tests PRIVATE ${EVENCUT_STATIC_RUNTIME})
  add_test(NAME evencut_tests COMMAND evencut_tests 500)
  if(EVENCUT_BUILD_SHARED)
    add_executable(evencut_c_api_test tests/c/test_c_api.c)
    target_link_libraries(evencut_c_api_test PRIVATE evencut)
    target_link_options(evencut_c_api_test PRIVATE ${EVENCUT_STATIC_RUNTIME})
    add_test(NAME evencut_c_api_test COMMAND evencut_c_api_test)
  endif()
  if(EVENCUT_BUILD_CLI)
    add_test(NAME evencut_cli_version COMMAND evencut_cli --version)
    # options a mode cannot honour must be rejected before any file is read
    add_test(NAME evencut_cli_rejects_tree_forbid COMMAND evencut_cli -k 2 --parents p.txt --forbid 1 w.txt)
    add_test(NAME evencut_cli_rejects_tree_secondary
             COMMAND evencut_cli -k 2 --parents p.txt --secondary m.txt --secondary-limit 5 w.txt)
    add_test(NAME evencut_cli_rejects_evaluate_forbid COMMAND evencut_cli --evaluate 4,7,10 --forbid 4 w.txt)
    add_test(NAME evencut_cli_rejects_secondary_without_limit COMMAND evencut_cli -k 2 --secondary m.txt w.txt)
    set_tests_properties(evencut_cli_rejects_tree_forbid evencut_cli_rejects_tree_secondary
                         evencut_cli_rejects_evaluate_forbid
                         PROPERTIES PASS_REGULAR_EXPRESSION "is not supported in")
    set_tests_properties(evencut_cli_rejects_secondary_without_limit
                         PROPERTIES PASS_REGULAR_EXPRESSION "needs --secondary-limit")
    # input that starts with a UTF-8 byte order mark (PowerShell pipes, Notepad)
    add_test(NAME evencut_cli_accepts_bom COMMAND evencut_cli -k 3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/bom_weights.txt)
    set_tests_properties(evencut_cli_accepts_bom PROPERTIES PASS_REGULAR_EXPRESSION "\"max_load\":18")
  endif()
endif()

if(EVENCUT_BUILD_BENCH)
  add_executable(evencut_bench bench/bench_evencut.cpp)
  target_link_libraries(evencut_bench PRIVATE evencut_headers)
  target_link_options(evencut_bench PRIVATE ${EVENCUT_STATIC_RUNTIME})
endif()

# Installation ----------------------------------------------------------------------------
if(SKBUILD)
  # Python wheel: only the shared library, next to the package's __init__.py
  install(TARGETS evencut LIBRARY DESTINATION evencut RUNTIME DESTINATION evencut)
else()
  include(CMakePackageConfigHelpers)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  set(EVENCUT_EXPORTS evencut_headers)
  if(EVENCUT_BUILD_SHARED)
    list(APPEND EVENCUT_EXPORTS evencut)
  endif()
  install(TARGETS ${EVENCUT_EXPORTS} EXPORT evencutTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  if(EVENCUT_BUILD_CLI)
    install(TARGETS evencut_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  install(EXPORT evencutTargets NAMESPACE evencut:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/evencut)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/evencutConfigVersion.cmake
      COMPATIBILITY SameMajorVersion)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/evencutConfig.cmake
      "include(\"\${CMAKE_CURRENT_LIST_DIR}/evencutTargets.cmake\")\n")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/evencutConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/evencutConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/evencut)
endif()
