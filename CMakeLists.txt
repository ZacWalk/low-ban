cmake_minimum_required(VERSION 3.21)

project(low-ban LANGUAGES CXX VERSION 1.0.0)

if(NOT WIN32)
    message(FATAL_ERROR "low-ban is a Win32 SDK application and only builds on Windows.")
endif()
if(NOT MSVC)
    message(FATAL_ERROR "low-ban requires MSVC; no other toolchain is supported.")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "low-ban is 64-bit only; configure with an x64 toolchain.")
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Develop against a sibling libdesktop checkout when one exists; a fresh clone
# uses the first published revision that contains the neutral pf window layer.
if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_LIBDESKTOP
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libdesktop/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_LIBDESKTOP "${CMAKE_CURRENT_SOURCE_DIR}/../libdesktop"
        CACHE PATH "Local libdesktop checkout to build against")
endif()

include(FetchContent)
FetchContent_Declare(libdesktop
    GIT_REPOSITORY https://github.com/ZacWalk/libdesktop.git
    GIT_TAG        31aae322af6e0696a9597726705b1121bd31580b
)
FetchContent_MakeAvailable(libdesktop)

platform_add_app(low-ban
    SOURCES
        src/baseline.cpp
        src/capture.cpp
        src/codec.cpp
        src/draw.cpp
        src/engine.cpp
        src/evaluate.cpp
        src/face.cpp
        src/face-model.cpp
        src/image.cpp
        src/landmarks.cpp
        src/main.cpp
        src/selftest.cpp
        src/session.cpp
        src/warp.cpp
    ICON        src/app.ico
    DESCRIPTION "low-ban - low bandwidth video experiment"
    OUTPUT_NAME low-ban-64
)

target_sources(low-ban PRIVATE
    src/baseline.h
    src/capture.h
    src/codec.h
    src/draw.h
    src/engine.h
    src/face.h
    src/image.h
    src/landmarks.h
    src/session.h
    src/warp.h
)

target_compile_features(low-ban PRIVATE cxx_std_20)
target_include_directories(low-ban PRIVATE src)
target_compile_definitions(low-ban PRIVATE WIN32 _WINDOWS NOMINMAX UNICODE _UNICODE)
target_compile_options(low-ban PRIVATE /W3 /MP /permissive-)
target_link_libraries(low-ban PRIVATE windowscodecs)

set_target_properties(low-ban PROPERTIES
    DEBUG_POSTFIX d
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exe"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/exe"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/exe"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(low_ban_tests tests/smoke.cpp)
    target_compile_features(low_ban_tests PRIVATE cxx_std_17)
    add_test(NAME low_ban_unit COMMAND low_ban_tests)
    set_tests_properties(low_ban_unit PROPERTIES TIMEOUT 30 LABELS low-ban)

    add_test(NAME low_ban_selftest COMMAND low-ban /test)
    set_tests_properties(low_ban_selftest PROPERTIES
        TIMEOUT 120 LABELS low-ban WORKING_DIRECTORY "$<TARGET_FILE_DIR:low-ban>")

    add_test(NAME low_ban_manifest COMMAND "${CMAKE_COMMAND}"
        "-DEXE=$<TARGET_FILE:low-ban>" -P "${PROJECT_SOURCE_DIR}/tests/manifest.cmake")
    set_tests_properties(low_ban_manifest PROPERTIES TIMEOUT 30 LABELS low-ban)
endif()
