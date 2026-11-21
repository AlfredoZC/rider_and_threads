#include "bench/BuildInfo.h"

#if __has_include("BuildGit.h")
#include "BuildGit.h"
#endif
#if __has_include("BuildInfoGenerated.h")
#include "BuildInfoGenerated.h"
#endif

namespace bench {

BuildInfo currentBuildInfo() {
    BuildInfo info;
#ifdef BENCH_BUILD_TYPE
    info.buildType = BENCH_BUILD_TYPE;
#endif
#ifdef BENCH_SANITIZER
    info.sanitizer = BENCH_SANITIZER;
#endif
#ifdef BENCH_CXX_FLAGS
    info.cxxFlags = BENCH_CXX_FLAGS;
#endif
#ifdef BENCH_CUDA_FLAGS
    info.cudaFlags = BENCH_CUDA_FLAGS;
#endif
#ifdef BENCH_GIT_COMMIT
    info.gitCommit = BENCH_GIT_COMMIT;
#endif
#ifdef USE_CUDA
    info.cuda = true;
#endif
#ifdef __OPTIMIZE__
    info.optimized = true;
#else
    info.optimized = false;
#endif
#if defined(__clang__)
    info.compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    info.compiler = std::string("gcc ") + __VERSION__;
#else
    info.compiler = "unknown";
#endif

    // Los sanitizers se detectan tambien por el propio compilador, por si CMake no avisa.
#if defined(__SANITIZE_THREAD__)
    info.sanitizer = "thread";
#elif defined(__SANITIZE_ADDRESS__)
    info.sanitizer = "address,undefined";
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
    info.sanitizer = "thread";
#elif __has_feature(address_sanitizer)
    info.sanitizer = "address,undefined";
#endif
#endif
    return info;
}

std::string benchRefusalReason(const BuildInfo& info) {
    const std::string prefix = "Never time a Debug or sanitizer build: ";
    if (info.sanitizer != "none") {
        return prefix + "este binario esta compilado con sanitizers (" + info.sanitizer +
               "). Compila con -DCMAKE_BUILD_TYPE=Release y sin ENABLE_TSAN/ENABLE_ASAN.";
    }
    if (info.buildType != "Release") {
        return prefix + "el tipo de build es '" + info.buildType +
               "', no Release. Compila con -DCMAKE_BUILD_TYPE=Release.";
    }
    if (!info.optimized) {
        return prefix + "el compilador no optimizo este binario (sin -O1/-O2/-O3, p. ej. -O0 en los flags de Release).";
    }
    // Simbolos de depuracion de dispositivo (-G / --device-debug) desactivan las optimizaciones de nvcc.
    std::size_t tokenStart = 0;
    const std::string& flags = info.cudaFlags;
    while (tokenStart < flags.size()) {
        const std::size_t tokenEnd = flags.find(' ', tokenStart);
        const std::string token = flags.substr(tokenStart, tokenEnd == std::string::npos ? std::string::npos : tokenEnd - tokenStart);
        if (token == "-G" || token == "--device-debug") {
            return prefix + "los flags de CUDA incluyen '" + token + "' (depuracion de dispositivo).";
        }
        if (tokenEnd == std::string::npos) {
            break;
        }
        tokenStart = tokenEnd + 1;
    }
    return "";
}

}  // namespace bench
