#include "bench/EnvInfo.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

int main() {
    bench::EnvInputs gpuMachine;
    gpuMachine.cpuModel = "Test CPU @ 1 GHz";
    gpuMachine.hardwareConcurrency = 16;
    gpuMachine.os = "Linux 6.6";
    gpuMachine.memTotalKb = 8 * 1024 * 1024;
    gpuMachine.build.buildType = "Release";
    gpuMachine.build.sanitizer = "none";
    gpuMachine.build.compiler = "g++ test";
    gpuMachine.build.cxxFlags = "-O3 -DNDEBUG";
    gpuMachine.build.gitCommit = "abc1234";
    gpuMachine.build.cuda = true;
    gpuMachine.build.cudaFlags = "-O3";
    gpuMachine.gpu.available = true;
    gpuMachine.gpu.name = "Test GPU";
    gpuMachine.gpu.computeMajor = 8;
    gpuMachine.gpu.computeMinor = 9;
    gpuMachine.gpu.totalMemoryBytes = 6ull << 30;
    gpuMachine.gpu.multiProcessorCount = 20;
    gpuMachine.gpu.clockRateKHz = 2370000;
    gpuMachine.gpu.memoryClockRateKHz = 8001000;
    gpuMachine.gpu.memoryBusWidthBits = 96;
    gpuMachine.gpu.runtimeVersion = 12090;
    gpuMachine.gpu.driverVersion = 13000;
    gpuMachine.physicalCores = 8;
    gpuMachine.nvidiaDriver = "576.52";
    gpuMachine.command = {"delivery_sim", "config/ciudad.json", "--headless", "--bench", "out/r.csv"};
    gpuMachine.config = "config/ciudad.json";
    gpuMachine.reps = 10;
    gpuMachine.seed = 12345;
    gpuMachine.threads = {1, 2, 4};
    gpuMachine.apps = {"calib_copy", "calib_launch"};

    const auto gpuJson = nlohmann::json::parse(bench::envJson(gpuMachine));
    CHECK(gpuJson.at("cpu").at("model") == "Test CPU @ 1 GHz");
    CHECK(gpuJson.at("cpu").at("hardware_concurrency") == 16);
    CHECK(gpuJson.at("os") == "Linux 6.6");
    CHECK(gpuJson.at("build").at("type") == "Release");
    CHECK(gpuJson.at("build").at("sanitizer") == "none");
    CHECK(gpuJson.at("build").at("compiler") == "g++ test");
    CHECK(gpuJson.at("build").at("cxx_flags") == "-O3 -DNDEBUG");
    CHECK(gpuJson.at("build").at("git_commit") == "abc1234");
    CHECK(gpuJson.at("build").at("cuda") == true);
    CHECK(gpuJson.at("gpu").at("available") == true);
    CHECK(gpuJson.at("gpu").at("name") == "Test GPU");
    CHECK(gpuJson.at("gpu").at("compute_capability") == "8.9");
    CHECK(gpuJson.at("gpu").at("sm_count") == 20);
    CHECK(gpuJson.at("gpu").at("bus_width_bits") == 96);
    // 2 * 8001000 kHz * 96/8 B = 192.024 GB/s
    CHECK(gpuJson.at("gpu").at("peak_bandwidth_gbs").get<double>() > 192.0);
    CHECK(gpuJson.at("gpu").at("peak_bandwidth_gbs").get<double>() < 192.1);
    CHECK(gpuJson.at("cuda").at("runtime") == "12.9");
    CHECK(gpuJson.at("cuda").at("driver") == "576.52");
    CHECK(gpuJson.at("cuda").at("driver_max_cuda") == "13.0");
    CHECK(gpuJson.at("cpu").at("physical_cores") == 8);
    CHECK(gpuJson.at("run").at("config") == "config/ciudad.json");
    CHECK(gpuJson.at("run").at("command").size() == 5);
    CHECK(gpuJson.at("run").at("command")[0] == "delivery_sim");
    CHECK(gpuJson.at("run").at("seed") == 12345);
    CHECK(gpuJson.at("run").at("reps") == 10);
    CHECK(gpuJson.at("run").at("thread_sweep").size() == 3);


    // sanitizePath: nunca deja rutas absolutas personales en el JSON.
    CHECK(bench::sanitizePath("config/ciudad.json", "/home/user/repo") == "config/ciudad.json");
    CHECK(bench::sanitizePath("--bench", "/home/user/repo") == "--bench");
    CHECK(bench::sanitizePath("/home/user/repo/out/r.csv", "/home/user/repo") == "out/r.csv");
    CHECK(bench::sanitizePath("/home/user/repo", "/home/user/repo") == ".");
    CHECK(bench::sanitizePath("/tmp/other/r.csv", "/home/user/repo") == "r.csv");
    CHECK(bench::sanitizePath("/home/user/repository/x.csv", "/home/user/repo") == "x.csv");  // prefijo no es subcarpeta
    CHECK(bench::sanitizePath("../up.csv", "/home/user/repo") == "../up.csv");

    // Nucleos fisicos: pares (physical id, core id) unicos de /proc/cpuinfo.
    {
        std::istringstream cpuinfo(
            "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\ncpu cores\t: 2\n\n"
            "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 0\ncpu cores\t: 2\n\n"
            "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 1\ncpu cores\t: 2\n\n"
            "processor\t: 3\nphysical id\t: 1\ncore id\t\t: 1\ncpu cores\t: 2\n\n");
        CHECK(bench::countPhysicalCores(cpuinfo) == 3);
        std::istringstream withoutCoreIds("processor\t: 0\nmodel name\t: x\n");
        CHECK(bench::countPhysicalCores(withoutCoreIds) == 0);
    }
    // Sin GPU: el bloque existe, con available=false y el motivo; sin versiones CUDA.
    bench::EnvInputs cpuMachine = gpuMachine;
    cpuMachine.gpu = accel::GpuInfo();
    cpuMachine.gpu.message = "compilado sin CUDA";
    cpuMachine.build.cuda = false;
    const auto cpuJson = nlohmann::json::parse(bench::envJson(cpuMachine));
    CHECK(cpuJson.at("gpu").at("available") == false);
    CHECK(cpuJson.at("gpu").at("reason") == "compilado sin CUDA");
    CHECK(cpuJson.at("cuda").is_null());

    std::cout << "test_bench_env OK\n";
    return 0;
}
