#include "bench/EnvInfo.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <utility>
#include <thread>

#include <nlohmann/json.hpp>
#include <sys/utsname.h>

namespace bench {

namespace {

// "12090" -> "12.9" (formato de cudaRuntimeGetVersion: 1000 * major + 10 * minor).
std::string cudaVersionText(int versionCode) {
    if (versionCode <= 0) {
        return "unknown";
    }
    return std::to_string(versionCode / 1000) + "." + std::to_string((versionCode % 1000) / 10);
}

std::string readCpuModel() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::size_t start = colon + 1;
                while (start < line.size() && line[start] == ' ') {
                    ++start;
                }
                return line.substr(start);
            }
        }
    }
    return "unknown";
}

// Version real del driver NVIDIA: /proc/driver/nvidia/version (Linux nativo) o `nvidia-smi` (WSL).
// cudaDriverGetVersion NO sirve: devuelve la version maxima de CUDA que el driver soporta.
// Solo se aceptan digitos y puntos (el valor va al JSON).
bool plausibleVersion(const std::string& text) {
    if (text.empty() || text.size() > 24) {
        return false;
    }
    for (char character : text) {
        if (!((character >= '0' && character <= '9') || character == '.')) {
            return false;
        }
    }
    return true;
}

std::string readNvidiaDriverVersion() {
    {
        std::ifstream file("/proc/driver/nvidia/version");
        std::string word;
        bool next = false;
        while (file >> word) {
            if (next) {
                if (plausibleVersion(word)) {
                    return word;
                }
                break;
            }
            next = (word == "Module");
        }
    }
    std::string driverText;
    if (FILE* pipe = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null", "r")) {
        char line[64];
        if (std::fgets(line, sizeof line, pipe) != nullptr) {
            driverText = line;
        }
        pclose(pipe);
    }
    while (!driverText.empty() && (driverText.back() == '\n' || driverText.back() == '\r' || driverText.back() == ' ')) {
        driverText.pop_back();
    }
    return plausibleVersion(driverText) ? driverText : "unknown";
}

std::size_t readMemTotalKb() {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::size_t value = 0;
    std::string unit;
    while (file >> key >> value >> unit) {
        if (key == "MemTotal:") {
            return value;
        }
    }
    return 0;
}

}  // namespace

std::string sanitizePath(const std::string& argument, const std::string& workingDirectory) {
    if (argument.empty() || argument.front() != '/') {
        return argument;
    }
    if (argument == workingDirectory) {
        return ".";
    }
    if (!workingDirectory.empty() && argument.size() > workingDirectory.size() + 1 && argument.compare(0, workingDirectory.size(), workingDirectory) == 0 && argument[workingDirectory.size()] == '/') {
        return argument.substr(workingDirectory.size() + 1);
    }
    std::string trimmed = argument;
    while (trimmed.size() > 1 && trimmed.back() == '/') {
        trimmed.pop_back();
    }
    const std::size_t slash = trimmed.find_last_of('/');
    return trimmed.substr(slash + 1);
}

unsigned countPhysicalCores(std::istream& cpuinfo) {
    std::set<std::pair<std::string, std::string>> cores;
    std::string line;
    std::string physical = "0";
    while (std::getline(cpuinfo, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, colon);
        while (!key.empty() && (key.back() == '\t' || key.back() == ' ')) {
            key.pop_back();
        }
        std::string fieldValue = line.substr(colon + 1);
        while (!fieldValue.empty() && fieldValue.front() == ' ') {
            fieldValue.erase(fieldValue.begin());
        }
        if (key == "physical id") {
            physical = fieldValue;
        } else if (key == "core id") {
            cores.insert({physical, fieldValue});
        }
    }
    return static_cast<unsigned>(cores.size());
}

EnvInputs collectEnv() {
    EnvInputs environment;
    environment.cpuModel = readCpuModel();
    environment.hardwareConcurrency = std::thread::hardware_concurrency();
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        environment.physicalCores = countPhysicalCores(cpuinfo);
    }
    environment.memTotalKb = readMemTotalKb();
    struct utsname systemName;
    if (uname(&systemName) == 0) {
        environment.os = std::string(systemName.sysname) + " " + systemName.release;
    }
    environment.build = currentBuildInfo();
    environment.gpu = accel::gpuInfo();
    if (environment.gpu.available) {
        environment.nvidiaDriver = readNvidiaDriverVersion();
    }
    return environment;
}

std::string envJson(const EnvInputs& inputs) {
    using nlohmann::json;
    json document;
    document["schema"] = 1;
    document["cpu"] = {{"model", inputs.cpuModel}, {"hardware_concurrency", inputs.hardwareConcurrency},
                {"physical_cores", inputs.physicalCores}};
    document["memory_total_kb"] = inputs.memTotalKb;
    document["os"] = inputs.os;
    document["build"] = {{"type", inputs.build.buildType},   {"sanitizer", inputs.build.sanitizer},
                  {"compiler", inputs.build.compiler}, {"cxx_flags", inputs.build.cxxFlags},
                  {"cuda_flags", inputs.build.cudaFlags}, {"git_commit", inputs.build.gitCommit},
                  {"cuda", inputs.build.cuda}};
    json gpu = {{"available", inputs.gpu.available}};
    if (inputs.gpu.available) {
        gpu["name"] = inputs.gpu.name;
        gpu["compute_capability"] = std::to_string(inputs.gpu.computeMajor) + "." + std::to_string(inputs.gpu.computeMinor);
        gpu["sm_count"] = inputs.gpu.multiProcessorCount;
        gpu["memory_bytes"] = inputs.gpu.totalMemoryBytes;
        gpu["sm_clock_khz"] = inputs.gpu.clockRateKHz;
        gpu["memory_clock_khz"] = inputs.gpu.memoryClockRateKHz;
        gpu["bus_width_bits"] = inputs.gpu.memoryBusWidthBits;
        gpu["peak_bandwidth_gbs"] = accel::peakBandwidthGBs(inputs.gpu);
    } else {
        gpu["reason"] = inputs.gpu.message;
    }
    document["gpu"] = gpu;
    if (inputs.gpu.available) {
        document["cuda"] = {{"runtime", cudaVersionText(inputs.gpu.runtimeVersion)},
                     {"driver", inputs.nvidiaDriver},
                     {"driver_max_cuda", cudaVersionText(inputs.gpu.driverVersion)}};
    } else {
        document["cuda"] = nullptr;
    }
    document["run"] = {{"seed", inputs.seed}, {"reps", inputs.reps}, {"thread_sweep", inputs.threads}, {"apps", inputs.apps},
                {"command", inputs.command}, {"config", inputs.config}};
    return document.dump(2) + "\n";
}

}  // namespace bench
