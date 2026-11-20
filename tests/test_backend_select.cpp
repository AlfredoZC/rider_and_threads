#include <cassert>
#include <iostream>
#include <string>

#include "accel/Backend.h"

// Seleccion de backend con una sonda falsa (sin GPU real).
namespace {

accel::GpuInfo goodProbe() {
    accel::GpuInfo i;
    i.available = true;
    i.name = "Fake GPU";
    i.computeMajor = 8;
    i.computeMinor = 9;
    i.totalMemoryBytes = static_cast<std::size_t>(6) * 1024 * 1024 * 1024;
    return i;
}

accel::GpuInfo badProbe() {
    accel::GpuInfo i;
    i.message = "CUDA: sin dispositivo";
    return i;
}

}  // namespace

int main() {
    bool parsed = false;
    assert(accel::parseBackendChoice("cpu", parsed) == accel::BackendChoice::cpu && parsed);
    assert(accel::parseBackendChoice("gpu", parsed) == accel::BackendChoice::gpu && parsed);
    assert(accel::parseBackendChoice("auto", parsed) == accel::BackendChoice::automatic && parsed);
    accel::parseBackendChoice("cuda", parsed);
    assert(!parsed);

    int probeCalls = 0;
    auto counting = [&probeCalls] {
        ++probeCalls;
        return goodProbe();
    };

    // cpu: no crea worker ni toca la sonda
    accel::BackendSelection cpu = accel::selectBackend(accel::BackendChoice::cpu, counting);
    assert(cpu.worker == nullptr && probeCalls == 0 && !cpu.failed);
    assert(cpu.line.find("CPU") != std::string::npos);
    assert(cpu.line.find("--backend cpu") != std::string::npos);

    // auto con GPU buena: worker saludable
    accel::BackendSelection good = accel::selectBackend(accel::BackendChoice::automatic, counting);
    assert(good.worker && good.worker->healthy() && probeCalls == 1 && !good.failed);
    assert(good.line.find("GPU") != std::string::npos);
    assert(good.line.find("Fake GPU") != std::string::npos);

    // auto sin GPU: respaldo CPU con el motivo
    accel::BackendSelection bad = accel::selectBackend(accel::BackendChoice::automatic, badProbe);
    assert(bad.worker == nullptr && !bad.failed);
    assert(bad.line.find("CPU") != std::string::npos);
    assert(bad.line.find("sin dispositivo") != std::string::npos);

    // gpu forzado sin GPU: fallo explicito, no respaldo silencioso
    accel::BackendSelection forced = accel::selectBackend(accel::BackendChoice::gpu, badProbe);
    assert(forced.worker == nullptr && forced.failed);
    assert(forced.line.find("sin dispositivo") != std::string::npos);

    std::cout << "test_backend_select OK\n";
    return 0;
}
