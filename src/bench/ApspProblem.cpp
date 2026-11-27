#include "bench/ApspProblem.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <new>

#include "Config.h"
#include "accel/TravelTimeTable.h"
#include "bench/BenchGraph.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

std::vector<std::size_t> apspDefaultSizes() { return {25, 100, 400, 1000, 2000, 3741}; }

void ApspProblem::clear() {
    nodeCount = 0;
    fromCity = false;
    graph.reset();
    for (std::vector<std::int32_t>* cells : {&initialCells, &referenceCells, &resultCells}) {
        cells->clear();
        cells->shrink_to_fit();
    }
}

std::string prepareApspProblem(const std::string& appName, std::size_t size, std::uint32_t seed,
                               const BenchContext& context, ApspReference reference, ApspProblem& problem) {
    problem.clear();  // un prepare que falla no deja nada reservado
    if (size == 0 || size > kApspMaxSize) {
        return appName + ": tamano fuera de rango (1 .. " + std::to_string(kApspMaxSize) + " nodos)";
    }
    try {
        const Config region = cityRegionOrSynthetic(context.config, size, seed, &problem.fromCity);
        problem.graph = std::make_unique<CityGraph>(region);
        problem.nodeCount = problem.graph->nodeCount();
        problem.initialCells = accel::buildInitialCells(problem.nodeCount, accel::streetsOf(*problem.graph));
        if (reference == ApspReference::Dijkstra) {
            accel::dijkstraFromEveryNode(*problem.graph, problem.referenceCells);
        } else {
            problem.referenceCells = problem.initialCells;
            accel::floydWarshallSingleThread(problem.referenceCells, problem.nodeCount);
        }
        problem.resultCells.assign(problem.referenceCells.size(), 0);
    } catch (const std::bad_alloc&) {
        problem.clear();
        return appName + ": sin memoria en el host para la tabla de " + std::to_string(size) + " nodos";
    } catch (const std::exception& caught) {
        problem.clear();
        return appName + ": " + caught.what();
    }
    return "";
}

RunResult failedRun(const std::string& message) {
    RunResult runResult;
    runResult.ok = false;
    runResult.error = message;
    return runResult;
}

void markCorrectness(const ApspProblem& problem, RunResult& runResult) {
    runResult.correct = problem.resultCells == problem.referenceCells;
}

RunResult runGpuFloyd(ApspProblem& problem, GpuFloydFunction gpuFunction, Version version, const char* label) {
    RunResult runResult;
    accel::GpuFloydTiming gpuTiming;
    try {
        problem.resultCells.assign(problem.resultCells.size(), 0);  // fuera de la region medida
        const auto startTime = Clock::now();
        gpuFunction(problem.initialCells, problem.nodeCount, problem.resultCells, &gpuTiming);
        const auto stopTime = Clock::now();
        if (version == Version::GpuKernel) {
            runResult.timing.ms = gpuTiming.kernelMs;
            runResult.timing.kernelMs = gpuTiming.kernelMs;
        } else {
            runResult.timing.ms = elapsedMs(startTime, stopTime);
            runResult.timing.h2dMs = gpuTiming.hostToDeviceMs;
            runResult.timing.kernelMs = gpuTiming.kernelMs;
            runResult.timing.d2hMs = gpuTiming.deviceToHostMs;
        }
    } catch (const std::exception& caught) {
        return failedRun(std::string(label) + " " + versionName(version) + ": " + caught.what());
    }
    markCorrectness(problem, runResult);
    return runResult;
}

std::string describeRelaxations(const SummaryStats& stats) {
    const double nodes = static_cast<double>(stats.size);
    const double relaxationsPerSecond = stats.medianMs > 0.0 ? nodes * nodes * nodes / (stats.medianMs * 1e-3) : 0.0;
    char description[160];
    std::snprintf(description, sizeof description, "%.2f G relaxations/s (n^3 min-plus updates)", relaxationsPerSecond / 1e9);
    return description;
}

std::string describeCells(const SummaryStats& stats) {
    const double nodes = static_cast<double>(stats.size);
    const double cellsPerSecond = stats.medianMs > 0.0 ? nodes * nodes / (stats.medianMs * 1e-3) : 0.0;
    char description[160];
    std::snprintf(description, sizeof description, "%.2f M table cells/s (n^2 cells)", cellsPerSecond / 1e6);
    return description;
}

}  // namespace bench
