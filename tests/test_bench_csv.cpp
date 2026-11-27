#include "bench/BenchCsv.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

int main() {
    CHECK(bench::csvHeader() ==
          "application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms");

    bench::Row cpuRow;
    cpuRow.application = "calib_copy";
    cpuRow.version = "cpuN";
    cpuRow.size = 4096;
    cpuRow.rep = 3;
    cpuRow.ms = 1.5;
    cpuRow.correct = true;
    cpuRow.threads = 4;
    cpuRow.seed = 12345;
    CHECK(bench::csvRow(cpuRow) == "calib_copy,cpuN,4096,3,1.500000,yes,4,12345,,,");

    bench::Row gpuRow = cpuRow;
    gpuRow.version = "gpu_e2e";
    gpuRow.correct = false;
    gpuRow.threads = 1;
    gpuRow.h2dMs = 0.25;
    gpuRow.kernelMs = 0.125;
    gpuRow.d2hMs = 0.5;
    CHECK(bench::csvRow(gpuRow) ==
          "calib_copy,gpu_e2e,4096,3,1.500000,no,1,12345,0.250000,0.125000,0.500000");

    // Escape: comas, comillas y saltos de linea se citan (aunque el codigo solo escribe identificadores).
    CHECK(bench::csvEscape("plain") == "plain");
    CHECK(bench::csvEscape("a,b") == "\"a,b\"");
    CHECK(bench::csvEscape("say \"hi\"") == "\"say \"\"hi\"\"\"");
    CHECK(bench::csvEscape("line\nbreak") == "\"line\nbreak\"");
    CHECK(bench::csvEscape("") == "");

    // Un nombre de aplicacion con coma no rompe el numero de columnas.
    bench::Row oddNameRow = cpuRow;
    oddNameRow.application = "a,b";
    CHECK(bench::csvRow(oddNameRow).rfind("\"a,b\",cpuN,", 0) == 0);


    // openOutputFile: exclusivo sin force, trunca con force y nunca sigue un enlace simbolico.
    {
        char directoryTemplate[] = "/tmp/bench_csv_testXXXXXX";
        char* createdDirectory = mkdtemp(directoryTemplate);
        CHECK(createdDirectory != nullptr);
        const std::string directory = createdDirectory;
        const std::string outputPath = directory + "/a.out";
        std::string errorMessage;
        std::FILE* handle = bench::openOutputFile(outputPath, false, errorMessage);
        CHECK(handle != nullptr && errorMessage.empty());
        std::fputs("viejo\n", handle);
        std::fclose(handle);
        // existe: sin force falla y no toca el contenido
        CHECK(bench::openOutputFile(outputPath, false, errorMessage) == nullptr && !errorMessage.empty());
        // con force trunca
        handle = bench::openOutputFile(outputPath, true, errorMessage);
        CHECK(handle != nullptr);
        std::fclose(handle);
        CHECK(std::filesystem::file_size(outputPath) == 0);
        // enlace simbolico: ni con force se sigue
        const std::string target = directory + "/target.txt";
        { std::ofstream targetFile(target); targetFile << "intacto"; }
        const std::string linkPath = directory + "/link.csv";
        CHECK(symlink(target.c_str(), linkPath.c_str()) == 0);
        CHECK(bench::openOutputFile(linkPath, true, errorMessage) == nullptr);
        CHECK(bench::openOutputFile(linkPath, false, errorMessage) == nullptr);
        CHECK(std::filesystem::file_size(target) == 7);
        std::filesystem::remove_all(directory);
    }
    std::cout << "test_bench_csv OK\n";
    return 0;
}
