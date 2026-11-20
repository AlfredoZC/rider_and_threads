#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bench/BenchApp.h"

namespace bench {

using AppFactory = std::function<std::unique_ptr<BenchApp>()>;

// Registro de aplicaciones medibles. Cada aplicacion vive en su archivo de src/bench/ y se anota con
// BENCH_REGISTER_APP: no hay lista central que editar. Los registros ocurren al arrancar el
// programa, antes de main y desde un solo hilo.
class BenchRegistry {
public:
    static BenchRegistry& instance();
    void add(const std::string& name, AppFactory factory);
    std::vector<std::string> names() const;  // ordenados
    std::unique_ptr<BenchApp> create(const std::string& name) const;  // nullptr si no existe

private:
    std::vector<std::pair<std::string, AppFactory>> entries_;
};

struct AppRegistrar {
    AppRegistrar(const std::string& name, AppFactory factory) { BenchRegistry::instance().add(name, std::move(factory)); }
};

// Instancias de las aplicaciones pedidas (vacio = todas). Un nombre desconocido se omite: quien
// llama compara los tamanos si necesita avisar.
std::vector<std::unique_ptr<BenchApp>> makeApps(const std::vector<std::string>& names);

}  // namespace bench

#define BENCH_PP_CAT2(a, b) a##b
#define BENCH_PP_CAT(a, b) BENCH_PP_CAT2(a, b)
#define BENCH_REGISTER_APP(NAME, TYPE) \
    static const ::bench::AppRegistrar BENCH_PP_CAT(benchRegistrar_, __LINE__)(NAME, [] { return std::make_unique<TYPE>(); })
