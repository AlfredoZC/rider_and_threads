// Configuración de la simulación (sección 5 del brief).
// Se carga y valida por completo en main ANTES de crear cualquier hilo; después
// es de solo lectura, así que compartirla entre hilos no puede producir data races.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

struct Bounds {
    double north = 0.0;
    double south = 0.0;
    double west = 0.0;
    double east = 0.0;
};

struct NodeConfig {
    std::string id;
    double lat = 0.0;
    double lon = 0.0;
};

struct StreetConfig {
    std::string id;
    std::string from;
    std::string to;
    bool oneWay = false;
};

struct RestaurantConfig {
    std::string id;
    std::string name;
    std::string node;
    int pickupSlots = 1;
    std::int64_t prepMinMs = 0;
    std::int64_t prepMaxMs = 0;
};

struct Config {
    // map
    std::filesystem::path imagePath;   // ya resuelta relativa al archivo de config
    std::string attribution;
    Bounds bounds;

    std::vector<NodeConfig> nodes;
    std::vector<StreetConfig> streets;
    std::vector<RestaurantConfig> restaurants;

    // fleet
    int couriers = 0;
    int bagCapacity = 0;
    double speedKmh = 0.0;
    std::string startNode;

    // orders
    std::int64_t meanIntervalMs = 0;
    int burstMax = 0;
    int maxPending = 0;
    std::uint32_t seed = 0;

    // dispatch
    std::int64_t quoteTimeoutMs = 0;    // tiempo REAL
    std::int64_t acceptTimeoutMs = 0;   // tiempo SIMULADO

    // incidents
    double breakdownProbability = 0.0;

    // simulation
    std::int64_t durationS = 0;         // 0 = hasta recibir una señal
    double timeScale = 1.0;
};

// Cualquier problema con la configuración: el mensaje ya es legible para el usuario.
class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Lee y valida el archivo. Lanza ConfigError si el archivo no se puede leer, el JSON
// está mal formado, falta una propiedad, un tipo o rango es incorrecto, o una calle /
// restaurante / startNode refiere a un nodo inexistente.
Config loadConfig(const std::string& path);
