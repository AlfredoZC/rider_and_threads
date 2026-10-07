#include "Config.h"

#include <fstream>
#include <limits>
#include <unordered_set>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace {

// Cada helper recibe la ruta "lógica" de la propiedad (p. ej. "fleet.couriers")
// para que el mensaje de error diga exactamente qué está mal.

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    throw ConfigError("'" + where + "': " + what);
}

json field(const json& obj, const std::string& key, const std::string& where) {
    if (!obj.is_object()) {
        fail(where, "debe ser un objeto");
    }
    const auto it = obj.find(key);
    if (it == obj.end()) {
        fail(where.empty() ? key : where + "." + key, "falta la propiedad");
    }
    return *it;
}

std::string join(const std::string& where, const std::string& key) {
    return where.empty() ? key : where + "." + key;
}

std::string getString(const json& obj, const std::string& key, const std::string& where) {
    const json v = field(obj, key, where);
    if (!v.is_string()) {
        fail(join(where, key), "debe ser un texto");
    }
    return v.get<std::string>();
}

bool getBool(const json& obj, const std::string& key, const std::string& where) {
    const json v = field(obj, key, where);
    if (!v.is_boolean()) {
        fail(join(where, key), "debe ser true o false");
    }
    return v.get<bool>();
}

double getNumber(const json& obj, const std::string& key, const std::string& where) {
    const json v = field(obj, key, where);
    if (!v.is_number()) {
        fail(join(where, key), "debe ser un número");
    }
    return v.get<double>();
}

std::int64_t getInteger(const json& obj, const std::string& key, const std::string& where,
                        std::int64_t min = std::numeric_limits<std::int64_t>::min()) {
    const json v = field(obj, key, where);
    if (!v.is_number_integer()) {
        fail(join(where, key), "debe ser un número entero");
    }
    const std::int64_t value = v.get<std::int64_t>();
    if (value < min) {
        fail(join(where, key), "debe ser >= " + std::to_string(min));
    }
    return value;
}

int getInt(const json& obj, const std::string& key, const std::string& where, std::int64_t min) {
    const std::int64_t value = getInteger(obj, key, where, min);
    if (value > std::numeric_limits<int>::max()) {
        fail(join(where, key), "es demasiado grande");
    }
    return static_cast<int>(value);
}

json getArray(const json& obj, const std::string& key, const std::string& where) {
    const json v = field(obj, key, where);
    if (!v.is_array()) {
        fail(join(where, key), "debe ser un arreglo");
    }
    return v;
}

void parseMap(const json& root, const std::filesystem::path& configDir, Config& cfg) {
    const json map = field(root, "map", "");
    cfg.imagePath = configDir / getString(map, "image", "map");
    cfg.attribution = getString(map, "attribution", "map");

    const json b = field(map, "bounds", "map");
    cfg.bounds.north = getNumber(b, "north", "map.bounds");
    cfg.bounds.south = getNumber(b, "south", "map.bounds");
    cfg.bounds.west = getNumber(b, "west", "map.bounds");
    cfg.bounds.east = getNumber(b, "east", "map.bounds");
    if (!(cfg.bounds.north > cfg.bounds.south)) {
        fail("map.bounds", "north debe ser mayor que south");
    }
    if (!(cfg.bounds.east > cfg.bounds.west)) {
        fail("map.bounds", "east debe ser mayor que west");
    }
}

void parseNodes(const json& root, Config& cfg, std::unordered_set<std::string>& ids) {
    const json nodes = getArray(root, "nodes", "");
    if (nodes.empty()) {
        fail("nodes", "debe tener al menos un nodo");
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const std::string where = "nodes[" + std::to_string(i) + "]";
        NodeConfig n;
        n.id = getString(nodes[i], "id", where);
        n.lat = getNumber(nodes[i], "lat", where);
        n.lon = getNumber(nodes[i], "lon", where);
        if (!ids.insert(n.id).second) {
            fail(where + ".id", "el id '" + n.id + "' está repetido");
        }
        cfg.nodes.push_back(std::move(n));
    }
}

void requireNode(const std::unordered_set<std::string>& nodeIds, const std::string& id,
                 const std::string& where) {
    if (nodeIds.count(id) == 0) {
        fail(where, "refiere al nodo inexistente '" + id + "'");
    }
}

void parseStreets(const json& root, Config& cfg, const std::unordered_set<std::string>& nodeIds) {
    const json streets = getArray(root, "streets", "");
    std::unordered_set<std::string> ids;
    for (std::size_t i = 0; i < streets.size(); ++i) {
        const std::string where = "streets[" + std::to_string(i) + "]";
        StreetConfig s;
        s.id = getString(streets[i], "id", where);
        s.from = getString(streets[i], "from", where);
        s.to = getString(streets[i], "to", where);
        s.oneWay = getBool(streets[i], "oneWay", where);
        requireNode(nodeIds, s.from, where + ".from");
        requireNode(nodeIds, s.to, where + ".to");
        if (!ids.insert(s.id).second) {
            fail(where + ".id", "el id '" + s.id + "' está repetido");
        }
        cfg.streets.push_back(std::move(s));
    }
}

void parseRestaurants(const json& root, Config& cfg, const std::unordered_set<std::string>& nodeIds) {
    const json restaurants = getArray(root, "restaurants", "");
    if (restaurants.empty()) {
        fail("restaurants", "debe tener al menos un restaurante");
    }
    std::unordered_set<std::string> ids;
    for (std::size_t i = 0; i < restaurants.size(); ++i) {
        const std::string where = "restaurants[" + std::to_string(i) + "]";
        const json& r = restaurants[i];
        RestaurantConfig rc;
        rc.id = getString(r, "id", where);
        rc.name = getString(r, "name", where);
        rc.node = getString(r, "node", where);
        rc.pickupSlots = getInt(r, "pickupSlots", where, 1);
        requireNode(nodeIds, rc.node, where + ".node");

        const json prep = getArray(r, "prepTimeMs", where);
        if (prep.size() != 2 || !prep[0].is_number_integer() || !prep[1].is_number_integer()) {
            fail(where + ".prepTimeMs", "debe ser un arreglo [min, max] de enteros");
        }
        rc.prepMinMs = prep[0].get<std::int64_t>();
        rc.prepMaxMs = prep[1].get<std::int64_t>();
        if (rc.prepMinMs < 0 || rc.prepMinMs > rc.prepMaxMs) {
            fail(where + ".prepTimeMs", "debe cumplir 0 <= min <= max");
        }
        if (!ids.insert(rc.id).second) {
            fail(where + ".id", "el id '" + rc.id + "' está repetido");
        }
        cfg.restaurants.push_back(std::move(rc));
    }
}

void parseSettings(const json& root, Config& cfg, const std::unordered_set<std::string>& nodeIds) {
    const json fleet = field(root, "fleet", "");
    cfg.couriers = getInt(fleet, "couriers", "fleet", 1);
    cfg.bagCapacity = getInt(fleet, "bagCapacity", "fleet", 1);
    cfg.speedKmh = getNumber(fleet, "speedKmh", "fleet");
    if (!(cfg.speedKmh > 0.0)) {
        fail("fleet.speedKmh", "debe ser mayor que 0");
    }
    cfg.startNode = getString(fleet, "startNode", "fleet");
    requireNode(nodeIds, cfg.startNode, "fleet.startNode");

    const json orders = field(root, "orders", "");
    cfg.meanIntervalMs = getInteger(orders, "meanIntervalMs", "orders", 1);
    cfg.burstMax = getInt(orders, "burstMax", "orders", 1);
    cfg.maxPending = getInt(orders, "maxPending", "orders", 0);
    // Cualquier entero es una semilla válida; se reduce a 32 bits para std::mt19937
    cfg.seed = static_cast<std::uint32_t>(getInteger(orders, "seed", "orders"));

    const json dispatch = field(root, "dispatch", "");
    cfg.quoteTimeoutMs = getInteger(dispatch, "quoteTimeoutMs", "dispatch", 0);
    cfg.acceptTimeoutMs = getInteger(dispatch, "acceptTimeoutMs", "dispatch", 0);

    const json incidents = field(root, "incidents", "");
    cfg.breakdownProbability = getNumber(incidents, "breakdownProbability", "incidents");
    if (cfg.breakdownProbability < 0.0 || cfg.breakdownProbability > 1.0) {
        fail("incidents.breakdownProbability", "debe estar entre 0 y 1");
    }

    const json simulation = field(root, "simulation", "");
    cfg.durationS = getInteger(simulation, "durationS", "simulation", 0);
    cfg.timeScale = getNumber(simulation, "timeScale", "simulation");
    if (!(cfg.timeScale > 0.0)) {
        fail("simulation.timeScale", "debe ser mayor que 0");
    }
}

}  // namespace

Config loadConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw ConfigError("no se puede leer el archivo '" + path + "'");
    }

    json root;
    try {
        root = json::parse(in);
    } catch (const json::parse_error& e) {
        throw ConfigError("JSON mal formado en '" + path + "': " + e.what());
    }
    if (!root.is_object()) {
        throw ConfigError("'" + path + "' debe contener un objeto JSON");
    }

    Config cfg;
    std::unordered_set<std::string> nodeIds;
    try {
        parseMap(root, std::filesystem::path(path).parent_path(), cfg);
        parseNodes(root, cfg, nodeIds);
        parseStreets(root, cfg, nodeIds);
        parseRestaurants(root, cfg, nodeIds);
        parseSettings(root, cfg, nodeIds);
    } catch (const ConfigError& e) {
        throw ConfigError("configuración inválida en '" + path + "': " + e.what());
    }
    return cfg;
}
