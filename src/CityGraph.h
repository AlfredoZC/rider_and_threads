// El mapa de la ciudad como grafo dirigido, listo para consultar caminos.
//
// Config trae las calles como una lista plana, que sirve para dibujar pero no
// para preguntar "¿a dónde puedo ir desde este nodo?". Esta clase reorganiza
// esa lista en un índice de adyacencia, le calcula a cada arista su largo en
// metros y su tiempo de viaje, y traduce los ids de texto a índices enteros
// para que recorrer el grafo sea barato.
//
// Se construye UNA vez, antes de crear cualquier hilo, y después es
// completamente inmutable: todos los métodos son const y no existe ninguno que
// modifique. Por eso los couriers, el dispatcher y los hilos de estimación
// pueden consultarlo todos a la vez sin ningún candado. Un dato que nadie
// escribe no puede tener data races.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Config.h"

// Índice interno de un nodo. Entero y no string: Dijkstra consulta vecinos y
// distancias miles de veces por ruta, y con enteros eso es acceso directo a un
// vector en vez de un hash de string. Los ids de texto se traducen solo en la
// frontera del grafo.
using NodeIndex = std::uint32_t;

// "no es ningún nodo". Se usa cuando se consulta un id que no existe.
inline constexpr NodeIndex kNoNode = 0xFFFFFFFFu;

// Una calle recorrida en un sentido concreto. Una calle de doble sentido del
// config produce dos aristas; una de un solo sentido produce una, y por eso ir
// contramano no es que esté prohibido: es que no existe el camino.
struct Edge {
  NodeIndex to = kNoNode;
  double meters = 0.0;
  std::int64_t travelMs = 0;  // tiempo SIMULADO a speedKmh; timeScale no entra acá
};

class CityGraph {
 public:
  // Construye el grafo desde una Config ya validada por loadConfig().
  explicit CityGraph(const Config& cfg);

  std::size_t nodeCount() const { return nodes_.size(); }
  std::size_t edgeCount() const { return edgeCount_; }

  // Traducción en la frontera. Devuelve kNoNode si el id no existe.
  NodeIndex indexOf(const std::string& id) const;

  const std::string& idOf(NodeIndex i) const { return nodes_[i].id; }
  double latOf(NodeIndex i) const { return nodes_[i].lat; }
  double lonOf(NodeIndex i) const { return nodes_[i].lon; }

  // Las aristas que SALEN de i. Es la pregunta que hace Dijkstra en su bucle
  // interno.
  const std::vector<Edge>& edgesFrom(NodeIndex i) const { return adjacency_[i]; }

 private:
  struct Node {
    std::string id;
    double lat = 0.0;
    double lon = 0.0;
  };

  std::vector<Node> nodes_;                         // índice -> nodo
  std::unordered_map<std::string, NodeIndex> ids_;  // id de texto -> índice
  std::vector<std::vector<Edge>> adjacency_;        // índice -> aristas que salen
  std::size_t edgeCount_ = 0;
};
