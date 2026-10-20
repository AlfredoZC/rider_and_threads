// El resumen final del §6: un unico objeto JSON, ultima linea de stdout.
//
// El enunciado exige que sus numeros sean exactos y que concuerden con el log:
// created = delivered + rejected + pending, los valores de deliveredPerCourier suman
// delivered, y cada contador es igual a la cantidad de lineas que coinciden en el log.
// Esa ultima condicion implica que alguien lo va a leer con un script, y para eso esta
// tools/validate_log.py.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CourierActor.h"
#include "OrderBook.h"

// Arma la linea. breakdowns se pasa aparte porque todavia no existen las averias (DP.6)
// y el campo es obligatorio: hoy vale siempre 0.
std::string buildSummary(const OrderStats& stats, std::int64_t breakdowns,
                         const std::vector<std::unique_ptr<CourierActor>>& fleet);
