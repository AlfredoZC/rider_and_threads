#include "Summary.h"

std::string buildSummary(const OrderStats& stats, std::int64_t breakdowns,
                         const std::vector<std::unique_ptr<CourierActor>>& fleet) {
    std::string out;
    out.reserve(256);
    out += "{\"created\":" + std::to_string(stats.created);
    out += ",\"delivered\":" + std::to_string(stats.delivered);
    out += ",\"rejected\":" + std::to_string(stats.rejected);
    out += ",\"pending\":" + std::to_string(stats.pending);
    out += ",\"reassigned\":" + std::to_string(stats.reassigned);
    out += ",\"breakdowns\":" + std::to_string(breakdowns);
    out += ",\"avgDeliveryMs\":" + std::to_string(stats.avgDeliveryMs);
    out += ",\"maxDeliveryMs\":" + std::to_string(stats.maxDeliveryMs);

    // Las claves llevan el mismo prefijo que en el log ("c3"), para que un script pueda
    // cruzar el resumen con las lineas del archivo sin traducir nada.
    out += ",\"deliveredPerCourier\":{";
    bool first = true;
    for (const auto& actor : fleet) {
        // El dato sale del propio courier, no de un contador paralelo del resumen: el §6
        // exige que estos valores sumen delivered, y la forma de garantizarlo es que
        // vengan del mismo lugar donde se incrementaron.
        const Courier view = actor->snapshot();
        if (!first) {
            out += ',';
        }
        first = false;
        out += "\"c" + std::to_string(view.id) + "\":" + std::to_string(view.delivered);
    }
    out += "}}";
    return out;
}
