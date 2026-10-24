#include "Summary.h"

std::string buildSummary(const OrderStats& stats, std::int64_t breakdowns,
                         const std::vector<CourierActor*>& fleet,
                         std::int64_t reinforcementsJoined,
                         std::int64_t reinforcementsRetired) {
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
    out += ",\"reinforcementsJoined\":" + std::to_string(reinforcementsJoined);
    out += ",\"reinforcementsRetired\":" + std::to_string(reinforcementsRetired);

    out += ",\"deliveredPerCourier\":{";
    bool first = true;
    for (const CourierActor* actor : fleet) {
        const Courier view = actor->snapshot();
        if (!first) {
            out += ',';
        }
        first = false;
        out += "\"" + courierLabel(view.id) + "\":" + std::to_string(view.delivered);
    }
    out += "}}";
    return out;
}

std::string buildSummary(const OrderStats& stats, std::int64_t breakdowns,
                         const std::vector<std::unique_ptr<CourierActor>>& fleet) {
    std::vector<CourierActor*> actors;
    actors.reserve(fleet.size());
    for (const auto& actor : fleet) {
        actors.push_back(actor.get());
    }
    return buildSummary(stats, breakdowns, actors);
}
