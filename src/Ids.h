#pragma once 

#include <cstdint>
#include <string>

using OrderId = std::uint64_t;
using CourierId = std::uint64_t;

inline constexpr OrderId kNoOrder = 0;
inline constexpr CourierId kNoCourier = 0;

inline constexpr CourierId kReinforcementMask = CourierId{1} << 63;

inline CourierId reinforcementId(std::uint64_t sequence) {
    return kReinforcementMask | (sequence + 1);
}

inline bool isReinforcementId(CourierId id) {
    return (id & kReinforcementMask) != 0;
}

inline std::string courierLabel(CourierId id) {
    return isReinforcementId(id)
               ? "r" + std::to_string((id & ~kReinforcementMask) - 1)
               : "c" + std::to_string(id);
}
