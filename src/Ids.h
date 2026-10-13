#pragma once 

#include <cstdint>

using OrderId = std::uint64_t;
using CourierId = std::uint64_t;


// EL 0 nunca es un id valid: significa "ninguno". Las factories numera desde 1.

inline constexpr OrderId kNoOrder = 0;
inline constexpr CourierId kNoCourier = 0;
