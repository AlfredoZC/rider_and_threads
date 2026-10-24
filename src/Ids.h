#pragma once 

#include <cstdint>

using OrderId = std::uint64_t;
using CourierId = std::uint64_t;

inline constexpr OrderId kNoOrder = 0;
inline constexpr CourierId kNoCourier = 0;
