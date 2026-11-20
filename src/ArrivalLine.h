#pragma once

#include <cstdint>
#include <string>

// Valor de un campo de la linea `[fila]` de stderr, escapado para que la linea sea siempre una
// sola linea y los campos no se puedan falsificar: `\`, espacio, `=` y los caracteres de control
// (< 0x20 y 0x7f, incluidos \n \r \t) salen como `\xHH` (hex en minuscula). El resto, tal cual:
// los ids normales (`r0`, `c1`, `o17`) quedan identicos.
inline std::string escapeLineField(const std::string& value) {
    static const char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c < 0x20 || c == 0x7f || c == '\\' || c == ' ' || c == '=') {
            out += "\\x";
            out += kHex[c >> 4];
            out += kHex[c & 0x0f];
        } else {
            out += ch;
        }
    }
    return out;
}

// Linea de stderr `[fila] ...` (con el salto de linea final) que informa la llegada de un courier
// a la fila de un restaurante.
inline std::string arrivalLine(std::int64_t t, const std::string& courier, const std::string& order,
                               const std::string& restaurant, bool paused) {
    return "[fila] t=" + std::to_string(t) + " courier=" + escapeLineField(courier) +
           " order=" + escapeLineField(order) + " restaurant=" + escapeLineField(restaurant) +
           " pausa=" + (paused ? "si" : "no") + "\n";
}
