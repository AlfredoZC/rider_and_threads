#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace bench {

// Una fila del CSV: una corrida de una version. Las seis primeras columnas son las del enunciado
// (seccion 2.4): application, version, size, rep, ms, correct. Las demas son contexto.
struct Row {
    std::string application;
    std::string version;
    std::size_t size = 0;
    std::size_t rep = 0;
    double ms = 0.0;
    bool correct = false;
    unsigned threads = 1;
    std::uint32_t seed = 0;
    std::optional<double> h2dMs;
    std::optional<double> kernelMs;
    std::optional<double> d2hMs;
};

// Encabezado, sin salto de linea.
std::string csvHeader();
// Cita el campo solo si trae coma, comillas o salto de linea (RFC 4180).
std::string csvEscape(const std::string& field);
// La fila, sin salto de linea. Los tiempos van con 6 decimales; los que no aplican, vacios.
std::string csvRow(const Row& row);

// Abre un archivo de salida nuevo. Sin `force` la apertura es exclusiva (O_EXCL: falla si existe, sin
// ventana de carrera); con `force` trunca un archivo comun existente. En ambos casos O_NOFOLLOW: no
// sigue enlaces simbolicos. Devuelve nullptr y llena `errorMessage` si no pudo.
std::FILE* openOutputFile(const std::string& path, bool force, std::string& error);

// Destino de las filas (el runner no sabe si es un archivo o memoria).
class RowSink {
public:
    virtual ~RowSink() = default;
    virtual void write(const Row& row) = 0;
};

// Archivo CSV: escribe el encabezado al abrir y hace flush por fila.
class CsvFile : public RowSink {
public:
    CsvFile() = default;
    ~CsvFile() override;
    CsvFile(const CsvFile&) = delete;
    CsvFile& operator=(const CsvFile&) = delete;

    // Sin `force` falla si el archivo ya existe (apertura exclusiva "wx", sin ventana de carrera).
    // Devuelve "" si abrio o un mensaje de error.
    std::string open(const std::string& path, bool force);
    void write(const Row& row) override;
    bool good() const { return file_ != nullptr && !failed_; }

private:
    std::FILE* file_ = nullptr;
    bool failed_ = false;
};

}  // namespace bench
