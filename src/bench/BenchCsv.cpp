#include "bench/BenchCsv.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstring>

namespace bench {

namespace {

std::string formatMilliseconds(double milliseconds) {
    char digits[64];
    const auto conversion =
        std::to_chars(digits, digits + sizeof digits, milliseconds, std::chars_format::fixed, 6);
    if (conversion.ec != std::errc()) {
        return "nan";
    }
    return std::string(digits, conversion.ptr);
}

void appendOptionalMilliseconds(std::string& line, const std::optional<double>& milliseconds) {
    line += ',';
    if (milliseconds) {
        line += formatMilliseconds(*milliseconds);
    }
}

}  // namespace

std::string csvHeader() {
    return "application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms";
}

std::string csvEscape(const std::string& field) {
    if (field.find_first_of(",\"\r\n") == std::string::npos) {
        return field;
    }
    std::string escaped = "\"";
    for (char character : field) {
        if (character == '"') {
            escaped += '"';
        }
        escaped += character;
    }
    escaped += '"';
    return escaped;
}

std::string csvRow(const Row& row) {
    std::string line = csvEscape(row.application);
    line += ',';
    line += csvEscape(row.version);
    line += ',';
    line += std::to_string(row.size);
    line += ',';
    line += std::to_string(row.rep);
    line += ',';
    line += formatMilliseconds(row.ms);
    line += ',';
    line += row.correct ? "yes" : "no";
    line += ',';
    line += std::to_string(row.threads);
    line += ',';
    line += std::to_string(row.seed);
    appendOptionalMilliseconds(line, row.h2dMs);
    appendOptionalMilliseconds(line, row.kernelMs);
    appendOptionalMilliseconds(line, row.d2hMs);
    return line;
}

std::FILE* openOutputFile(const std::string& path, bool force, std::string& error) {
    const int openFlags = O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC | (force ? O_TRUNC : O_EXCL);
    const int fileDescriptor = ::open(path.c_str(), openFlags, 0644);
    if (fileDescriptor < 0) {
        if (!force && errno == EEXIST) {
            error = "'" + path + "' ya existe; usa --force para sobrescribirlo";
        } else if (errno == ELOOP) {
            error = "'" + path + "' es un enlace simbolico; no se sigue";
        } else {
            error = "no se pudo crear '" + path + "': " + std::strerror(errno);
        }
        return nullptr;
    }
    std::FILE* stream = ::fdopen(fileDescriptor, "w");
    if (stream == nullptr) {
        error = "no se pudo abrir '" + path + "': " + std::strerror(errno);
        ::close(fileDescriptor);
    }
    return stream;
}

CsvFile::~CsvFile() {
    if (file_ != nullptr) {
        std::fclose(file_);
    }
}

std::string CsvFile::open(const std::string& path, bool force) {
    std::string error;
    file_ = openOutputFile(path, force, error);
    if (file_ == nullptr) {
        return error;
    }
    const std::string header = csvHeader() + "\n";
    if (std::fputs(header.c_str(), file_) < 0 || std::fflush(file_) != 0) {
        failed_ = true;
        return "no se pudo escribir en '" + path + "'";
    }
    return "";
}

void CsvFile::write(const Row& row) {
    if (file_ == nullptr || failed_) {
        return;
    }
    const std::string line = csvRow(row) + "\n";
    if (std::fputs(line.c_str(), file_) < 0 || std::fflush(file_) != 0) {
        failed_ = true;
    }
}

}  // namespace bench
