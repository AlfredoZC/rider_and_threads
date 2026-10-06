// Fase 1: carga y valida la configuración y dibuja el grafo de calles sobre el mapa.
// Todavía no hay hilos. Tecla G: mostrar/ocultar el grafo. ESC o cerrar la ventana: salir.
//
// Uso: delivery_sim <config.json> [--log <path>]
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Config.h"

namespace {

struct Arguments {
    std::string configPath;
    std::string logPath = "events.log";
};

// Devuelve false (y explica en stderr) si los argumentos no son válidos.
bool parseArguments(int argc, char** argv, Arguments& args) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <config.json> [--log <path>]\n";
        return false;
    }
    args.configPath = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--log" && i + 1 < argc) {
            args.logPath = argv[++i];
        } else if (flag == "--log") {
            std::cerr << "Error: --log necesita una ruta\n";
            return false;
        } else {
            std::cerr << "Error: argumento desconocido '" << flag << "'\n";
            return false;
        }
    }
    return true;
}

// Proyección lineal lat/lon -> píxel. La latitud crece hacia arriba y los píxeles
// hacia abajo, por eso y usa (north - lat).
cv::Point2f toPixel(double lat, double lon, const Bounds& b, const cv::Size& img) {
    const double x = (lon - b.west) / (b.east - b.west) * img.width;
    const double y = (b.north - lat) / (b.north - b.south) * img.height;
    return {static_cast<float>(x), static_cast<float>(y)};
}

// Las fuentes Hershey de OpenCV no tienen "©": se reemplaza por "(c)" para dibujarlo.
std::string drawableText(std::string text) {
    const std::string copyright = "\xC2\xA9";   // © en UTF-8
    const auto pos = text.find(copyright);
    if (pos != std::string::npos) {
        text.replace(pos, copyright.size(), "(c)");
    }
    return text;
}

void drawLabel(cv::Mat& img, const std::string& text, cv::Point origin, double scale) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    cv::rectangle(img, origin + cv::Point(-4, baseline + 4),
                  origin + cv::Point(size.width + 4, -size.height - 4), cv::Scalar(255, 255, 255),
                  cv::FILLED);
    cv::putText(img, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
}

// Dibuja calles (azul = doble sentido, rojo con flecha = un sentido), nodos y restaurantes.
void drawGraph(cv::Mat& img, const Config& cfg) {
    std::unordered_map<std::string, cv::Point2f> px;
    for (const auto& n : cfg.nodes) {
        px[n.id] = toPixel(n.lat, n.lon, cfg.bounds, img.size());
    }
    for (const auto& s : cfg.streets) {
        if (s.oneWay) {
            cv::arrowedLine(img, px[s.from], px[s.to], cv::Scalar(40, 40, 220), 2, cv::LINE_AA, 0, 0.25);
        } else {
            cv::line(img, px[s.from], px[s.to], cv::Scalar(220, 80, 30), 2, cv::LINE_AA);
        }
    }
    for (const auto& n : cfg.nodes) {
        cv::circle(img, px[n.id], 3, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(img, px[n.id], 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    const cv::Point2f start = px[cfg.startNode];
    cv::circle(img, start, 8, cv::Scalar(0, 160, 0), 3, cv::LINE_AA);
}

void drawRestaurants(cv::Mat& img, const Config& cfg) {
    std::unordered_map<std::string, cv::Point2f> px;
    for (const auto& n : cfg.nodes) {
        px[n.id] = toPixel(n.lat, n.lon, cfg.bounds, img.size());
    }
    for (const auto& r : cfg.restaurants) {
        const cv::Point center(cvRound(px[r.node].x), cvRound(px[r.node].y));
        cv::rectangle(img, center - cv::Point(7, 7), center + cv::Point(7, 7), cv::Scalar(0, 0, 0),
                      cv::FILLED);
        drawLabel(img, r.id + " " + r.name, center + cv::Point(10, -8), 0.4);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Arguments args;
    if (!parseArguments(argc, argv, args)) {
        return 2;
    }

    Config cfg;
    try {
        cfg = loadConfig(args.configPath);
    } catch (const ConfigError& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    cv::setNumThreads(0);   // OpenCV solo para la ventana: sin hilos internos propios

    cv::Mat image = cv::imread(cfg.imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: no se pudo leer la imagen del mapa '" << cfg.imagePath.string() << "'\n";
        return 1;
    }

    // Que la ventana quepa en pantalla. Se escala una sola vez, no en cada frame.
    const double kMaxWidth = 1400.0;
    const double kMaxHeight = 950.0;
    const double scale = std::min({1.0, kMaxWidth / image.cols, kMaxHeight / image.rows});
    if (scale < 1.0) {
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // Dos fondos precalculados: con y sin el grafo (tecla G)
    cv::Mat withGraph = image.clone();
    drawGraph(withGraph, cfg);
    for (cv::Mat* frame : {&image, &withGraph}) {
        drawRestaurants(*frame, cfg);
        drawLabel(*frame,
                  std::to_string(cfg.nodes.size()) + " nodos, " + std::to_string(cfg.streets.size()) +
                      " calles, " + std::to_string(cfg.restaurants.size()) + " restaurantes  [G: grafo]",
                  cv::Point(10, 22), 0.5);
        drawLabel(*frame, drawableText(cfg.attribution), cv::Point(10, frame->rows - 10), 0.45);
    }

    const std::string windowName = "Equipetrol Delivery";
    const int kEscKey = 27;
    const int kFrameMs = 33;
    bool showGraph = true;

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, withGraph);
    std::cout << "Simulation has started..." << std::endl;

    while (true) {
        cv::imshow(windowName, showGraph ? withGraph : image);
        const int key = cv::waitKey(kFrameMs);
        if (key == kEscKey) {
            break;
        }
        if (key == 'g' || key == 'G') {
            showGraph = !showGraph;
        }
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
