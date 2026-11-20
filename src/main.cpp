#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pthread.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Config.h"
#include "CoverageLayer.h"
#include "CourierActor.h"
#include "DemandHeat.h"
#include "Dispatcher.h"
#include "EventLog.h"
#include "Fleet.h"
#include "OrderBook.h"
#include "OrderGenerator.h"
#include "KitchenScheduler.h"
#include "Restaurant.h"
#include "SimClock.h"
#include "Summary.h"
#include "CityGraph.h"
#include "SimContext.h"
#include "accel/Backend.h"
#include "accel/DeliveryHistogram.h"
#include "accel/DeliveryPercentiles.h"
#include "accel/DeliveryTimeline.h"
#include "accel/OpenOrders.h"
#include "accel/GpuInfo.h"
#include "bench/BenchCli.h"
#include "bench/BenchMain.h"

namespace {

std::atomic<bool> g_stopSignal{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "el manejador de señales solo puede tocar atomicos sin candado");

extern "C" void onStopSignal(int) { g_stopSignal.store(true); }

struct Arguments {
    std::string configPath;
    std::string logPath = "events.log";
    bool noWindow = false;
    bool headless = false;
    accel::BackendChoice backend = accel::BackendChoice::automatic;
    std::string snapshotPath;  // --snapshot: PNG del ultimo cuadro dibujado
    std::string layers = "heat";  // --layers none|heat|coverage|both: capas visibles al empezar
};

bool parseArguments(int argc, char** argv, Arguments& args) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0]
                  << " <config.json> [--log <path>] [--no-window | --headless]\n"
                  << "       [--backend cpu|gpu|auto] [--snapshot <png>] [--layers none|heat|coverage|both]\n"
                  << "  --no-window  sin ventana, pero carga la imagen y dibuja cada cuadro\n"
                  << "  --headless   sin imagen, sin dibujo ni highgui; espera bloqueado\n"
                  << "  --backend    cpu | gpu | auto (por defecto): auto usa la GPU si esta lista\n"
                  << "  --snapshot   guarda el ultimo cuadro dibujado en un PNG (no vale con --headless)\n"
                  << "  --layers     capas al empezar (por defecto heat); teclas H (demanda), C (cobertura), G (grafo)\n";
        return false;
    }
    args.configPath = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--no-window") {
            args.noWindow = true;
        } else if (flag == "--headless") {
            args.headless = true;
        } else if (flag == "--backend" && i + 1 < argc) {
            bool ok = false;
            args.backend = accel::parseBackendChoice(argv[++i], ok);
            if (!ok) {
                std::cerr << "Error: --backend admite cpu, gpu o auto\n";
                return false;
            }
        } else if (flag == "--backend") {
            std::cerr << "Error: --backend necesita cpu, gpu o auto\n";
            return false;
        } else if (flag == "--layers" && i + 1 < argc) {
            args.layers = argv[++i];
            if (args.layers != "none" && args.layers != "heat" && args.layers != "coverage" && args.layers != "both") {
                std::cerr << "Error: --layers admite none, heat, coverage o both\n";
                return false;
            }
        } else if (flag == "--layers") {
            std::cerr << "Error: --layers necesita none, heat, coverage o both\n";
            return false;
        } else if (flag == "--snapshot" && i + 1 < argc) {
            args.snapshotPath = argv[++i];
        } else if (flag == "--snapshot") {
            std::cerr << "Error: --snapshot necesita una ruta\n";
            return false;
        } else if (flag == "--log" && i + 1 < argc) {
            args.logPath = argv[++i];
        } else if (flag == "--log") {
            std::cerr << "Error: --log necesita una ruta\n";
            return false;
        } else {
            std::cerr << "Error: argumento desconocido '" << flag << "'\n";
            return false;
        }
    }
    if (args.headless && !args.snapshotPath.empty()) {
        std::cerr << "Error: --snapshot necesita dibujar cuadros (usa --no-window o la ventana, no --headless)\n";
        return false;
    }
    return true;
}

cv::Point2f toPixel(double lat, double lon, const Bounds& b, const cv::Size& img) {
    const double x = (lon - b.west) / (b.east - b.west) * img.width;
    const double y = (b.north - lat) / (b.north - b.south) * img.height;
    return {static_cast<float>(x), static_cast<float>(y)};
}

std::string drawableText(std::string text) {
    const std::string copyright = "\xC2\xA9";
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

using PixelMap = std::unordered_map<std::string, cv::Point2f>;

PixelMap buildPixelMap(const Config& cfg, const cv::Size& size) {
    PixelMap px;
    px.reserve(cfg.nodes.size());
    for (const NodeConfig& n : cfg.nodes) {
        px[n.id] = toPixel(n.lat, n.lon, cfg.bounds, size);
    }
    return px;
}

void drawGraph(cv::Mat& img, const Config& cfg, const PixelMap& px) {
    for (const auto& s : cfg.streets) {
        if (s.oneWay) {
            cv::arrowedLine(img, px.at(s.from), px.at(s.to), cv::Scalar(40, 40, 220), 2,
                            cv::LINE_AA, 0, 0.25);
        } else {
            cv::line(img, px.at(s.from), px.at(s.to), cv::Scalar(220, 80, 30), 2, cv::LINE_AA);
        }
    }
    for (const auto& n : cfg.nodes) {
        cv::circle(img, px.at(n.id), 3, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(img, px.at(n.id), 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    cv::circle(img, px.at(cfg.startNode), 8, cv::Scalar(0, 160, 0), 3, cv::LINE_AA);
}

void drawRestaurants(cv::Mat& img, const Config& cfg, const PixelMap& px) {
    for (const auto& r : cfg.restaurants) {
        const cv::Point center(cvRound(px.at(r.node).x), cvRound(px.at(r.node).y));
        cv::rectangle(img, center - cv::Point(7, 7), center + cv::Point(7, 7),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        drawLabel(img, r.id + " " + r.name, center + cv::Point(10, -8), 0.4);
    }
}

cv::Scalar courierColor(CourierState s) {
    switch (s) {
        case CourierState::idle:         return cv::Scalar(150, 150, 150);
        case CourierState::toPickup:     return cv::Scalar(40, 200, 255);
        case CourierState::atRestaurant: return cv::Scalar(0, 140, 255);
        case CourierState::toDropoff:    return cv::Scalar(60, 190, 60);
        case CourierState::brokenDown:   return cv::Scalar(40, 40, 220);
        case CourierState::onBreak:      return cv::Scalar(220, 120, 200);
    }
    return cv::Scalar(255, 255, 255);
}

cv::Point2f courierPixel(const Courier& view, const PixelMap& px, std::int64_t nowMs) {
    const auto from = px.find(view.fromNode);
    const auto to = px.find(view.toNode);
    if (from == px.end()) {
        return {0.0f, 0.0f};
    }
    if (to == px.end()) {
        return from->second;
    }
    const float ratio = static_cast<float>(positionRatio(view, nowMs));
    return from->second + (to->second - from->second) * ratio;
}

void drawActiveDeliveries(cv::Mat& img, const PixelMap& px, const SimContext& ctx,
                          const std::vector<Courier>& views, std::int64_t nowMs) {
    const cv::Scalar violeta(200, 80, 200);
    for (const Courier& view : views) {
        for (const OrderId id : view.bag) {
            const auto order = ctx.book.find(id);
            if (!order.has_value() || isFinalState(order->state)) {
                continue;
            }
            const auto dropoff = px.find(order->dropoffNode);
            if (dropoff == px.end()) {
                continue;
            }
            const cv::Point2f desde = courierPixel(view, px, nowMs);
            const cv::Point destino(cvRound(dropoff->second.x), cvRound(dropoff->second.y));
            cv::line(img, cv::Point(cvRound(desde.x), cvRound(desde.y)), destino, violeta, 1,
                     cv::LINE_AA);
            cv::circle(img, destino, 4, violeta, cv::FILLED, cv::LINE_AA);
            cv::circle(img, destino, 4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }
}

void drawCouriers(cv::Mat& img, const PixelMap& px, const std::vector<Courier>& views,
                  std::int64_t nowMs) {
    for (const Courier& view : views) {
        const cv::Point2f pos = courierPixel(view, px, nowMs);
        const cv::Point center(cvRound(pos.x), cvRound(pos.y));

        cv::circle(img, center, 6, courierColor(view.state), cv::FILLED, cv::LINE_AA);
        cv::circle(img, center, 6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

void drawPanel(cv::Mat& img, const OrderStats& stats, std::int64_t nowMs,
               const std::string& attribution, std::int64_t joined,
               std::int64_t retired) {
    const int x = 10;
    int y = 24;
    drawLabel(img,
              "t=" + std::to_string(nowMs / 1000) + "s  creadas " +
                  std::to_string(stats.created) + "  entregadas " +
                  std::to_string(stats.delivered) + "  rechazadas " +
                  std::to_string(stats.rejected),
              cv::Point(x, y), 0.5);
    y += 22;
    drawLabel(img,
              "pendientes " + std::to_string(stats.pending) + "  averias " +
                  std::to_string(stats.breakdowns) + "  reasignadas " +
                  std::to_string(stats.reassigned) + "  media " +
                  std::to_string(stats.avgDeliveryMs / 1000) + "s",
              cv::Point(x, y), 0.5);

    y += 22;
    drawLabel(img, "refuerzos: " + std::to_string(joined) + " entraron, " +
                       std::to_string(retired) + " se retiraron",
              cv::Point(x, y), 0.45);

    y += 26;
    const struct {
        CourierState state;
        const char* name;
    } legend[] = {{CourierState::idle, "libre"},
                  {CourierState::toPickup, "al restaurante"},
                  {CourierState::atRestaurant, "esperando turno"},
                  {CourierState::toDropoff, "entregando"},
                  {CourierState::brokenDown, "averiado"},
                  {CourierState::onBreak, "descansando"}};
    int lx = x;
    for (const auto& item : legend) {
        cv::circle(img, cv::Point(lx + 6, y - 4), 5, courierColor(item.state), cv::FILLED,
                   cv::LINE_AA);
        cv::circle(img, cv::Point(lx + 6, y - 4), 5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        drawLabel(img, item.name, cv::Point(lx + 16, y), 0.38);
        lx += 16 + 8 * static_cast<int>(std::strlen(item.name)) + 14;
    }

    drawLabel(img, drawableText(attribution), cv::Point(x, img.rows - 10), 0.45);
}

// Capas semitransparentes sobre el mapa (demanda y cobertura). Cada una se arma UNA vez por resultado nuevo (a
// tamano de la imagen) y cada cuadro solo hace frame = frame * (1 - alfa) + color * alfa con operaciones
// saturadas de 8 bits.
struct MapOverlay {
    std::uint64_t version = 0;
    cv::Mat premultipliedColor;  // color * alfa
    cv::Mat inverseAlpha;        // 255 - alfa
};

struct LayerStyle {
    int colormap;
    bool emphasizeLow;  // false: la capa se ve donde el valor es ALTO (demanda); true: donde es BAJO (cobertura)
    const char* title;
    const char* keyHint;
};

constexpr double kLayerMaxOpacity = 0.7;
constexpr double kLayerTransparentBelow = 0.04;  // demanda: fraccion del maximo bajo la cual la capa es transparente
constexpr double kLayerTransparentAbove = 0.5;   // cobertura: fraccion del maximo desde la cual la capa es transparente

const LayerStyle kDemandStyle{cv::COLORMAP_JET, false, "demanda suavizada (pedidos/celda, 10 min)", "[H]"};
const LayerStyle kCoverageStyle{cv::COLORMAP_AUTUMN, true, "cobertura de la flota (rojo = poco cubierto)", "[C]"};

void buildOverlay(MapOverlay& overlay, const std::vector<float>& values, float maxValue, std::uint64_t version,
                  const LayerStyle& style, int gridSide, const cv::Size& imageSize) {
    cv::Mat field(gridSide, gridSide, CV_32F);
    std::memcpy(field.ptr<float>(), values.data(), values.size() * sizeof(float));
    cv::Mat level8;
    const double scale = maxValue > 0.0f ? 255.0 / maxValue : 0.0;
    field.convertTo(level8, CV_8U, scale);
    cv::resize(level8, level8, imageSize, 0, 0, cv::INTER_CUBIC);
    cv::Mat color;
    cv::applyColorMap(level8, color, style.colormap);
    cv::Mat opacityTable(1, 256, CV_8U);
    for (int level = 0; level < 256; ++level) {
        const double fraction = level / 255.0;
        double opacity = 0.0;
        if (style.emphasizeLow) {
            opacity = fraction >= kLayerTransparentAbove ? 0.0
                                                          : kLayerMaxOpacity * (1.0 - fraction / kLayerTransparentAbove);
        } else if (fraction > kLayerTransparentBelow) {
            opacity = kLayerMaxOpacity * std::min(1.0, (fraction - kLayerTransparentBelow) / 0.25);
        }
        opacityTable.at<unsigned char>(0, level) = static_cast<unsigned char>(std::lround(opacity * 255.0));
    }
    cv::Mat alpha8;
    cv::LUT(level8, opacityTable, alpha8);
    cv::Mat alpha3;
    cv::cvtColor(alpha8, alpha3, cv::COLOR_GRAY2BGR);
    cv::multiply(color, alpha3, overlay.premultipliedColor, 1.0 / 255.0);
    cv::subtract(cv::Scalar::all(255), alpha3, overlay.inverseAlpha);
    overlay.version = version;
}

void blendOverlay(cv::Mat& frame, const MapOverlay& overlay) {
    cv::Mat attenuated;
    cv::multiply(frame, overlay.inverseAlpha, attenuated, 1.0 / 255.0);
    cv::add(attenuated, overlay.premultipliedColor, frame);
}

// Leyenda de una capa; `slot` 0 es la de abajo y 1 la de arriba (cuando se ven las dos capas a la vez).
void drawLayerLegend(cv::Mat& img, float maxValue, bool usedGpu, const LayerStyle& style, int slot) {
    const int barWidth = 200;
    const int barHeight = 14;
    const int left = std::max(12, img.cols - barWidth - 40);
    const int top = img.rows - 52 - slot * 70;
    cv::Mat ramp(1, barWidth, CV_8U);
    for (int column = 0; column < barWidth; ++column) {
        ramp.at<unsigned char>(0, column) = static_cast<unsigned char>(column * 255 / (barWidth - 1));
    }
    cv::Mat rampColor;
    cv::applyColorMap(ramp, rampColor, style.colormap);
    cv::resize(rampColor, rampColor, cv::Size(barWidth, barHeight), 0, 0, cv::INTER_NEAREST);
    rampColor.copyTo(img(cv::Rect(left, top, barWidth, barHeight)));
    cv::rectangle(img, cv::Rect(left, top, barWidth, barHeight), cv::Scalar(0, 0, 0), 1);
    char maxText[48];
    std::snprintf(maxText, sizeof maxText, "%.2f", static_cast<double>(maxValue));
    const auto rightAlignedLabel = [&img](const std::string& text, int baselineY) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        drawLabel(img, text, cv::Point(std::max(8, img.cols - size.width - 12), baselineY), 0.4);
    };
    rightAlignedLabel(style.title, top - 26);
    rightAlignedLabel(std::string(style.keyHint) + " ocultar - calculo en " + (usedGpu ? "GPU" : "CPU"), top - 6);
    drawLabel(img, "0", cv::Point(left, top + barHeight + 18), 0.4);
    drawLabel(img, maxText, cv::Point(left + barWidth - 30, top + barHeight + 18), 0.4);
}

// Modo --headless: el hilo principal espera el fin de la simulacion BLOQUEADO, sin sondeo.
// SIGINT/SIGTERM se bloquean en todos los hilos (la mascara se hereda, asi que debe fijarse
// antes de crear el primero) y un hilo dedicado las recibe con sigwait, que corre en contexto
// normal (no en un manejador): ahi si se puede tomar un mutex y notificar la condition_variable.
// Una senal que llegue antes de crear ese hilo queda pendiente y sigwait la recoge.
class HeadlessStop {
public:
    // false si el sistema no pudo bloquearlas: sin bloqueo sigwait no es fiable.
    static bool blockSignals() {
        const sigset_t set = stopSet();
        return pthread_sigmask(SIG_BLOCK, &set, nullptr) == 0;
    }

    HeadlessStop() = default;
    HeadlessStop(const HeadlessStop&) = delete;
    HeadlessStop& operator=(const HeadlessStop&) = delete;

    // Si algo lanza entre start() y finish(), evita std::terminate por un hilo unible.
    ~HeadlessStop() {
        if (waiter_.joinable()) {
            finish();
        }
    }

    void start() { waiter_ = std::thread(&HeadlessStop::waitForSignal, this); }

    // Bloquea hasta una senal o hasta que el reloj simulado llegue a durationMs (0 = sin limite).
    void waitUntil(const SimClock& clock, std::int64_t durationMs) {
        // Una senal que llego entre std::signal y blockSignals ejecuto onStopSignal en este
        // mismo hilo (el unico que existia) y dejo la bandera: ya cuenta como parada recibida.
        if (g_stopSignal.load()) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        while (!signaled_) {
            if (durationMs <= 0) {
                cv_.wait(lock, [this] { return signaled_; });
            } else if (clock.nowMs() >= durationMs) {
                return;
            } else {
                cv_.wait_until(lock, clock.timePointAt(durationMs), [this] { return signaled_; });
            }
        }
    }

    // Libera al hilo de senales (si no se fue ya) y lo une.
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!waiterDone_) {
                pthread_kill(waiter_.native_handle(), SIGTERM);
            }
        }
        waiter_.join();
    }

private:
    static sigset_t stopSet() {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGTERM);
        return set;
    }

    void waitForSignal() {
        const sigset_t set = stopSet();
        int received = 0;
        // Si sigwait falla se trata como parada: es el lado seguro (la simulacion cierra limpia).
        (void)sigwait(&set, &received);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            waiterDone_ = true;
            signaled_ = true;
        }
        cv_.notify_all();
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    bool signaled_ = false;
    bool waiterDone_ = false;
    std::thread waiter_;
};

}

int main(int argc, char** argv) {
    // Modo bench (--bench / --bench-once): no corre la simulacion ni abre ventana; ver src/bench/.
    if (argc >= 2) {
        const std::vector<std::string> benchArgs(argv + 1, argv + argc);
        if (bench::isBenchInvocation(benchArgs)) {
            return bench::runBenchMode(benchArgs);
        }
    }

    Arguments args;
    if (!parseArguments(argc, argv, args)) {
        return 2;
    }

    // Disposicion explicita en ambos modos: con un SIG_IGN heredado (p. ej. bash con '&') una
    // senal bloqueada queda pendiente en Linux, pero POSIX no lo especifica; por eso no se
    // depende de ello. Orden: handler primero y bloqueo despues. Una senal anterior al bloqueo
    // corre el handler en el hilo principal (unico hilo) y deja g_stopSignal, que waitUntil
    // revisa; una posterior queda pendiente y sigwait la recoge. No hay orden que la pierda.
    std::signal(SIGINT, onStopSignal);
    std::signal(SIGTERM, onStopSignal);
    HeadlessStop headlessStop;
    if (args.headless && !HeadlessStop::blockSignals()) {
        std::cerr << "Error: no se pudo bloquear SIGINT/SIGTERM para --headless\n";
        return 1;
    }

    Config cfg;
    try {
        cfg = loadConfig(args.configPath);
    } catch (const ConfigError& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    cv::setNumThreads(0);

    const SimClock clock(cfg.timeScale);

    std::unique_ptr<EventLog> log;
    try {
        log = std::make_unique<EventLog>(args.logPath, clock);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    cv::Mat image;
    cv::Mat withGraph;
    PixelMap px;
    if (!args.headless) {
        image = cv::imread(cfg.imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: no se pudo leer la imagen del mapa '" << cfg.imagePath.string() << "'\n";
            return 1;
        }

        const double kMaxWidth = 1400.0;
        const double kMaxHeight = 950.0;
        const double scale = std::min({1.0, kMaxWidth / image.cols, kMaxHeight / image.rows});
        if (scale < 1.0) {
            cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        px = buildPixelMap(cfg, image.size());

        withGraph = image.clone();
        drawGraph(withGraph, cfg, px);
        for (cv::Mat* background : {&image, &withGraph}) {
            drawRestaurants(*background, cfg, px);
        }
    }

    const std::string windowName = "Equipetrol Delivery";
    const int kEscKey = 27;
    const int kFrameMs = 33;
    bool showGraph = false;
    bool showHeat = args.layers == "heat" || args.layers == "both";
    bool showCoverage = args.layers == "coverage" || args.layers == "both";  // teclas H y C alternan

    const CityGraph graph(cfg);

    OrderBook book(cfg.maxPending, clock, *log);
    OrderGenerator generator(cfg, clock, book);
    Restaurants restaurants;
    std::vector<Restaurant*> restaurantsInOrder;
    for (std::size_t i = 0; i < cfg.restaurants.size(); ++i) {
        const RestaurantConfig& r = cfg.restaurants[i];
        KitchenSchedule schedule;
        if (cfg.kitchen.enabled) {
            schedule = KitchenSchedule{cfg.kitchen.pauseEveryMs, cfg.kitchen.pauseMs,
                                       static_cast<std::int64_t>(i),
                                       static_cast<std::int64_t>(cfg.restaurants.size())};
        }
        auto restaurant = std::make_unique<Restaurant>(r.id, r.pickupSlots, schedule, &clock);
        restaurantsInOrder.push_back(restaurant.get());
        restaurants.emplace(r.id, std::move(restaurant));
    }
    KitchenScheduler kitchen(restaurantsInOrder, clock, *log);

    // Backend GPU: despues de los errores de arranque posibles, antes del primer hilo de la
    // simulacion. El hilo dueño de la GPU se apaga mas abajo, DESPUES de los hilos que lo usan.
    accel::BackendSelection backend = accel::selectBackend(args.backend);
    std::cerr << backend.line << std::endl;
    if (backend.failed) {
        return 1;
    }

    const SimContext ctx{graph, clock, *log, book, restaurants, backend.worker.get()};

    Fleet fleet(cfg, ctx);
    for (int i = 0; i < cfg.couriers; ++i) {
        auto actor = std::make_unique<CourierActor>(
            static_cast<CourierId>(i) + 1, cfg.startNode, cfg.bagCapacity, ctx,
            cfg.breakdownProbability, cfg.seed, false, 0, cfg.breakAfterDeliveries,
            cfg.breakMs);
        actor->setFleet(&fleet);
        fleet.addAndStart(std::move(actor));
    }
    Dispatcher dispatcher(cfg, ctx, fleet);
    std::thread dispatcherThread(&Dispatcher::run, &dispatcher);

    std::thread generatorThread(&OrderGenerator::run, &generator);

    if (!args.noWindow && !args.headless) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    }
    kitchen.start();
    std::cout << "Simulation has started..." << std::endl;

    const std::int64_t durationMs = secondsToMsSaturated(cfg.durationS);

    DemandHeatService heatService(cfg, ctx);
    heatService.start();
    MapOverlay heatOverlay;
    CoverageFieldService coverageService(cfg, ctx, fleet);
    coverageService.start();
    MapOverlay coverageOverlay;

    cv::Mat frame;
    std::vector<Courier> views;
    views.reserve(static_cast<std::size_t>(cfg.couriers));
    if (args.headless) {
        headlessStop.start();
        headlessStop.waitUntil(clock, durationMs);
        headlessStop.finish();
    }
    std::vector<double> frameWorkMs;
    const auto loopStart = std::chrono::steady_clock::now();
    while (!args.headless) {
        const auto frameStart = std::chrono::steady_clock::now();
        (showGraph ? withGraph : image).copyTo(frame);

        const std::int64_t nowMs = clock.nowMs();

        views.clear();
        for (CourierActor* actor : fleet.activeSnapshot()) {
            views.push_back(actor->snapshot());
        }
        // Capas: dibujan el ULTIMO resultado listo; nunca esperan a la GPU (lo calculan los hilos de los servicios).
        int legendSlot = 0;
        if (showCoverage) {
            const CoverageSnapshot coverageSnapshot = coverageService.latest();
            if (coverageSnapshot.coverage) {
                if (coverageOverlay.version != coverageSnapshot.version || coverageOverlay.premultipliedColor.empty()) {
                    buildOverlay(coverageOverlay, *coverageSnapshot.coverage, coverageSnapshot.scaleValue,
                                 coverageSnapshot.version, kCoverageStyle, kCoverageGridSide, frame.size());
                }
                blendOverlay(frame, coverageOverlay);
                drawLayerLegend(frame, coverageSnapshot.scaleValue, coverageSnapshot.usedGpu, kCoverageStyle, legendSlot++);
            }
        }
        if (showHeat) {
            const DemandHeatSnapshot heatSnapshot = heatService.latest();
            if (heatSnapshot.demandHeat) {
                if (heatOverlay.version != heatSnapshot.version || heatOverlay.premultipliedColor.empty()) {
                    buildOverlay(heatOverlay, *heatSnapshot.demandHeat, heatSnapshot.maxValue, heatSnapshot.version,
                                 kDemandStyle, kDemandGridSide, frame.size());
                }
                blendOverlay(frame, heatOverlay);
                drawLayerLegend(frame, heatSnapshot.maxValue, heatSnapshot.usedGpu, kDemandStyle, legendSlot++);
            }
        }
        drawActiveDeliveries(frame, px, ctx, views, nowMs);
        drawCouriers(frame, px, views, nowMs);
        drawPanel(frame, book.stats(), nowMs, cfg.attribution,
                  fleet.reinforcementsJoined(), fleet.reinforcementsRetired());
        if (args.noWindow) {
            frameWorkMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count());
            std::this_thread::sleep_for(std::chrono::milliseconds(kFrameMs));
        } else {
            cv::imshow(windowName, frame);
            frameWorkMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count());
            const int key = cv::waitKey(kFrameMs);
            if (key == kEscKey) {
                break;
            }
            if (key == 'g' || key == 'G') {
                showGraph = !showGraph;
            }
            if (key == 'h' || key == 'H') {
                showHeat = !showHeat;
            }
            if (key == 'c' || key == 'C') {
                showCoverage = !showCoverage;
            }
            if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
                break;
            }
        }
        if (g_stopSignal.load()) {
            break;
        }
        if (durationMs > 0 && clock.nowMs() >= durationMs) {
            break;
        }
    }

    // Cuadros dibujados (ventana o --no-window): el tiempo de dibujo EXCLUYE la espera del ritmo de cuadros
    // (sleep / waitKey). En --headless no hay cuadros y no se imprime nada.
    if (!args.headless && !frameWorkMs.empty()) {
        const double loopSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loopStart).count();
        std::vector<double> sortedWorkMs(frameWorkMs);
        std::sort(sortedWorkMs.begin(), sortedWorkMs.end());
        const auto percentile = [&sortedWorkMs](double fraction) {
            return sortedWorkMs[static_cast<std::size_t>(fraction * static_cast<double>(sortedWorkMs.size() - 1))];
        };
        char fpsLine[200];
        std::snprintf(fpsLine, sizeof fpsLine,
                      "[estadisticas] cuadros: %zu, fps medio %.1f, tiempo de dibujo p50 %.2f ms / p95 %.2f ms",
                      frameWorkMs.size(), loopSeconds > 0.0 ? static_cast<double>(frameWorkMs.size()) / loopSeconds : 0.0,
                      percentile(0.50), percentile(0.95));
        std::cerr << fpsLine << std::endl;
    }

    if (!args.snapshotPath.empty()) {
        if (frame.empty() || !cv::imwrite(args.snapshotPath, frame)) {
            std::cerr << "Aviso: no se pudo guardar la captura '" << args.snapshotPath << "'\n";
        }
    }

    generator.requestStop();
    generatorThread.join();

    kitchen.requestStop();
    kitchen.join();

    Event stopping;
    stopping.name = EventName::kSimulationStopping;
    log->write(stopping);

    dispatcher.requestStop();
    dispatcherThread.join();

    for (auto& entry : restaurants) {
        entry.second->stop();
    }

    fleet.requestStopAll();
    fleet.joinAll();

    // Estadistica extra (solo stderr, el resumen oficial de stdout no cambia): histograma de los
    // tiempos de entrega por minuto, pedido al hilo de la GPU con respaldo en la CPU. Va antes de
    // apagar el worker, con todos los hilos de la simulacion ya unidos.
    {
        const accel::DeliveryHistogramResult histogram =
            accel::deliveryTimeHistogram(book.deliveredTimesMs(), backend.worker.get());
        std::cerr << accel::formatDeliveryHistogram(histogram.bins, histogram.usedGpu ? "gpu" : "cpu")
                  << std::endl;
    }

    // Pedidos en curso a lo largo del tiempo (scan de +1/-1 por evento) y su pico, tambien por el
    // hilo de la GPU con respaldo CPU. Solo stderr; los pedidos sin cerrar quedan abiertos al final.
    {
        const std::vector<OrderSpan> spans = book.orderSpans();
        std::vector<accel::OrderLifetime> lifetimes;
        lifetimes.reserve(spans.size());
        for (const OrderSpan& span : spans) {
            lifetimes.push_back({span.createdMs, span.closedMs == kNoTime ? accel::kStillOpen : span.closedMs});
        }
        const accel::OpenOrdersSummary openOrders = accel::openOrdersSummary(lifetimes, backend.worker.get());
        std::cerr << accel::formatOpenOrders(openOrders.peakOpenOrders, openOrders.peakTimeMs, openOrders.openAtEnd,
                                             openOrders.usedGpu ? "gpu" : "cpu")
                  << std::endl;
    }

    // Percentiles p50, p90 y p99 del tiempo de entrega (radix sort + rango mas cercano), tambien por el
    // hilo de la GPU con respaldo CPU. Solo stderr.
    {
        const accel::DeliveryPercentilesSummary percentiles =
            accel::deliveryPercentiles(book.deliveredTimesMs(), backend.worker.get());
        std::cerr << accel::formatDeliveryPercentiles(percentiles.percentiles, percentiles.count,
                                                      percentiles.usedGpu ? "gpu" : "cpu")
                  << std::endl;
    }

    // Linea de tiempo global de entregas: la lista de cada courier (ya ordenada por hora) se fusiona en la
    // GPU con respaldo CPU y de ahi salen las entregas por hora simulada. Un courier sin entregas aporta una
    // lista vacia. Solo stderr; la suma de las horas es `delivered` del resumen.
    {
        const std::vector<DeliveryEntry> entries = book.deliveries();
        std::unordered_map<CourierId, std::size_t> listOfCourier;
        const std::vector<CourierActor*> actors = fleet.allSnapshot();
        for (std::size_t list = 0; list < actors.size(); ++list) {
            listOfCourier.emplace(actors[list]->id(), list);
        }
        std::vector<std::vector<accel::DeliveryRecord>> perCourier(actors.size());
        for (const DeliveryEntry& entry : entries) {  // llegan ordenadas por courier y hora
            const auto found = listOfCourier.find(entry.courier);
            if (found != listOfCourier.end()) {
                perCourier[found->second].push_back({static_cast<std::int32_t>(std::clamp<std::int64_t>(entry.deliveredMs, 0, INT_MAX)),
                                                     static_cast<std::int32_t>(std::min<OrderId>(entry.orderId, INT_MAX))});
            }
        }
        std::vector<accel::DeliveryRecord> records;
        std::vector<int> listStart{0};
        for (const auto& list : perCourier) {
            records.insert(records.end(), list.begin(), list.end());
            listStart.push_back(static_cast<int>(records.size()));
        }
        const accel::DeliveryTimelineSummary timeline =
            accel::deliveryTimelineSummary(records, listStart, backend.worker.get());
        std::cerr << accel::formatDeliveriesPerHour(timeline.deliveriesPerHour, timeline.usedGpu ? "gpu" : "cpu")
                  << std::endl;
    }

    // Mapa de calor: se une el hilo del servicio, se calcula una vez mas con los pedidos finales (todavia con
    // el worker vivo) y se informa la zona mas caliente en stderr (el resumen oficial no cambia).
    heatService.requestStop();
    heatService.join();
    heatService.refresh();
    coverageService.requestStop();
    coverageService.join();
    coverageService.refresh();
    std::cerr << heatService.hottestZone() << std::endl;
    std::cerr << coverageService.leastCoveredZone() << std::endl;

    // Ya no queda ningun hilo que envie trabajo a la GPU: se apaga (resuelve lo pendiente y une).
    if (backend.worker) {
        backend.worker->stop();
    }

    book.markRemainingAsPending();
    const OrderStats stats = book.stats();
    std::int64_t pausesTotal = 0;
    std::int64_t delayedTotal = 0;
    for (const Restaurant* r : restaurantsInOrder) {
        pausesTotal += r->pauses();
        delayedTotal += r->delayedByPause();
    }

    std::cout << buildSummary(stats, stats.breakdowns, fleet.allSnapshot(),
                              fleet.reinforcementsJoined(),
                              fleet.reinforcementsRetired(), pausesTotal,
                              delayedTotal) << std::endl;

    if (!args.noWindow && !args.headless) {
        cv::destroyAllWindows();
    }
    return 0;
}
