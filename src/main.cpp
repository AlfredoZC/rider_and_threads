#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Config.h"
#include "CourierActor.h"
#include "Dispatcher.h"
#include "EventLog.h"
#include "OrderBook.h"
#include "OrderGenerator.h"
#include "Restaurant.h"
#include "SimClock.h"
#include "Summary.h"
#include "CityGraph.h"
#include "SimContext.h"

namespace {

std::atomic<bool> g_stopSignal{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "el manejador de señales solo puede tocar atomicos sin candado");

extern "C" void onStopSignal(int) { g_stopSignal.store(true); }

struct Arguments {
    std::string configPath;
    std::string logPath = "events.log";
    bool noWindow = false;
};

bool parseArguments(int argc, char** argv, Arguments& args) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0]
                  << " <config.json> [--log <path>] [--no-window]\n";
        return false;
    }
    args.configPath = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--no-window") {
            args.noWindow = true;
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
        if (view.bag.empty()) {
            continue;
        }
        const auto order = ctx.book.find(view.bag.front());
        if (!order.has_value()) {
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

void drawCouriers(cv::Mat& img, const PixelMap& px, const std::vector<Courier>& views,
                  std::int64_t nowMs) {
    const std::size_t total = views.size();
    for (std::size_t i = 0; i < total; ++i) {
        const Courier& view = views[i];
        const cv::Point2f pos = courierPixel(view, px, nowMs);

        const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(i) /
                             static_cast<double>(total == 0 ? 1 : total);
        const cv::Point center(cvRound(pos.x + 9.0 * std::cos(angle)),
                               cvRound(pos.y + 9.0 * std::sin(angle)));

        cv::circle(img, center, 6, courierColor(view.state), cv::FILLED, cv::LINE_AA);
        cv::circle(img, center, 6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

void drawPanel(cv::Mat& img, const OrderStats& stats, std::int64_t nowMs,
               const std::string& attribution) {
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

    y += 26;
    const struct {
        CourierState state;
        const char* name;
    } legend[] = {{CourierState::idle, "libre"},
                  {CourierState::toPickup, "al restaurante"},
                  {CourierState::atRestaurant, "esperando turno"},
                  {CourierState::toDropoff, "entregando"},
                  {CourierState::brokenDown, "averiado"}};
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

}

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

    cv::setNumThreads(0);

    const SimClock clock(cfg.timeScale);

    std::unique_ptr<EventLog> log;
    try {
        log = std::make_unique<EventLog>(args.logPath, clock);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    cv::Mat image = cv::imread(cfg.imagePath.string(), cv::IMREAD_COLOR);
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

    const PixelMap px = buildPixelMap(cfg, image.size());

    cv::Mat withGraph = image.clone();
    drawGraph(withGraph, cfg, px);
    for (cv::Mat* background : {&image, &withGraph}) {
        drawRestaurants(*background, cfg, px);
    }

    const std::string windowName = "Equipetrol Delivery";
    const int kEscKey = 27;
    const int kFrameMs = 33;
    bool showGraph = false;

    const CityGraph graph(cfg);

    OrderBook book(cfg.maxPending, clock, *log);
    OrderGenerator generator(cfg, clock, book);
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id, std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }

    const SimContext ctx{graph, clock, *log, book, restaurants};

    std::vector<std::unique_ptr<CourierActor>> fleet;
    fleet.reserve(static_cast<std::size_t>(cfg.couriers));
    for (int i = 0; i < cfg.couriers; ++i) {
        fleet.push_back(std::make_unique<CourierActor>(
            static_cast<CourierId>(i) + 1, cfg.startNode, cfg.bagCapacity, ctx,
            cfg.breakdownProbability, cfg.seed));
    }
    for (auto& actor : fleet) {
        actor->setFleet(&fleet);
    }

    std::vector<std::thread> threads;
    threads.reserve(fleet.size());
    for (auto& actor : fleet) {
        threads.emplace_back(&CourierActor::run, actor.get());
    }
    Dispatcher dispatcher(cfg, ctx, fleet);
    std::thread dispatcherThread(&Dispatcher::run, &dispatcher);

    std::thread generatorThread(&OrderGenerator::run, &generator);

    if (!args.noWindow) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    }
    std::cout << "Simulation has started..." << std::endl;

    std::signal(SIGINT, onStopSignal);
    std::signal(SIGTERM, onStopSignal);

    const std::int64_t durationMs = cfg.durationS * 1000;

    cv::Mat frame;
    std::vector<Courier> views;
    views.reserve(fleet.size());
    while (true) {
        (showGraph ? withGraph : image).copyTo(frame);

        const std::int64_t nowMs = clock.nowMs();

        views.clear();
        for (const auto& actor : fleet) {
            views.push_back(actor->snapshot());
        }
        drawActiveDeliveries(frame, px, ctx, views, nowMs);
        drawCouriers(frame, px, views, nowMs);
        drawPanel(frame, book.stats(), nowMs, cfg.attribution);
        if (args.noWindow) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFrameMs));
        } else {
            cv::imshow(windowName, frame);
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
        if (g_stopSignal.load()) {
            break;
        }
        if (durationMs > 0 && clock.nowMs() >= durationMs) {
            break;
        }
    }

    Event stopping;
    stopping.name = EventName::kSimulationStopping;
    log->write(stopping);

    generator.requestStop();
    generatorThread.join();

    dispatcher.requestStop();
    dispatcherThread.join();

    for (auto& entry : restaurants) {
        entry.second->stop();
    }

    for (auto& actor : fleet) {
        actor->requestStop();
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    book.markRemainingAsPending();
    const OrderStats stats = book.stats();

    std::cout << buildSummary(stats, stats.breakdowns, fleet) << std::endl;

    cv::destroyAllWindows();
    return 0;
}
