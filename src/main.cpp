// Carga y valida la configuración, dibuja el grafo de calles sobre el mapa y lanza un
// hilo por courier. Los couriers todavía no reparten: esperan sin consumir CPU hasta que
// el hilo principal les pide terminar. Tecla G: mostrar/ocultar el grafo. ESC o cerrar la
// ventana: salir.
//
// Uso: delivery_sim <config.json> [--log <path>] [--no-window]
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

// La bandera de parada por señal.
//
// Es global porque un manejador de señales no recibe parametros: el sistema operativo
// lo llama sin contexto, asi que la unica via es una variable de alcance global.
//
// Y es atomic<bool> y no un bool comun por dos motivos. Uno: la señal puede llegar en
// cualquier instante y en cualquier hilo, asi que escribir la bandera concurre con las
// lecturas del bucle principal, y eso sin atomico es un data race. Dos: el estandar
// solo permite tocar, dentro de un manejador, variables volatile sig_atomic_t o
// atomicos SIN CANDADO. De ahi el static_assert.
std::atomic<bool> g_stopSignal{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "el manejador de señales solo puede tocar atomicos sin candado");

// Lo unico que hace es levantar la bandera.
//
// Un manejador interrumpe al hilo en el punto que sea, incluso teniendo tomado un
// candado. Si aca se escribiera en el log se pediria el mutex del EventLog, y si el
// hilo interrumpido ya lo tenia, el programa se bloquearia para siempre contra si
// mismo, sin nadie que pueda soltarlo. Tampoco se puede imprimir ni pedir memoria.
// Se llama seguridad frente a señales asincronas: levantar una bandera y nada mas.
extern "C" void onStopSignal(int) { g_stopSignal.store(true); }

struct Arguments {
    std::string configPath;
    std::string logPath = "events.log";
    // Corre la simulacion completa, armando cada cuadro, pero sin abrir la ventana.
    // Existe por ThreadSanitizer: Qt lanza sus propios hilos y habla con el escritorio
    // por dbus y glib, y esas dos vienen sin simbolos, asi que ensucian el reporte con
    // carreras ajenas que ni se pueden nombrar para suprimirlas. Sin ventana, lo que
    // queda bajo analisis es exactamente la concurrencia de este programa.
    bool noWindow = false;
};

// Devuelve false (y explica en stderr) si los argumentos no son válidos.
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

// Las coordenadas en pixeles de cada nodo. Se calcula UNA vez: la proyeccion no cambia
// porque la imagen no cambia de tamaño durante la simulacion. Antes cada funcion de
// dibujo rehacia este mapa, es decir 668 hashes de cadena por llamada; hacerlo treinta
// veces por segundo para dibujar couriers habria sido un desperdicio puro.
using PixelMap = std::unordered_map<std::string, cv::Point2f>;

PixelMap buildPixelMap(const Config& cfg, const cv::Size& size) {
    PixelMap px;
    px.reserve(cfg.nodes.size());
    for (const NodeConfig& n : cfg.nodes) {
        px[n.id] = toPixel(n.lat, n.lon, cfg.bounds, size);
    }
    return px;
}

// Dibuja calles (azul = doble sentido, rojo con flecha = un sentido), nodos y el inicio.
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

// Un color por estado (§6.4: los estados tienen que distinguirse). OpenCV usa BGR, asi
// que el primer numero es azul y el ultimo rojo.
cv::Scalar courierColor(CourierState s) {
    switch (s) {
        case CourierState::idle:         return cv::Scalar(150, 150, 150);   // gris
        case CourierState::toPickup:     return cv::Scalar(40, 200, 255);    // amarillo
        case CourierState::atRestaurant: return cv::Scalar(0, 140, 255);     // naranja
        case CourierState::toDropoff:    return cv::Scalar(60, 190, 60);     // verde
        case CourierState::brokenDown:   return cv::Scalar(40, 40, 220);     // rojo
    }
    return cv::Scalar(255, 255, 255);
}

// La posicion del courier en pixeles, interpolada dentro del tramo que esta recorriendo.
//
// Aca esta el requisito del §2: el punto avanza de forma continua mientras el courier
// anda, en vez de saltar de esquina en esquina. El courier no actualiza nada; anoto
// cuando salio y cuando llega, y la posicion se deduce del reloj.
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

// Las entregas activas: una linea del courier a la puerta del cliente (§6.4).
//
// Recibe las copias ya tomadas. Antes esta funcion y drawCouriers pedian cada una su
// propio snapshot(), asi que cada cuadro tomaba y soltaba el candado de cada courier dos
// veces: 16 veces por cuadro con ocho couriers, 480 por segundo. Y peor, las dos podian
// ver estados distintos del mismo courier dentro del mismo cuadro, dibujando la linea
// desde una posicion y el punto en otra.
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

        // Los couriers quietos en el mismo nodo se superpondrian, asi que se abren en
        // abanico. Solo afecta al dibujo: el dato sigue siendo el nodo.
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

    // Leyenda: el mismo color que usa cada estado en el mapa.
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

    // El reloj se crea antes que nada porque marca el instante cero de la simulación:
    // todas las t del log se miden desde acá.
    const SimClock clock(cfg.timeScale);

    // El log se abre antes de lanzar ningún hilo. Si la ruta no sirve, es mejor
    // enterarse ahora que con ocho hilos corriendo.
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

    // Que la ventana quepa en pantalla. Se escala una sola vez, no en cada frame.
    const double kMaxWidth = 1400.0;
    const double kMaxHeight = 950.0;
    const double scale = std::min({1.0, kMaxWidth / image.cols, kMaxHeight / image.rows});
    if (scale < 1.0) {
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // Dos capas. El FONDO -mapa, calles y restaurantes- se dibuja una sola vez, porque
    // no cambia nunca. Lo que se mueve se dibuja encima, en cada cuadro, sobre una copia.
    //
    // Sin esa separacion habria que repintar 1027 calles treinta veces por segundo para
    // mover ocho puntos.
    const PixelMap px = buildPixelMap(cfg, image.size());

    cv::Mat withGraph = image.clone();
    drawGraph(withGraph, cfg, px);
    for (cv::Mat* background : {&image, &withGraph}) {
        drawRestaurants(*background, cfg, px);
    }

    const std::string windowName = "Equipetrol Delivery";
    const int kEscKey = 27;
    const int kFrameMs = 33;
    // El grafo arranca OCULTO. Con 1027 calles y sus flechas, la capa tapa el mapa
    // y los puntos cuestan de distinguir, que es justo lo que el §2 quiere ver. La
    // tecla G lo muestra cuando hace falta inspeccionar la red de calles.
    bool showGraph = false;

    // El grafo se construye una vez, antes de cualquier hilo, y después es inmutable:
    // por eso lo comparten todos los couriers sin ningún candado.
    const CityGraph graph(cfg);

    // El libro y el generador se declaran ANTES que la flota porque los couriers los
    // referencian. En C++ los objetos se destruyen en orden inverso al de creación, así
    // que el libro sobrevive a quienes lo usan.
    OrderBook book(cfg.maxPending, clock, *log);
    OrderGenerator generator(cfg, clock, book);
    // Un monitor por restaurante. Se construyen antes que la flota porque los
    // couriers los referencian a traves del contexto.
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id, std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }

    const SimContext ctx{graph, clock, *log, book, restaurants};

    // La flota. unique_ptr y no vector<CourierActor> por dos razones: el actor tiene un
    // mutex y una condition_variable adentro, así que no es copiable ni movible y el
    // vector no podría reubicarlo al crecer; y otros hilos van a guardar punteros a
    // estos objetos, cuya dirección no puede cambiar.
    std::vector<std::unique_ptr<CourierActor>> fleet;
    fleet.reserve(static_cast<std::size_t>(cfg.couriers));
    for (int i = 0; i < cfg.couriers; ++i) {
        fleet.push_back(std::make_unique<CourierActor>(
            static_cast<CourierId>(i) + 1, cfg.startNode, cfg.bagCapacity, ctx,
            cfg.breakdownProbability, cfg.seed));
    }
    // La flota se conecta recien ahora, con todos construidos: un courier necesita
    // conocer a sus companeros para traspasarles sus ordenes al averiarse, y en el
    // constructor la lista todavia se estaba llenando.
    for (auto& actor : fleet) {
        actor->setFleet(&fleet);
    }

    // Los hilos se lanzan DESPUÉS de construir todos los actores: si se lanzaran dentro
    // del constructor, un hilo podría ver un objeto a medio construir.
    std::vector<std::thread> threads;
    threads.reserve(fleet.size());
    for (auto& actor : fleet) {
        threads.emplace_back(&CourierActor::run, actor.get());
    }
    // El dispatcher se construye DESPUES de la flota, porque guarda una referencia a
    // ella. Su hilo arranca junto con los demas.
    Dispatcher dispatcher(cfg, ctx, fleet);
    std::thread dispatcherThread(&Dispatcher::run, &dispatcher);

    std::thread generatorThread(&OrderGenerator::run, &generator);

    if (!args.noWindow) {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    }
    std::cout << "Simulation has started..." << std::endl;

    std::signal(SIGINT, onStopSignal);
    std::signal(SIGTERM, onStopSignal);

    // durationS esta en segundos SIMULADOS. Con 0 corre hasta recibir una señal.
    const std::int64_t durationMs = cfg.durationS * 1000;

    cv::Mat frame;
    std::vector<Courier> views;   // se reutiliza: no pide memoria por cuadro
    views.reserve(fleet.size());
    while (true) {
        // copyTo y no clone(): reutiliza el buffer de frame en vez de pedir 4 MB al
        // sistema treinta veces por segundo. Por eso frame se declara FUERA del bucle.
        (showGraph ? withGraph : image).copyTo(frame);

        const std::int64_t nowMs = clock.nowMs();

        // Una sola copia por courier y por cuadro. Las dos funciones de dibujo la
        // comparten, asi que ven el mismo instante y el candado de cada courier se
        // toma una vez en vez de dos.
        views.clear();
        for (const auto& actor : fleet) {
            views.push_back(actor->snapshot());
        }
        drawActiveDeliveries(frame, px, ctx, views, nowMs);
        drawCouriers(frame, px, views, nowMs);
        drawPanel(frame, book.stats(), nowMs, cfg.attribution);
        if (args.noWindow) {
            // El cuadro se armo igual -de eso se trata: el codigo de dibujo tambien
            // queda bajo analisis- pero no se muestra. La espera reemplaza a waitKey
            // para que el bucle mantenga el mismo ritmo de treinta cuadros por segundo.
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
        // Las dos condiciones de corte del §7. La señal se nota en un cuadro como
        // mucho, o sea 33 ms: muy por debajo de los 5 segundos que da el enunciado.
        if (g_stopSignal.load()) {
            break;
        }
        if (durationMs > 0 && clock.nowMs() >= durationMs) {
            break;
        }
    }

    // §7: registrar simulationStopping, terminar todos los hilos y no abandonar ninguno.
    Event stopping;
    stopping.name = EventName::kSimulationStopping;
    log->write(stopping);

    // El orden va en cascada, de la fuente hacia el final:
    //   1. el generador deja de crear ordenes,
    //   2. el dispatcher deja de repartir las que quedan,
    //   3. los couriers abandonan el viaje en curso.
    // Al reves, cada etapa seguiria alimentando a la siguiente mientras esta se apaga.
    generator.requestStop();
    generatorThread.join();

    dispatcher.requestStop();
    dispatcherThread.join();

    // Los restaurantes se avisan antes que los couriers: si uno esta haciendo fila,
    // tiene que poder salir de ella para despues atender su propia parada.
    for (auto& entry : restaurants) {
        entry.second->stop();
    }

    // A TODOS primero y recien despues se los espera: pedir y esperar de a uno
    // serializaria un cierre que puede ser simultaneo.
    for (auto& actor : fleet) {
        actor->requestStop();
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Recien ahora, con todos los hilos terminados, se cuentan las que quedaron vivas.
    // Antes seria una carrera: un courier podria entregar una orden justo despues de que
    // la contaramos como pendiente.
    book.markRemainingAsPending();
    const OrderStats stats = book.stats();

    // §6: el resumen es la ULTIMA linea de stdout. Nada se imprime despues.
    std::cout << buildSummary(stats, stats.breakdowns, fleet) << std::endl;

    cv::destroyAllWindows();
    return 0;
}
