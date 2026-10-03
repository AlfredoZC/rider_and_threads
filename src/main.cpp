// Fase 0: ventana mínima para comprobar que OpenCV se enlaza y que la ventana abre.
// En las próximas fases este archivo crece: config, hilos, render y apagado.
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    const std::string windowName = "Equipetrol Delivery";
    const int kEscKey = 27;
    const int kFrameMs = 33;   // ~30 fps: el único "sleep" permitido es el ritmo de frames

    // Imagen negra de 800x600 en color (alto, ancho, 3 canales de 8 bits)
    cv::Mat frame(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    // Las fuentes Hershey de OpenCV no tienen el símbolo ©, por eso "(c)"
    cv::putText(frame, "(c) OpenStreetMap contributors", cv::Point(470, 585),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, frame);

    // std::endl hace flush: la línea sale aunque stdout esté redirigido a un archivo
    std::cout << "Simulation has started..." << std::endl;

    while (true) {
        cv::imshow(windowName, frame);
        int key = cv::waitKey(kFrameMs);   // espera hasta 33 ms y procesa eventos de la ventana
        if (key == kEscKey) {
            break;
        }
        // Si el usuario cierra la ventana con la X, también terminamos
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
