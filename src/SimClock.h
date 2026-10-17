// El reloj de la simulacion: traduce entre tiempo real y tiempo simulado (§5 timeScale).
//
// No guarda "la hora actual" en ninguna variable. Un hilo-reloj que fuera actualizando
// un currentSimMs seria estado mutable compartido por todos los hilos del programa:
// necesitaria candado, seria un punto de contencion por el que pasa todo, y el valor
// leido estaria siempre algo atrasado. En vez de eso, el reloj solo guarda el instante
// de arranque y la escala, los dos fijados en el constructor y nunca modificados, y
// cada consulta le pregunta la hora al sistema operativo.
//
// Por eso es compartible entre todos los hilos sin ningun candado: no hay nada que se
// escriba. steady_clock::now() es thread-safe por norma del estandar.
#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

class SimClock {
public:
    // loadConfig ya garantiza timeScale > 0; el assert documenta y verifica esa
    // precondicion aca, donde dividir por cero daria comportamiento indefinido.
    explicit SimClock(double timeScale)
        : start_(std::chrono::steady_clock::now()), timeScale_(timeScale) {
        assert(timeScale > 0.0);
    }

    // Milisegundos SIMULADOS desde que arranco la simulacion. Es la "t" del log (§6).
    std::int64_t nowMs() const {
        const std::chrono::duration<double, std::milli> real =
            std::chrono::steady_clock::now() - start_;
        return static_cast<std::int64_t>(real.count() * timeScale_);
    }

    // Cuanto tiempo REAL representa una duracion simulada.
    //
    // Es el puente hacia los wait_for de las condition_variable: ahi es donde un courier
    // va a esperar a que transcurra el viaje SIN dormir, de modo que una averia o una
    // señal de parada puedan despertarlo antes.
    // Devuelve microsegundos y no milisegundos a proposito: con timeScale 60, toda
    // duracion simulada menor a 30 ms redondearia a 0 ms reales, y un wait_for de cero
    // dentro de un bucle se vuelve un bucle que gira consumiendo CPU, que es justo lo
    // que prohibe el §4. En microsegundos, 1 ms simulado a escala 60 son 17 us: chico,
    // pero no cero.
    std::chrono::microseconds realDuration(std::int64_t simMs) const {
        return std::chrono::microseconds(
            std::llround(static_cast<double>(simMs) * 1000.0 / timeScale_));
    }

    // Representa el paso de tiempo simulado durmiendo el tiempo real equivalente.
    //
    // El §4 permite dormir SOLO para esto o para el ritmo de los frames. Dormir para
    // esperar a que otro hilo este listo es un defecto, y un bucle que gire revisando
    // una condicion tambien.
    void sleepSim(std::int64_t simMs) const {
        std::this_thread::sleep_for(realDuration(simMs));
    }

    double timeScale() const { return timeScale_; }

private:
    const std::chrono::steady_clock::time_point start_;
    const double timeScale_;
};
