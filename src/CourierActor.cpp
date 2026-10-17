#include "CourierActor.h"

CourierActor::CourierActor(CourierId id, const std::string &startNode,
                           int bagCapacity)
    : id_(id), state_(id, startNode, bagCapacity) {}

void CourierActor::run() {
  std::unique_lock<std::mutex> lock(mutex_);

  // Todavia no hay ordenes que repartir, asi que el courier solo espera a que
  // le pidan terminar. wait() LIBERA el candado y deja el hilo dormido: no
  // consume CPU y mientras tanto otros hilos pueden tomar el candado (por
  // ejemplo el renderer llamando a snapshot()). Al despertar, wait() vuelve a
  // tomar el candado solo.
  //
  // El predicado no es un adorno: una condition_variable puede despertar sola,
  // sin que nadie la haya notificado (despertar espurio). wait(lock, pred)
  // vuelve a dormirse si el predicado da falso, asi que el hilo solo sigue
  // cuando la condicion se cumple de verdad.
  //
  // En el commit siguiente esto pasa a ser un while, y el predicado tambien
  // devuelve true cuando hay trabajo en la bolsa.
  cv_.wait(lock, [this] { return stopRequested_; });
}

void CourierActor::requestStop() {
  {
    // La bandera se escribe CON el candado tomado. Si se escribiera sin el,
    // podria pasar que el hilo que espera evalue el predicado (falso), y antes
    // de dormirse ocurra la escritura y la notificacion: el aviso se perderia y
    // el hilo quedaria dormido para siempre. Es el "lost wakeup", y el candado
    // es lo que lo impide.
    const std::lock_guard<std::mutex> guard(mutex_);
    stopRequested_ = true;
  }
  // El notify va FUERA del candado: si se hiciera adentro, el hilo que
  // despierta tendria que volver a dormirse de inmediato esperando el candado
  // que todavia tenemos nosotros. notify_one y no notify_all porque en esta cv
  // espera un solo hilo: el de este courier.
  cv_.notify_one();
}

Courier CourierActor::snapshot() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  return state_; // se copia adentro del candado y se devuelve afuera
}
