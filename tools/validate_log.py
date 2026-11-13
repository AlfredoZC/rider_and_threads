#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter, defaultdict

CAMPOS = {
    "orderCreated": {"order", "restaurant", "urgent"},
    "orderAssigned": {"order", "courier"},
    "pickupStarted": {"order", "courier", "restaurant"},
    "orderPickedUp": {"order", "courier", "restaurant"},
    "orderDelivered": {"order", "courier"},
    "orderRejected": {"order", "reason"},
    "courierBreakdown": {"courier"},
    "courierJoined": {"courier"},
    "courierRetired": {"courier"},
    "orderReassigned": {"order", "from", "to"},
    "courierBreakStarted": {"courier"},
    "courierBreakEnded": {"courier"},
    "simulationStopping": set(),
    "restaurantPaused": {"restaurant"},
    "restaurantResumed": {"restaurant"},
}

LLEVA_A = {
    "orderCreated": "created",
    "orderAssigned": "assigned",
    "orderPickedUp": "pickedUp",
    "orderDelivered": "delivered",
    "orderRejected": "rejected",
}

LEGALES = {
    "created": {"assigned", "rejected"},
    "assigned": {"pickedUp", "created"},
    "pickedUp": {"delivered"},
}

FINALES = {"delivered", "rejected", "pending"}

MOTIVOS = {"queueFull", "unreachable", "noCourier"}

def calendario(ruta_config):
    """Devuelve (P, D, {restaurante: (indice, total)}) o None si no hay bloque kitchen."""
    with open(ruta_config, encoding="utf-8") as f:
        cfg = json.load(f)
    cocina = cfg.get("kitchen")
    if cocina is None:
        return None
    ids = [r["id"] for r in cfg["restaurants"]]
    return cocina["pauseEveryMs"], cocina["pauseMs"], {rid: (i, len(ids)) for i, rid in enumerate(ids)}


# Cada valor sale escapado: barra, espacio, `=` y controles como \xHH (hex en minuscula); el resto
# literal. Una barra o un `=` sin escapar dentro de un valor es un formato invalido.
CAMPO_FILA = r"((?:[^\s\\=]|\\x[0-9a-f]{2})+)"
FILA = re.compile(r"^\[fila\] t=(-?\d+) courier=" + CAMPO_FILA + " order=" + CAMPO_FILA +
                  " restaurant=" + CAMPO_FILA + r" pausa=(si|no)$")
ESCAPE_FILA = re.compile(r"\\x([0-9a-f]{2})")


def desescapar(valor):
    r"""Revierte el escape de la linea [fila] (`\xHH`: barra, espacio, =, control)."""
    return ESCAPE_FILA.sub(lambda m: chr(int(m.group(1), 16)), valor)


def leer_fila(ruta_stderr):
    """Lee las lineas `[fila] ...` de stderr. Devuelve (llegadas por restaurante, con pausa, errores)."""
    llegadas = defaultdict(list)
    con_pausa = 0
    errores = []
    with open(ruta_stderr, encoding="utf-8", errors="replace") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.rstrip("\r\n")
            if not linea.startswith("[fila]"):
                continue
            m = FILA.match(linea)
            if m is None:
                errores.append(f"stderr linea {n}: formato [fila] invalido: {linea[:100]!r}")
                continue
            llegadas[desescapar(m.group(4))].append(desescapar(m.group(3)))
            if m.group(5) == "si":
                con_pausa += 1
    return llegadas, con_pausa, errores


def main(argv):
    if len(argv) not in (3, 4, 5):
        print("uso: validate_log.py <log> <stdout> [config.json] [stderr]")
        return 2
    ruta_log, ruta_salida = argv[1], argv[2]
    cal = calendario(argv[3]) if len(argv) >= 4 and argv[3] not in ("", "-") else None
    ruta_stderr = argv[4] if len(argv) == 5 else None
    en_pausa = {}
    pausas_de = Counter()
    iniciados = defaultdict(list)

    errores = []
    eventos = Counter()
    estado = {}
    ultima_t = {}
    entregas = Counter()
    vivas = set()
    refuerzos_unidos = set()
    refuerzos_retirados = set()
    simulacion_parando = False
    en_descanso = set()
    descansos = Counter()
    urgentes = set()
    creada_t = {}
    urgentes_entregados = 0
    suma_urgentes_ms = 0

    with open(ruta_log, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                e = json.loads(linea)
            except json.JSONDecodeError as exc:
                errores.append(f"linea {n}: no es JSON valido ({exc})")
                continue

            if not isinstance(e, dict):
                errores.append(f"linea {n}: la linea no es un objeto JSON")
                continue

            nombre = e.get("event")
            if not isinstance(nombre, str) or nombre not in CAMPOS:
                errores.append(f"linea {n}: evento desconocido {nombre!r}")
                continue
            eventos[nombre] += 1

            if simulacion_parando and nombre == "orderCreated":
                errores.append(f"linea {n}: se creo una orden tras simulationStopping")
            if nombre == "simulationStopping":
                simulacion_parando = True

            t_valida = isinstance(e.get("t"), int) and not isinstance(e.get("t"), bool)
            if not t_valida:
                errores.append(f"linea {n}: falta t o no es entero")

            rest = e.get("restaurant")
            if nombre == "restaurantPaused":
                if rest in en_pausa:
                    errores.append(f"linea {n}: {rest} se pausa estando ya en pausa")
                en_pausa[rest] = e.get("t")
                if simulacion_parando:
                    errores.append(f"linea {n}: restaurantPaused despues de simulationStopping")
                if cal is not None:
                    P, D, orden_json = cal
                    if rest not in orden_json:
                        errores.append(f"linea {n}: restaurante {rest} no esta en la config")
                    else:
                        i, total = orden_json[rest]
                        esperado = pausas_de[rest] * P + (i * P) // total
                        if e.get("t") != esperado:
                            errores.append(f"linea {n}: {rest} se pausa en t={e.get('t')}, "
                                           f"el calendario dice {esperado}")
            elif nombre == "restaurantResumed":
                if rest not in en_pausa:
                    errores.append(f"linea {n}: {rest} se reanuda sin estar en pausa")
                else:
                    inicio = en_pausa.pop(rest)
                    if simulacion_parando:
                        errores.append(f"linea {n}: restaurantResumed despues de simulationStopping")
                    if cal is not None and e.get("t") != inicio + cal[1]:
                        errores.append(f"linea {n}: {rest} se reanuda en t={e.get('t')}, "
                                       f"el calendario dice {inicio + cal[1]}")
                    pausas_de[rest] += 1
            elif nombre == "pickupStarted":
                if rest in en_pausa:
                    errores.append(f"linea {n}: pickupStarted de {e.get('order')} en {rest} "
                                   f"dentro de una pausa")
                iniciados[rest].append(e.get("order"))
            extra = set(e) - {"t", "event"} - CAMPOS[nombre]
            if extra:
                errores.append(f"linea {n}: {nombre} lleva campos de mas: {sorted(extra)}")

            tipos_validos = t_valida
            for campo in ("order", "courier"):
                if campo in e and not isinstance(e[campo], str):
                    errores.append(
                        f"linea {n}: {campo} tiene que ser cadena, es {type(e[campo]).__name__}")
                    tipos_validos = False
            if not tipos_validos:
                # sin t entero o con ids mal tipados no se pueden aplicar las reglas
                # de estado/tiempo: queda reportado y se sigue con la linea siguiente
                continue

            if nombre == "orderReassigned":
                for campo in ("from", "to"):
                    valor = e.get(campo, "")
                    if not (isinstance(valor, str) and re.fullmatch(r"[cr][0-9]+", valor)):
                        errores.append(
                            f"linea {n}: {campo}={valor!r} tiene que ser un id de courier")
                if e.get("from") == e.get("to"):
                    errores.append(f"linea {n}: un courier se traspasa a si mismo")

            for campo in ("courier", "from", "to"):
                valor = e.get(campo)
                if isinstance(valor, str) and re.fullmatch(r"r[0-9]+", valor):
                    if nombre == "courierJoined" and campo == "courier":
                        if valor in refuerzos_unidos:
                            errores.append(f"linea {n}: {valor} se unio dos veces")
                        refuerzos_unidos.add(valor)
                    elif valor not in refuerzos_unidos:
                        errores.append(f"linea {n}: {valor} aparece antes de courierJoined")
                    elif valor in refuerzos_retirados:
                        errores.append(f"linea {n}: {valor} aparece despues de courierRetired")
                    elif nombre == "courierRetired" and campo == "courier":
                        refuerzos_retirados.add(valor)

            if nombre == "courierBreakStarted":
                c = e.get("courier")
                if c in en_descanso:
                    errores.append(f"linea {n}: {c} empieza un descanso estando en descanso")
                en_descanso.add(c)
                descansos[c] += 1
            elif nombre == "courierBreakEnded":
                c = e.get("courier")
                if c not in en_descanso:
                    errores.append(f"linea {n}: {c} termina un descanso que no empezo")
                en_descanso.discard(c)
            else:
                if nombre == "orderReassigned":
                    involucrado = e.get("to")
                else:
                    involucrado = e.get("courier")
                if isinstance(involucrado, str) and involucrado in en_descanso:
                    errores.append(
                        f"linea {n}: {nombre} de {involucrado} durante su descanso")

            if nombre == "orderRejected" and e.get("reason") not in MOTIVOS:
                errores.append(f"linea {n}: motivo de rechazo invalido {e.get('reason')!r}")

            if nombre == "orderCreated":
                if not isinstance(e.get("urgent"), bool):
                    errores.append(f"linea {n}: orderCreated necesita urgent booleano")
                elif e["urgent"]:
                    urgentes.add(e.get("order"))
                if isinstance(e.get("t"), int):
                    creada_t[e.get("order")] = e["t"]
            if nombre == "orderDelivered" and e.get("order") in urgentes:
                urgentes_entregados += 1
                suma_urgentes_ms += e["t"] - creada_t.get(e.get("order"), e["t"])

            orden = e.get("order")
            if orden is None:
                continue

            if orden in ultima_t and e["t"] < ultima_t[orden]:
                errores.append(
                    f"linea {n}: la t de {orden} retrocedio de {ultima_t[orden]} a {e['t']}")
            ultima_t[orden] = e["t"]

            destino = LLEVA_A.get(nombre)
            if destino is None:
                continue
            actual = estado.get(orden)
            if actual is None:
                if destino != "created":
                    errores.append(f"linea {n}: {orden} empieza en {destino}, no en created")
                vivas.add(orden)
            elif actual in FINALES:
                errores.append(
                    f"linea {n}: {orden} ya estaba en {actual}, estado final, y pasa a {destino}")
            elif destino not in LEGALES.get(actual, set()):
                errores.append(f"linea {n}: {orden} pasa de {actual} a {destino}, que no es legal")
            estado[orden] = destino
            if destino in FINALES:
                vivas.discard(orden)
            if nombre == "orderDelivered":
                entregas[e.get("courier")] += 1

    with open(ruta_salida, encoding="utf-8") as f:
        lineas = [l.strip() for l in f if l.strip()]
    if not lineas:
        print("ERROR: la salida estandar esta vacia")
        return 1
    try:
        resumen = json.loads(lineas[-1])
    except json.JSONDecodeError as exc:
        print(f"ERROR: la ultima linea de stdout no es JSON ({exc}): {lineas[-1][:120]}")
        return 1

    if not isinstance(resumen, dict):
        print("ERROR: la ultima linea de stdout no es un objeto JSON")
        return 1

    obligatorios = ["created", "delivered", "rejected", "pending", "reassigned",
                    "breakdowns", "avgDeliveryMs", "maxDeliveryMs", "deliveredPerCourier",
                    "reinforcementsJoined", "reinforcementsRetired", "breaks",
                    "breaksPerCourier", "pauses", "pickupsDelayedByPause"]
    for campo in obligatorios:
        if campo not in resumen:
            errores.append(f"resumen: falta el campo {campo}")
    if errores and any("falta el campo" in x for x in errores):
        for x in errores:
            print("  " + x)
        return 1

    suma = resumen["delivered"] + resumen["rejected"] + resumen["pending"]
    if resumen["created"] != suma:
        errores.append(
            f"resumen: created={resumen['created']} pero delivered+rejected+pending={suma}")

    if resumen["delivered"] != eventos["orderDelivered"]:
        errores.append(f"resumen: delivered={resumen['delivered']} pero el log tiene "
                       f"{eventos['orderDelivered']} lineas orderDelivered")
    if resumen["rejected"] != eventos["orderRejected"]:
        errores.append(f"resumen: rejected={resumen['rejected']} pero el log tiene "
                       f"{eventos['orderRejected']} lineas orderRejected")
    if resumen["reassigned"] != eventos["orderReassigned"]:
        errores.append(f"resumen: reassigned={resumen['reassigned']} pero el log tiene "
                       f"{eventos['orderReassigned']} lineas orderReassigned")
    if resumen["breakdowns"] != eventos["courierBreakdown"]:
        errores.append(f"resumen: breakdowns={resumen['breakdowns']} pero el log tiene "
                       f"{eventos['courierBreakdown']} lineas courierBreakdown")
    if resumen["breaks"] != eventos["courierBreakStarted"]:
        errores.append(f"resumen: breaks={resumen['breaks']} pero el log tiene "
                       f"{eventos['courierBreakStarted']} lineas courierBreakStarted")
    por_descanso = resumen["breaksPerCourier"]
    if set(por_descanso) != set(resumen["deliveredPerCourier"]):
        errores.append("resumen: breaksPerCourier y deliveredPerCourier tienen claves distintas")
    if sum(por_descanso.values()) != resumen["breaks"]:
        errores.append("resumen: breaksPerCourier no suma breaks")
    for courier, cuantos in descansos.items():
        if por_descanso.get(courier, 0) != cuantos:
            errores.append(f"resumen: {courier} descanso {cuantos} veces segun el log pero "
                           f"{por_descanso.get(courier, 0)} segun el resumen")
    if resumen["reinforcementsJoined"] != eventos["courierJoined"]:
        errores.append("resumen: reinforcementsJoined no coincide con courierJoined")
    if resumen["reinforcementsRetired"] != eventos["courierRetired"]:
        errores.append("resumen: reinforcementsRetired no coincide con courierRetired")

    if resumen["pauses"] != eventos["restaurantPaused"]:
        errores.append(f"resumen: pauses={resumen['pauses']} pero el log tiene "
                       f"{eventos['restaurantPaused']} lineas restaurantPaused")
    demoradas = resumen["pickupsDelayedByPause"]
    if not isinstance(demoradas, int) or isinstance(demoradas, bool) or demoradas < 0:
        errores.append(f"resumen: pickupsDelayedByPause={demoradas!r} tiene que ser un entero >= 0")
    if not isinstance(resumen["pauses"], int) or isinstance(resumen["pauses"], bool):
        errores.append(f"resumen: pauses={resumen['pauses']!r} tiene que ser un entero")

    if ruta_stderr is not None:
        llegadas, con_pausa, errs = leer_fila(ruta_stderr)
        errores.extend(errs)
        for rest, ordenes in iniciados.items():
            esperado = llegadas.get(rest, [])
            if ordenes != esperado[:len(ordenes)]:
                errores.append(f"{rest}: los pickupStarted {ordenes[:8]}... no siguen el orden de "
                               f"llegada de stderr {esperado[:8]}...")
        if isinstance(demoradas, int) and demoradas < con_pausa:
            errores.append(f"resumen: pickupsDelayedByPause={demoradas} es menor que las "
                           f"{con_pausa} llegadas con pausa=si de stderr")
        print(f"stderr: {sum(len(v) for v in llegadas.values())} llegadas a fila, "
              f"{con_pausa} con pausa=si")

    if resumen["pending"] != len(vivas):
        errores.append(f"resumen: pending={resumen['pending']} pero quedaron {len(vivas)} "
                       f"ordenes sin estado final en el log")

    prom_urg = suma_urgentes_ms // urgentes_entregados if urgentes_entregados else 0
    for clave, esperado in (("urgentCreated", len(urgentes)),
                            ("urgentDelivered", urgentes_entregados),
                            ("avgUrgentDeliveryMs", prom_urg)):
        if resumen.get(clave) != esperado:
            errores.append(f"resumen: {clave}={resumen.get(clave)} pero el log da {esperado}")

    por_courier = resumen["deliveredPerCourier"]
    if sum(por_courier.values()) != resumen["delivered"]:
        errores.append(f"resumen: deliveredPerCourier suma {sum(por_courier.values())} "
                       f"pero delivered={resumen['delivered']}")
    for courier, cuantas in entregas.items():
        if por_courier.get(courier, 0) != cuantas:
            errores.append(f"resumen: {courier} entrego {cuantas} segun el log pero "
                           f"{por_courier.get(courier, 0)} segun el resumen")

    if eventos["simulationStopping"] != 1:
        errores.append(f"log: simulationStopping aparece {eventos['simulationStopping']} "
                       f"veces, tiene que aparecer una")

    print(f"lineas de log analizadas: {sum(eventos.values())}")
    for nombre in sorted(eventos):
        print(f"  {nombre:20} {eventos[nombre]}")
    print(f"resumen: {json.dumps(resumen, ensure_ascii=False)}")

    if errores:
        print(f"\n{len(errores)} PROBLEMA(S):")
        for x in errores[:40]:
            print("  " + x)
        return 1
    print("\nTODO CORRECTO: el log es valido y el resumen coincide con el")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
