#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter, defaultdict

CAMPOS = {
    "orderCreated": {"order", "restaurant"},
    "orderAssigned": {"order", "courier"},
    "pickupStarted": {"order", "courier", "restaurant"},
    "orderPickedUp": {"order", "courier", "restaurant"},
    "orderDelivered": {"order", "courier"},
    "orderRejected": {"order", "reason"},
    "courierBreakdown": {"courier"},
    "courierJoined": {"courier"},
    "courierRetired": {"courier"},
    "orderReassigned": {"order", "from", "to"},
    "simulationStopping": set(),
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

def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    ruta_log, ruta_salida = argv[1], argv[2]

    errores = []
    eventos = Counter()
    estado = {}
    ultima_t = {}
    entregas = Counter()
    vivas = set()
    refuerzos_unidos = set()
    refuerzos_retirados = set()
    simulacion_parando = False

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

            nombre = e.get("event")
            if nombre not in CAMPOS:
                errores.append(f"linea {n}: evento desconocido {nombre!r}")
                continue
            eventos[nombre] += 1

            if simulacion_parando and nombre == "orderCreated":
                errores.append(f"linea {n}: se creo una orden tras simulationStopping")
            if nombre == "simulationStopping":
                simulacion_parando = True

            if not isinstance(e.get("t"), int):
                errores.append(f"linea {n}: falta t o no es entero")
            extra = set(e) - {"t", "event"} - CAMPOS[nombre]
            if extra:
                errores.append(f"linea {n}: {nombre} lleva campos de mas: {sorted(extra)}")

            for campo in ("order", "courier"):
                if campo in e and not isinstance(e[campo], str):
                    errores.append(
                        f"linea {n}: {campo} tiene que ser cadena, es {type(e[campo]).__name__}")

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

            if nombre == "orderRejected" and e.get("reason") not in MOTIVOS:
                errores.append(f"linea {n}: motivo de rechazo invalido {e.get('reason')!r}")

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
                entregas[e["courier"]] += 1

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

    obligatorios = ["created", "delivered", "rejected", "pending", "reassigned",
                    "breakdowns", "avgDeliveryMs", "maxDeliveryMs", "deliveredPerCourier",
                    "reinforcementsJoined", "reinforcementsRetired"]
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
    if resumen["reinforcementsJoined"] != eventos["courierJoined"]:
        errores.append("resumen: reinforcementsJoined no coincide con courierJoined")
    if resumen["reinforcementsRetired"] != eventos["courierRetired"]:
        errores.append("resumen: reinforcementsRetired no coincide con courierRetired")

    if resumen["pending"] != len(vivas):
        errores.append(f"resumen: pending={resumen['pending']} pero quedaron {len(vivas)} "
                       f"ordenes sin estado final en el log")

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
