import os
import socket
import base64
import time

import numpy as np

# Forzar backend X11 en Wayland (KDE Plasma / CachyOS)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2

# ========= Configuración =========
ROBOT_IP   = "192.168.0.103"  # IP del TurtleBot4
ROBOT_PORT = 6000              # Debe coincidir con el nodo de telemetría

DESIRED_DOMAIN_ID = 4          # Debe coincidir con ROS_DOMAIN_ID del robot
PAIRING_CODE      = "ROBOT_A_42"
EXPECTED_ROBOT_NAME = "turtlebot4_lite_1"  # por seguridad extra


def do_handshake(sock: socket.socket, robot_addr):
    sock.settimeout(1.0)
    print(f"[HANDSHAKE] Iniciando con {robot_addr}...")
    while True:
        # Enviar HELLO <domain> <pairing_code>
        msg = f"HELLO {DESIRED_DOMAIN_ID} {PAIRING_CODE}".encode("utf-8")
        sock.sendto(msg, robot_addr)

        try:
            data, addr = sock.recvfrom(4096)
            text = data.decode("utf-8").strip()
            parts = text.split()

            if len(parts) >= 3 and parts[0] == "ACK":
                domain_str = parts[1]
                robot_name = " ".join(parts[2:])

                print(f"[HANDSHAKE] Recibido: '{text}' desde {addr}")

                try:
                    domain_id = int(domain_str)
                except ValueError:
                    print("[HANDSHAKE] domain_id inválido, reintentando...")
                    continue

                if domain_id != DESIRED_DOMAIN_ID:
                    print(f"[HANDSHAKE] ROS_DOMAIN_ID no coincide "
                          f"(esperado={DESIRED_DOMAIN_ID}, recibido={domain_id}). Reintentando...")
                    continue

                if robot_name != EXPECTED_ROBOT_NAME:
                    print(f"[HANDSHAKE] robot_name no coincide "
                          f"(esperado={EXPECTED_ROBOT_NAME}, recibido={robot_name}). Reintentando...")
                    continue

                print(f"[HANDSHAKE] Emparejado con '{robot_name}' (domain {domain_id}).")
                sock.settimeout(0.033)  # 33ms para no bloquear el event loop de OpenCV
                return
            else:
                print(f"[HANDSHAKE] Mensaje inesperado: '{text}', reintentando...")

        except socket.timeout:
            print("[HANDSHAKE] Timeout esperando ACK, reintentando...")

        except KeyboardInterrupt:
            print("[HANDSHAKE] Cancelado por el usuario.")
            raise


def handle_scan(parts):
    """
    parts: lista de strings del mensaje:
    SCAN <domain_id> <robot_name> <sec> <nsec> <angle_min> <angle_inc> <n> r1 ... rn
    """
    if len(parts) < 8:
        print("[SCAN] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])
        angle_min = float(parts[5])
        angle_inc = float(parts[6])
        n = int(parts[7])

        ranges_str = parts[8:]
        if len(ranges_str) != n:
            print(f"[SCAN] n={n} pero llegaron {len(ranges_str)} rangos. Usando min(len, n).")
        n_effective = min(n, len(ranges_str))

        ranges = [float(r) for r in ranges_str[:n_effective]]

        # Aquí puedes hacer lo que quieras con el LIDAR.
        # Demo: imprimir algunos valores cada vez.
        print(f"[SCAN] robot={robot_name} domain={domain_id} "
              f"t={sec}.{nsec:09d} n={n_effective} "
              f"ejemplo={ranges[:5]}")

    except ValueError as e:
        print(f"[SCAN] Error parseando mensaje: {e}")


def handle_img(parts):
    """
    parts: lista de strings del mensaje:
    IMG <domain_id> <robot_name> <sec> <nsec> <base64_jpeg>
    Como base64 puede tener espacios si algo raro pasa, juntamos desde índice 5.
    """
    if len(parts) < 6:
        print("[IMG] Mensaje demasiado corto.")
        return

    try:
        domain_id = int(parts[1])
        robot_name = parts[2]
        sec = int(parts[3])
        nsec = int(parts[4])

        b64_str = " ".join(parts[5:])  # el resto del mensaje
        jpeg_bytes = base64.b64decode(b64_str)

        # Decodificar JPEG a imagen OpenCV
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[IMG] Error al decodificar imagen.")
            return

        # Mostrar con OpenCV
        cv2.imshow(f"Camara {robot_name} (domain {domain_id})", img)
        cv2.waitKey(1)

    except Exception as e:
        print(f"[IMG] Error manejando imagen: {e}")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # IMPORTANTE: el cliente puede usar cualquier puerto local
    # Si quieres forzar uno: sock.bind(("0.0.0.0", 6001))

    robot_addr = (ROBOT_IP, ROBOT_PORT)

    # 1) Handshake
    do_handshake(sock, robot_addr)

    print("[MAIN] Recibiendo telemetría. Ctrl+C para salir.")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)  # tamaño grande para imagen
                text = data.decode("utf-8", errors="ignore")
                parts = text.split()

                if not parts:
                    continue

                msg_type = parts[0]

                if msg_type == "SCAN":
                    handle_scan(parts)
                elif msg_type == "IMG":
                    handle_img(parts)
                else:
                    print(f"[MAIN] Mensaje desconocido desde {addr}: '{msg_type}'")
            except socket.timeout:
                pass  # sin datos, seguimos al waitKey

            # Procesar eventos de ventana OpenCV en cada iteración
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[MAIN] Cerrando...")
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
