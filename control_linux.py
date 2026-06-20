import socket
import struct
import time
import sys
import select
import termios
import tty

ROBOT_IP = "192.168.0.100"
ROBOT_PORT = 5007

SEND_HZ = 60  # un poco más alto, igual tu receiver publica a 50Hz

# Velocidades
LIN = 0.99     # m/s
ANG = 3.00     # rad/s

def pack_cmd(v, w):
    return struct.pack("ff", float(v), float(w))

def get_key_nonblocking():
    """Lee una tecla sin bloquear en Linux."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    v = 0.0
    w = 0.0

    period = 1.0 / SEND_HZ
    last_send = 0.0

    # Guardar configuración actual de terminal
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print("=== UDP Teleop FAST (Linux) ===")
    print(f"Target: {ROBOT_IP}:{ROBOT_PORT} @ {SEND_HZ} Hz")
    print("W: forward | S: back | A: left | D: right | X: stop | Q: quit")
    print("---------------------------------")

    try:
        # Modo cbreak: permite leer teclas sin Enter
        tty.setcbreak(fd)

        while True:
            key = get_key_nonblocking()
            if key is not None:
                key = key.lower()

                if key == "w":
                    v, w = +LIN, 0.0
                elif key == "s":
                    v, w = -LIN, 0.0
                elif key == "a":
                    v, w = 0.0, +ANG
                elif key == "d":
                    v, w = 0.0, -ANG
                elif key == "x":
                    v, w = 0.0, 0.0
                elif key == "q":
                    sock.sendto(pack_cmd(0.0, 0.0), (ROBOT_IP, ROBOT_PORT))
                    print("\nSaliendo (STOP enviado).")
                    return

                print(f"\r v={v:+.2f} m/s | w={w:+.2f} rad/s     ", end="", flush=True)

            now = time.time()
            if now - last_send >= period:
                sock.sendto(pack_cmd(v, w), (ROBOT_IP, ROBOT_PORT))
                last_send = now

            time.sleep(0.001)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sock.close()
        print("\nTerminal restaurada.")
        
if __name__ == "__main__":
    main()
