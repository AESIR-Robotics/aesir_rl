import argparse
import os
import shutil
import subprocess
import sys
import time

DEFAULT_XML = "/home/aesir/aesir_rl2/aesir_rl/models/aesir_complete.xml"
DEFAULT_SIMULATE = ""  # deja vacío para que busque "simulate" en PATH


def resolve_simulate_command(simulate_bin, xml_path):
    if simulate_bin:
        path = os.path.expanduser(simulate_bin)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return [path, xml_path]
        if os.path.isfile(path) and path.endswith(".py"):
            return [sys.executable, path, xml_path]

    which = shutil.which("simulate")
    if which:
        return [which, xml_path]

    # Fallback: el paquete pip de mujoco no trae el binario "simulate" compilado,
    # pero SÍ trae la misma GUI completa accesible vía "python -m mujoco.viewer".
    # La lanzamos como subprocess (no como llamada bloqueante) para que el
    # watcher pueda terminarla y relanzarla normalmente.
    return [sys.executable, "-m", "mujoco.viewer", f"--mjcf={xml_path}"]


def launch_viewer(xml_path, simulate_bin):
    cmd = resolve_simulate_command(simulate_bin, xml_path)
    print("Lanzando viewer con:", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(xml_path)))


def watch_and_reload(simulate_bin, xml_path, interval=0.5):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"No existe el XML: {xml_path}")

    proc = None
    last_mtime = None

    while True:
        try:
            current_mtime = os.path.getmtime(xml_path)

            if proc is None or proc.poll() is not None:
                proc = launch_viewer(xml_path, simulate_bin)
                last_mtime = current_mtime

            elif current_mtime != last_mtime:
                print("Se detectó cambio en el XML. Reiniciando...")
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

                proc = launch_viewer(xml_path, simulate_bin)
                last_mtime = current_mtime

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nSaliendo...")
            if proc is not None and proc.poll() is None:
                proc.terminate()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de MuJoCo con auto-reload")
    parser.add_argument("xml_path", nargs="?", default=DEFAULT_XML)
    parser.add_argument("--simulate-bin", default=DEFAULT_SIMULATE)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    watch_and_reload(args.simulate_bin, args.xml_path, interval=args.interval)