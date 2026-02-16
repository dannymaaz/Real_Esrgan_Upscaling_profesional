"""
Instalador multiplataforma para Real-ESRGAN Upscaling Profesional.

Uso recomendado:
  python setup_environment.py
  python setup_environment.py --torch cu118 --with-face
  python setup_environment.py --skip-models
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run(cmd):
    print(f"\n[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Setup multiplataforma de dependencias")
    parser.add_argument(
        "--torch",
        choices=["cpu", "cu118"],
        default="cpu",
        help="Variante de PyTorch a instalar"
    )
    parser.add_argument(
        "--with-face",
        action="store_true",
        help="Instalar dependencias opcionales de GFPGAN"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="No descargar modelos al finalizar"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    python_exe = sys.executable
    pip_cmd = [python_exe, "-m", "pip"]
    os_name = platform.system().lower()
    machine = platform.machine().lower()

    print("=" * 64)
    print("Setup Real-ESRGAN Upscaling Profesional")
    print(f"Plataforma detectada: {platform.system()} ({machine})")
    print("=" * 64)

    if os_name == "darwin" and args.torch == "cu118":
        print("[info] macOS no usa CUDA NVIDIA; cambiando automaticamente a --torch cpu")
        args.torch = "cpu"

    run(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    run(pip_cmd + ["install", "-r", str(BASE_DIR / "requirements" / "base.txt")])

    if args.torch == "cu118":
        run(
            pip_cmd
            + [
                "install",
                "-r",
                str(BASE_DIR / "requirements" / "torch-cu118.txt"),
                "--index-url",
                "https://download.pytorch.org/whl/cu118",
            ]
        )
    else:
        run(pip_cmd + ["install", "-r", str(BASE_DIR / "requirements" / "torch-cpu.txt")])

    if args.with_face:
        run(pip_cmd + ["install", "-r", str(BASE_DIR / "requirements" / "face.txt")])
    else:
        print("[info] Instalacion sin GFPGAN (modo rapido).")
        print("[info] Si luego quieres mejora facial: python setup_environment.py --with-face --skip-models")

    if not args.skip_models:
        model_cmd = [python_exe, str(BASE_DIR / "download_models.py")]
        if not args.with_face:
            model_cmd.append("--skip-face")
        run(model_cmd)

    print("\n[ok] Setup completado.")
    print("[ok] Inicia la app con: python run.py")


if __name__ == "__main__":
    main()
