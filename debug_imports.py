import sys
from pathlib import Path

def test_imports():
    print("Iniciando prueba de dependencias...")
    try:
        import fastapi
        print("[OK] fastapi")
        import uvicorn
        print("[OK] uvicorn")
        import torch
        print(f"[OK] torch ({torch.__version__})")
        import torchvision
        print(f"[OK] torchvision ({torchvision.__version__})")
        import cv2
        print(f"[OK] cv2 ({cv2.__version__})")
        import realesrgan
        print("[OK] realesrgan")
        import gfpgan
        print("[OK] gfpgan")
        return True
    except Exception as e:
        print(f"\n[!] ERROR DETECTADO:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
