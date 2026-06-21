#!/usr/bin/env python3
"""One-command Barbershop launcher.

    python run_dashboard.py

Installs any missing dashboard packages, starts the Barbershop dashboard, and opens your browser at
http://localhost:8050. (The JARVIS HUD needs NO command at all — just double-click jarvis_hud.html.)
"""
import importlib
import subprocess
import sys
import threading
import webbrowser


def _ensure(module: str, pip_name: str) -> None:
    """Import a package; pip-install it (quietly) the first time if it's missing."""
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"· installing {pip_name} …")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)


def main() -> None:
    for module, pip_name in [("dash", "dash>=2.14"), ("plotly", "plotly"), ("pandas", "pandas"),
                             ("numpy", "numpy"), ("dash_cytoscape", "dash-cytoscape>=0.3.0")]:
        _ensure(module, pip_name)

    from barbershop import config
    from barbershop.dashboard import available_source, make_app

    url = f"http://localhost:{config.DASH_PORT}"
    print("\n" + "=" * 60)
    print(f"  Barbershop dashboard starting →  {url}")
    print("  Your browser should open automatically in a second.")
    print("  Press Ctrl+C in this window to stop it.")
    print("=" * 60 + "\n")
    threading.Timer(1.8, lambda: webbrowser.open(url)).start()
    make_app(source=available_source()).run(host=config.DASH_HOST, port=config.DASH_PORT, debug=False)


if __name__ == "__main__":
    main()
