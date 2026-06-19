# ▶️ How to Run It (start here)

Two things to run. Pick what you want — the **JARVIS HUD needs zero setup**.

---

## 1. 🛰️ The JARVIS HUD — the easiest (no install, ~10 seconds)

The HUD is a single web page. There is **nothing to install** and **no command to type**.

1. Get the file **`jarvis_hud.html`** (it's in the repo root; I also sent it to you in chat — just
   **download it**).
2. **Double-click it.** It opens in your web browser (Chrome, Edge, Safari, Firefox — any).
3. Watch the boot sequence, then the HUD powers on. The top-right pill says **`SIM`** — that means it's
   showing **demo data** so you can see everything moving. **That's normal.**

That's it. To make it show **real** training data instead of the demo, see section 3.

---

## 2. 📊 The Barbershop dashboard (the 5-screen diagnostics app)

This one is a small program, so it needs **Python**. Two ways — pick **A** (your computer) or **B** (Google
Colab, nothing installed on your machine).

### Option A — on your computer (recommended)

1. **Install Python** if you don't have it: https://www.python.org/downloads/ (tick *"Add Python to
   PATH"* on Windows). To check it's there, open a terminal and type `python --version`.
2. **Open a terminal** *in the project folder*:
   - **Windows:** open the folder in File Explorer → click the address bar → type `cmd` → Enter.
   - **Mac:** right-click the folder → *Services* → *New Terminal at Folder*.
3. **Type this one line and press Enter:**
   ```
   python run_dashboard.py
   ```
   It installs what it needs the first time, then your browser opens at **http://localhost:8050**.
   (If `python` isn't found, try `python3 run_dashboard.py` or `py run_dashboard.py`.)
4. **To stop it:** click the terminal window and press **Ctrl + C**.

### Option B — Google Colab (no install on your computer)

1. Open **`colab/Quantra_Barbershop.ipynb`** in Google Colab (colab.research.google.com → *File → Open
   notebook → GitHub*, paste the repo URL).
2. Run **Cell 1** (it clones the repo + installs everything — takes a minute).
3. Run **Cell 8** ("ngrok tunnel"). It prints a **public link** — click it to open the dashboard in a
   new tab. *(Cell 8 needs a free ngrok token; the cell tells you where to paste it.)*

**What you'll see:** tabs across the top —
`ℹ️ How to Use · 1 Training Wall · 2 Scoreboard · 3 Day Replay · 4 Trade Autopsy · 5 Pattern Finder ·
6 Repo Map`. Click around: a **scoreboard card** opens a day; a **trade marker** opens the autopsy; on
**Trade Autopsy** click any **SAW feature** to overlay it on the chart; **Repo Map** shows the live
import graph (for that one, run `pip install dash-cytoscape` first, or it shows a friendly note).

---

## 3. 🔴 Make the JARVIS HUD show LIVE data (optional)

The HUD listens for real training telemetry. In a terminal in the project folder:

```
pip install websockets
python -m barbershop.ws_broadcaster --demo
```

Now **refresh `jarvis_hud.html`** — the pill flips to **`LIVE`** (green) and the graph pulses with the
stream. (`--demo` is fake-but-real-pipe data; drop `--demo` once a real training run is writing
telemetry under `artifacts/telemetry/`.) Full details: **`JARVIS_HUD_GUIDE.md`**.

---

## 🧯 If something goes wrong

- **`python` not recognised** → use `python3` or (Windows) `py`. Or reinstall Python with *"Add to PATH"*.
- **"port 8050 in use"** → close the other dashboard window, or change `DASH_PORT` in `barbershop/config.py`.
- **Repo Map tab is just a note** → `pip install dash-cytoscape`, then reopen the tab.
- **HUD fonts look plain / stuck on SIM** → that's fine offline; the HUD still works (SIM = demo data).
- **Nothing opens** → open the browser yourself and go to **http://localhost:8050** (dashboard) or open
  `jarvis_hud.html` directly (HUD).

> You can't break anything by running these — the dashboard + HUD are **read-only**. They only show and
> explain; they never trade, never change training, and never touch your policy.
