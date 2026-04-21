"""
anemia_app_pro.py
-----------------
Professional Anemia Detection System - Advanced UI
Run with: python anemia_app_pro.py
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys
import time
import datetime

try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageTk, ImageFilter, ImageEnhance
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── Model paths ──
DENSENET_MODEL  = "./models/best_densenet_fold_1.keras"
VGG_MODEL       = "./models/best_vgg16_fold_1.keras"
INCEPTION_MODEL = "./models/best_inception_fold_1.keras"

# ── Color Palette ──
BG_DARK      = "#050B18"
BG_NAV       = "#080F1F"
BG_CARD      = "#0D1626"
BG_CARD2     = "#111D30"
BG_CARD3     = "#162038"
ACCENT_BLUE  = "#2D7DD2"
ACCENT_CYAN  = "#00C9FF"
ACCENT_GREEN = "#00E676"
ACCENT_RED   = "#FF4D6D"
ACCENT_GOLD  = "#FFD60A"
ACCENT_PURPLE= "#9D4EDD"
TEXT_WHITE   = "#FFFFFF"
TEXT_LIGHT   = "#CBD5E1"
TEXT_GRAY    = "#64748B"
TEXT_DIM     = "#334155"
BORDER       = "#1E2D45"
BORDER2      = "#243552"

# ── Terminal colors ──
TERM_RESET  = "\033[0m"
TERM_CYAN   = "\033[96m"
TERM_GREEN  = "\033[92m"
TERM_RED    = "\033[91m"
TERM_YELLOW = "\033[93m"
TERM_BLUE   = "\033[94m"
TERM_PURPLE = "\033[95m"
TERM_BOLD   = "\033[1m"
TERM_DIM    = "\033[2m"

def term_print(text="", color="", end="\n"):
    print(f"{color}{text}{TERM_RESET}", end=end, flush=True)

def print_banner():
    print()
    term_print("╔══════════════════════════════════════════════════════════════╗", TERM_CYAN)
    term_print("║                                                              ║", TERM_CYAN)
    term_print("║        🔬  ANEMIA DETECTION SYSTEM  — PRO EDITION           ║", TERM_CYAN)
    term_print("║          DenseNet121 + VGG16 + InceptionV3 Ensemble          ║", TERM_CYAN)
    term_print("║                                                              ║", TERM_CYAN)
    term_print("╚══════════════════════════════════════════════════════════════╝", TERM_CYAN)
    term_print(f"  Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", TERM_DIM)
    print()

def print_section(title):
    print()
    term_print(f"  ┌─ {title} {'─'*(54-len(title))}", TERM_BLUE)

def print_model_result(name, label, confidence, is_anemic):
    icon = "🔴" if is_anemic else "🟢"
    color = TERM_RED if is_anemic else TERM_GREEN
    bar_len = int(confidence / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    term_print(f"  │  {name:<14} {icon}  ", TERM_BLUE, end="")
    term_print(f"{label:<12}", color, end="")
    term_print(f"  [{bar}]  {confidence:.1f}%", TERM_DIM)


class AnemiaAppPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Anemia Detection System — Pro")
        self.root.geometry("1100x720")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BG_DARK)

        self.models_loaded = False
        self.current_image_path = None
        self.photo_img = None
        self.history = []
        self.current_tab = tk.StringVar(value="analyze")
        self.loading_dots = 0
        self.is_analyzing = False

        # Center window
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"1100x720+{(sw-1100)//2}+{(sh-720)//2}")

        print_banner()
        term_print("  Loading application...", TERM_DIM)

        self.build_ui()

        if ML_AVAILABLE:
            threading.Thread(target=self.load_models, daemon=True).start()
        else:
            self.log("⚠  TensorFlow not found. Install required libraries.", ACCENT_GOLD)
            term_print("  ⚠  TensorFlow not available.", TERM_YELLOW)

    # ══════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════
    def build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self.root, bg=BG_NAV, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        # Logo
        tk.Label(topbar, text="⬡", font=("Segoe UI", 20), bg=BG_NAV,
                 fg=ACCENT_CYAN).pack(side="left", padx=(16,4), pady=10)
        tk.Label(topbar, text="AnemiaAI", font=("Segoe UI", 13, "bold"),
                 bg=BG_NAV, fg=TEXT_WHITE).pack(side="left", pady=10)
        tk.Label(topbar, text=" PRO", font=("Segoe UI", 8, "bold"),
                 bg=ACCENT_CYAN, fg=BG_DARK, padx=4).pack(side="left", pady=18)

        # Nav tabs
        nav = tk.Frame(topbar, bg=BG_NAV)
        nav.pack(side="left", padx=30, fill="y")
        for tab, label in [("analyze","🔍  Analyze"), ("history","📋  History"), ("about","ℹ  About")]:
            self._nav_btn(nav, label, tab)

        # Status
        self.status_frame = tk.Frame(topbar, bg="#0D1A2E", padx=12, pady=4)
        self.status_frame.pack(side="right", padx=16, pady=14)
        self.status_dot = tk.Label(self.status_frame, text="●", font=("Segoe UI", 10),
                                   bg="#0D1A2E", fg=ACCENT_GOLD)
        self.status_dot.pack(side="left")
        self.status_lbl = tk.Label(self.status_frame, text="Loading models...",
                                   font=("Segoe UI", 9), bg="#0D1A2E", fg=TEXT_LIGHT)
        self.status_lbl.pack(side="left", padx=(4,0))

        # ── Content area (tab switcher) ──
        self.content = tk.Frame(self.root, bg=BG_DARK)
        self.content.pack(fill="both", expand=True)

        self.tab_analyze = tk.Frame(self.content, bg=BG_DARK)
        self.tab_history = tk.Frame(self.content, bg=BG_DARK)
        self.tab_about   = tk.Frame(self.content, bg=BG_DARK)

        self._build_analyze_tab()
        self._build_history_tab()
        self._build_about_tab()

        self.show_tab("analyze")

        # ── Footer ──
        footer = tk.Frame(self.root, bg=BG_NAV, height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer,
                 text="⚠  Educational purposes only — Always consult a qualified medical professional  •  v2.0 Pro",
                 font=("Segoe UI", 7), bg=BG_NAV, fg=TEXT_GRAY).pack(pady=6)

    def _nav_btn(self, parent, text, tab_id):
        btn = tk.Button(parent, text=text, font=("Segoe UI", 9),
                        bg=BG_NAV, fg=TEXT_GRAY, bd=0, padx=14, pady=16,
                        activebackground=BG_CARD, activeforeground=TEXT_WHITE,
                        cursor="hand2", relief="flat",
                        command=lambda t=tab_id: self.show_tab(t))
        btn.pack(side="left")
        setattr(self, f"nav_{tab_id}", btn)

    def show_tab(self, tab_id):
        self.current_tab.set(tab_id)
        for t in ["analyze", "history", "about"]:
            frame = getattr(self, f"tab_{t}")
            btn   = getattr(self, f"nav_{t}")
            if t == tab_id:
                frame.pack(fill="both", expand=True)
                btn.config(fg=ACCENT_CYAN, bg=BG_CARD)
            else:
                frame.pack_forget()
                btn.config(fg=TEXT_GRAY, bg=BG_NAV)

    # ──────────────────────────────────────────
    #  ANALYZE TAB
    # ──────────────────────────────────────────
    def _build_analyze_tab(self):
        tab = self.tab_analyze
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)

        # ── Left panel ──
        left = tk.Frame(tab, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(16,8), pady=16)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Image card
        img_card = self._card(left, "👁  EYE IMAGE UPLOAD")
        img_card.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        img_card.rowconfigure(1, weight=1)
        img_card.columnconfigure(0, weight=1)

        # Drop zone
        self.drop_zone = tk.Frame(img_card, bg=BG_CARD2, bd=0,
                                  highlightthickness=2, highlightbackground=BORDER2)
        self.drop_zone.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,10))

        self.img_display = tk.Label(self.drop_zone, bg=BG_CARD2,
            text="📷\n\nNo image selected\n\nClick Browse Image to upload\na conjunctiva eye photo",
            font=("Segoe UI", 10), fg=TEXT_GRAY, justify="center")
        self.img_display.pack(fill="both", expand=True, padx=10, pady=30)

        # Image info bar
        self.img_info = tk.Label(img_card, text="No file loaded",
                                 font=("Segoe UI", 8), bg=BG_CARD,
                                 fg=TEXT_GRAY, anchor="w")
        self.img_info.grid(row=2, column=0, sticky="ew", padx=12, pady=(0,8))

        # Buttons row
        btn_row = tk.Frame(img_card, bg=BG_CARD)
        btn_row.grid(row=3, column=0, sticky="ew", padx=12, pady=(0,12))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._btn(btn_row, "📂  Browse Image", ACCENT_BLUE, self.browse_image).grid(
            row=0, column=0, sticky="ew", padx=(0,4))
        self._btn(btn_row, "🗑  Clear", BG_CARD3, self.clear_image).grid(
            row=0, column=1, sticky="ew", padx=(4,0))

        # Analyze button
        self.analyze_btn = self._btn(img_card, "🔍  ANALYZE FOR ANEMIA",
                                     ACCENT_CYAN, self.run_prediction,
                                     fg=BG_DARK, font_size=11)
        self.analyze_btn.config(state="disabled", pady=12)
        self.analyze_btn.grid(row=4, column=0, sticky="ew", padx=12, pady=(0,12))

        # Log card
        log_card = self._card(left, "📟  ANALYSIS LOG")
        log_card.grid(row=1, column=0, sticky="nsew")
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)

        self.log_box = tk.Text(log_card, bg=BG_CARD2, fg=ACCENT_CYAN,
                               font=("Consolas", 8), bd=0, wrap="word",
                               state="disabled", height=6,
                               insertbackground=ACCENT_CYAN,
                               selectbackground=BORDER2)
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,12))

        scrollbar = tk.Scrollbar(log_card, command=self.log_box.yview,
                                 bg=BG_CARD, troughcolor=BG_CARD2, bd=0)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0,12))
        self.log_box.config(yscrollcommand=scrollbar.set)

        # ── Right panel ──
        right = tk.Frame(tab, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(8,16), pady=16)
        right.columnconfigure(0, weight=1)

        # Main result card
        res_card = self._card(right, "🧬  DIAGNOSIS RESULT")
        res_card.grid(row=0, column=0, sticky="ew", pady=(0,10))
        res_card.columnconfigure(0, weight=1)

        self.result_icon  = tk.Label(res_card, text="⬡", font=("Segoe UI", 38),
                                     bg=BG_CARD, fg=TEXT_DIM)
        self.result_icon.grid(row=1, column=0, pady=(8,0))

        self.result_main  = tk.Label(res_card, text="AWAITING INPUT",
                                     font=("Segoe UI", 16, "bold"),
                                     bg=BG_CARD, fg=TEXT_DIM)
        self.result_main.grid(row=2, column=0)

        self.result_sub   = tk.Label(res_card, text="Upload an eye image to begin analysis",
                                     font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_GRAY)
        self.result_sub.grid(row=3, column=0, pady=(2,10))

        # Confidence bar
        conf_frame = tk.Frame(res_card, bg=BG_CARD)
        conf_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=(0,4))
        conf_frame.columnconfigure(0, weight=1)

        conf_top = tk.Frame(conf_frame, bg=BG_CARD)
        conf_top.grid(row=0, column=0, sticky="ew")
        tk.Label(conf_top, text="CONFIDENCE", font=("Segoe UI", 7, "bold"),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(side="left")
        self.conf_pct = tk.Label(conf_top, text="0%", font=("Segoe UI", 7, "bold"),
                                 bg=BG_CARD, fg=TEXT_GRAY)
        self.conf_pct.pack(side="right")

        self.bar_bg = tk.Frame(conf_frame, bg=BG_CARD3, height=10)
        self.bar_bg.grid(row=1, column=0, sticky="ew", pady=4)
        self.bar_fill = tk.Frame(self.bar_bg, bg=TEXT_DIM, height=10, width=0)
        self.bar_fill.place(x=0, y=0, relheight=1)

        # Risk indicator
        self.risk_frame = tk.Frame(res_card, bg=BG_CARD)
        self.risk_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(4,12))
        self.risk_label = tk.Label(self.risk_frame, text="Risk Level: —",
                                   font=("Segoe UI", 9, "bold"),
                                   bg=BG_CARD, fg=TEXT_GRAY)
        self.risk_label.pack(side="left")
        self.risk_badge = tk.Label(self.risk_frame, text="UNKNOWN",
                                   font=("Segoe UI", 8, "bold"),
                                   bg=BG_CARD3, fg=TEXT_GRAY, padx=8, pady=2)
        self.risk_badge.pack(side="right")

        # ── Model breakdown card ──
        model_card = self._card(right, "🤖  MODEL BREAKDOWN")
        model_card.grid(row=1, column=0, sticky="ew", pady=(0,10))
        model_card.columnconfigure(0, weight=1)

        self.model_widgets = {}
        models_info = [
            ("DenseNet121", "121-layer dense connections", ACCENT_CYAN),
            ("VGG16",       "16-layer visual geometry",   ACCENT_BLUE),
            ("InceptionV3", "299x299 inception modules",  ACCENT_PURPLE),
        ]
        for i, (name, desc, color) in enumerate(models_info):
            row_f = tk.Frame(model_card, bg=BG_CARD2 if i%2==0 else BG_CARD,
                             padx=10, pady=6)
            row_f.grid(row=i+1, column=0, sticky="ew", padx=12,
                       pady=(0 if i>0 else 4, 4 if i<2 else 8))
            row_f.columnconfigure(1, weight=1)

            dot = tk.Label(row_f, text="●", font=("Segoe UI", 8),
                           bg=row_f["bg"], fg=color)
            dot.grid(row=0, column=0, rowspan=2, padx=(0,8))

            tk.Label(row_f, text=name, font=("Segoe UI", 9, "bold"),
                     bg=row_f["bg"], fg=TEXT_LIGHT).grid(row=0, column=1, sticky="w")
            tk.Label(row_f, text=desc, font=("Segoe UI", 7),
                     bg=row_f["bg"], fg=TEXT_GRAY).grid(row=1, column=1, sticky="w")

            res_lbl = tk.Label(row_f, text="—", font=("Segoe UI", 9, "bold"),
                               bg=row_f["bg"], fg=TEXT_DIM)
            res_lbl.grid(row=0, column=2, rowspan=2, padx=(8,0))

            conf_lbl = tk.Label(row_f, text="", font=("Segoe UI", 8),
                                bg=row_f["bg"], fg=TEXT_GRAY)
            conf_lbl.grid(row=0, column=3, rowspan=2, padx=(6,0))

            self.model_widgets[name] = (res_lbl, conf_lbl, dot, color)

        # ── Stats card ──
        stats_card = self._card(right, "📊  SESSION STATS")
        stats_card.grid(row=2, column=0, sticky="ew", pady=(0,10))

        stats_inner = tk.Frame(stats_card, bg=BG_CARD)
        stats_inner.grid(row=1, column=0, sticky="ew", padx=12, pady=(0,12))
        stats_inner.columnconfigure((0,1,2), weight=1)

        self.stat_total   = self._stat_box(stats_inner, "TOTAL", "0", 0)
        self.stat_anemic  = self._stat_box(stats_inner, "ANEMIC", "0", 1, ACCENT_RED)
        self.stat_normal  = self._stat_box(stats_inner, "NORMAL", "0", 2, ACCENT_GREEN)

        # ── Tips card ──
        tips_card = self._card(right, "💡  PHOTO GUIDE")
        tips_card.grid(row=3, column=0, sticky="ew")

        tips = [
            ("Pull lower eyelid gently downward",     "✋"),
            ("Expose pink inner conjunctiva tissue",  "👁"),
            ("Use bright natural or lamp light",       "💡"),
            ("Take close-up, keep camera steady",      "📷"),
            ("90%+ confidence = reliable result",      "✅"),
        ]
        for i, (tip, icon) in enumerate(tips):
            row_f = tk.Frame(tips_card, bg=BG_CARD)
            row_f.grid(row=i+1, column=0, sticky="ew", padx=12,
                       pady=(0 if i>0 else 2, 2 if i<4 else 10))
            tk.Label(row_f, text=icon, font=("Segoe UI Emoji", 9),
                     bg=BG_CARD, fg=ACCENT_GOLD).pack(side="left", padx=(0,6))
            tk.Label(row_f, text=tip, font=("Segoe UI", 8),
                     bg=BG_CARD, fg=TEXT_GRAY).pack(side="left")

    # ──────────────────────────────────────────
    #  HISTORY TAB
    # ──────────────────────────────────────────
    def _build_history_tab(self):
        tab = self.tab_history
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        hdr = tk.Frame(tab, bg=BG_DARK)
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(16,8))
        tk.Label(hdr, text="📋  Analysis History",
                 font=("Segoe UI", 14, "bold"), bg=BG_DARK, fg=TEXT_WHITE).pack(side="left")
        self._btn(hdr, "🗑  Clear History", BG_CARD3, self.clear_history).pack(side="right")

        # Table frame
        table_card = tk.Frame(tab, bg=BG_CARD, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        table_card.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0,16))
        table_card.rowconfigure(1, weight=1)
        table_card.columnconfigure(0, weight=1)

        # Table header
        cols = ["#", "Time", "Image File", "DenseNet", "VGG16", "Inception", "FINAL", "Confidence"]
        hdr_f = tk.Frame(table_card, bg=BG_CARD3)
        hdr_f.grid(row=0, column=0, sticky="ew")
        widths = [3, 10, 22, 10, 10, 10, 12, 10]
        for i, (col, w) in enumerate(zip(cols, widths)):
            tk.Label(hdr_f, text=col, font=("Segoe UI", 8, "bold"),
                     bg=BG_CARD3, fg=ACCENT_CYAN, width=w, anchor="w",
                     padx=8, pady=6).grid(row=0, column=i, sticky="ew")

        # Scrollable rows
        canvas = tk.Canvas(table_card, bg=BG_CARD, bd=0, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        sb = tk.Scrollbar(table_card, orient="vertical", command=canvas.yview,
                          bg=BG_CARD, troughcolor=BG_CARD2)
        sb.grid(row=1, column=1, sticky="ns")
        canvas.config(yscrollcommand=sb.set)

        self.history_frame = tk.Frame(canvas, bg=BG_CARD)
        self.history_window = canvas.create_window((0,0), window=self.history_frame, anchor="nw")

        def on_resize(e):
            canvas.itemconfig(self.history_window, width=e.width)
        canvas.bind("<Configure>", on_resize)
        self.history_frame.bind("<Configure>",
            lambda e: canvas.config(scrollregion=canvas.bbox("all")))

        self.history_canvas = canvas
        self.history_cols   = widths

        tk.Label(self.history_frame, text="No analyses yet — go to Analyze tab to begin",
                 font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_GRAY,
                 pady=40).pack()

    # ──────────────────────────────────────────
    #  ABOUT TAB
    # ──────────────────────────────────────────
    def _build_about_tab(self):
        tab = self.tab_about
        tab.columnconfigure((0,1), weight=1)

        tk.Label(tab, text="AnemiaAI Pro", font=("Segoe UI", 22, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).grid(row=0, column=0, columnspan=2,
                 pady=(24,4), padx=16, sticky="w")
        tk.Label(tab, text="Non-invasive anemia detection using deep learning ensemble",
                 font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_GRAY).grid(
                 row=1, column=0, columnspan=2, padx=16, sticky="w", pady=(0,16))

        # Models card
        m_card = self._card(tab, "🤖  MODELS USED")
        m_card.grid(row=2, column=0, sticky="nsew", padx=(16,8), pady=(0,10))
        models = [
            ("DenseNet121", "99.46%", "Dense connections, 121 layers", ACCENT_CYAN),
            ("VGG16",       "100%",   "Visual geometry, 16 layers",   ACCENT_BLUE),
            ("InceptionV3", "96.70%", "Inception modules, 299x299",   ACCENT_PURPLE),
            ("Ensemble",    "100%",   "Average of all 3 models",      ACCENT_GREEN),
        ]
        for i, (name, acc, desc, col) in enumerate(models):
            f = tk.Frame(m_card, bg=BG_CARD2, padx=12, pady=8)
            f.grid(row=i+1, column=0, sticky="ew", padx=12,
                   pady=(0 if i>0 else 4, 4 if i<3 else 10))
            f.columnconfigure(1, weight=1)
            tk.Label(f, text="●", font=("Segoe UI", 10), bg=BG_CARD2,
                     fg=col).grid(row=0, column=0, rowspan=2, padx=(0,10))
            tk.Label(f, text=name, font=("Segoe UI", 10, "bold"),
                     bg=BG_CARD2, fg=TEXT_WHITE).grid(row=0, column=1, sticky="w")
            tk.Label(f, text=desc, font=("Segoe UI", 8),
                     bg=BG_CARD2, fg=TEXT_GRAY).grid(row=1, column=1, sticky="w")
            tk.Label(f, text=acc, font=("Segoe UI", 11, "bold"),
                     bg=BG_CARD2, fg=col).grid(row=0, column=2, rowspan=2, padx=8)

        # Dataset card
        d_card = self._card(tab, "📊  DATASET INFO")
        d_card.grid(row=2, column=1, sticky="nsew", padx=(8,16), pady=(0,10))
        stats = [
            ("Total Images",      "183"),
            ("Anemic (img_1_*)",  "82"),
            ("Non-Anemic (img_2_*)", "101"),
            ("Cross Validation",  "5-Fold Stratified"),
            ("Input Size (VGG/Dense)", "224 × 224 px"),
            ("Input Size (Inception)", "299 × 299 px"),
            ("Augmentation",      "Rotation, Zoom, Flip"),
            ("Imbalance Fix",     "Upsampling"),
        ]
        for i, (k, v) in enumerate(stats):
            f = tk.Frame(d_card, bg=BG_CARD2 if i%2==0 else BG_CARD, padx=12, pady=6)
            f.grid(row=i+1, column=0, sticky="ew", padx=12,
                   pady=(0 if i>0 else 4, 4 if i<7 else 10))
            f.columnconfigure(0, weight=1)
            tk.Label(f, text=k, font=("Segoe UI", 8),
                     bg=f["bg"], fg=TEXT_GRAY).grid(row=0, column=0, sticky="w")
            tk.Label(f, text=v, font=("Segoe UI", 8, "bold"),
                     bg=f["bg"], fg=TEXT_LIGHT).grid(row=0, column=1, sticky="e")

        # How it works
        how_card = self._card(tab, "⚙  HOW IT WORKS")
        how_card.grid(row=3, column=0, columnspan=2, sticky="ew",
                      padx=16, pady=(0,16))
        steps = [
            "1. Upload a close-up photo of the inner lower eyelid (conjunctiva)",
            "2. Image is preprocessed separately for each model (resize + normalize)",
            "3. DenseNet121, VGG16 and InceptionV3 each analyze the image independently",
            "4. All 3 predictions are averaged together (ensemble method)",
            "5. Final decision: probability > 50% = Anemic, else Non-Anemic",
        ]
        for step in steps:
            tk.Label(how_card, text=step, font=("Segoe UI", 9),
                     bg=BG_CARD, fg=TEXT_LIGHT, anchor="w",
                     justify="left").grid(sticky="ew", padx=16, pady=2)
        tk.Frame(how_card, bg=BG_CARD, height=8).grid()

    # ══════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════
    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        frame.columnconfigure(0, weight=1)
        tk.Label(frame, text=title, font=("Segoe UI", 8, "bold"),
                 bg=BG_CARD, fg=ACCENT_CYAN, anchor="w",
                 padx=14, pady=8).grid(row=0, column=0, sticky="ew")
        tk.Frame(frame, bg=BORDER, height=1).grid(row=0, column=0,
                 sticky="ew", pady=(0,0))
        return frame

    def _btn(self, parent, text, bg, cmd, fg=TEXT_WHITE, font_size=9):
        return tk.Button(parent, text=text, font=("Segoe UI", font_size, "bold"),
                         bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                         bd=0, padx=12, pady=8, cursor="hand2",
                         relief="flat", command=cmd)

    def _stat_box(self, parent, label, value, col, color=TEXT_LIGHT):
        f = tk.Frame(parent, bg=BG_CARD2, padx=10, pady=8)
        f.grid(row=0, column=col, sticky="ew", padx=4, pady=8)
        lbl = tk.Label(f, text=value, font=("Segoe UI", 18, "bold"),
                       bg=BG_CARD2, fg=color)
        lbl.pack()
        tk.Label(f, text=label, font=("Segoe UI", 7),
                 bg=BG_CARD2, fg=TEXT_GRAY).pack()
        return lbl

    def log(self, msg, color=None):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def set_status(self, text, dot_color):
        self.status_lbl.config(text=text)
        self.status_dot.config(fg=dot_color)

    # ══════════════════════════════════════════
    #  MODEL LOADING
    # ══════════════════════════════════════════
    def load_models(self):
        print_section("LOADING MODELS")
        term_print(f"  │  Initializing TensorFlow {tf.__version__}...", TERM_DIM)

        models_ok = True
        for path, name in [(DENSENET_MODEL,"DenseNet121"),
                           (VGG_MODEL,"VGG16"),
                           (INCEPTION_MODEL,"InceptionV3")]:
            if not os.path.isfile(path):
                term_print(f"  │  ✗  {name} not found: {path}", TERM_RED)
                models_ok = False
            else:
                term_print(f"  │  ⟳  Loading {name}...", TERM_DIM)

        if not models_ok:
            self.root.after(0, lambda: self.set_status("✗  Models not found", ACCENT_RED))
            self.root.after(0, lambda: self.log("✗  Model files not found. Run training scripts first.", ACCENT_RED))
            return

        try:
            self.model_densenet  = tf.keras.models.load_model(DENSENET_MODEL)
            term_print(f"  │  ✓  DenseNet121 loaded", TERM_GREEN)
            self.model_vgg       = tf.keras.models.load_model(VGG_MODEL)
            term_print(f"  │  ✓  VGG16 loaded", TERM_GREEN)
            self.model_inception = tf.keras.models.load_model(INCEPTION_MODEL)
            term_print(f"  │  ✓  InceptionV3 loaded", TERM_GREEN)
            term_print(f"  │  ✓  All models ready!", TERM_GREEN)
            self.models_loaded = True
            self.root.after(0, lambda: self.set_status("✓  All Models Ready", ACCENT_GREEN))
            self.root.after(0, lambda: self.log("✓  All 3 models loaded successfully."))
        except Exception as e:
            term_print(f"  │  ✗  Error: {e}", TERM_RED)
            self.root.after(0, lambda: self.set_status("✗  Load Error", ACCENT_RED))
            self.root.after(0, lambda: self.log(f"✗  Error loading models: {e}"))

    # ══════════════════════════════════════════
    #  IMAGE HANDLING
    # ══════════════════════════════════════════
    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Eye / Conjunctiva Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")]
        )
        if path:
            self.current_image_path = path
            self.show_preview(path)
            self.analyze_btn.config(state="normal")
            self.reset_results()
            fname = os.path.basename(path)
            size  = os.path.getsize(path) // 1024
            self.img_info.config(text=f"📄  {fname}   •   {size} KB", fg=TEXT_LIGHT)
            self.log(f"📂  Image loaded: {fname}")
            term_print(f"\n  Image selected: {path}", TERM_DIM)

    def show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((380, 220))
            self.photo_img = ImageTk.PhotoImage(img)
            self.img_display.config(image=self.photo_img, text="")
            self.drop_zone.config(highlightbackground=ACCENT_BLUE)
        except Exception:
            self.img_display.config(text="⚠  Could not preview", image="")

    def clear_image(self):
        self.current_image_path = None
        self.photo_img = None
        self.img_display.config(image="",
            text="📷\n\nNo image selected\n\nClick Browse Image to upload\na conjunctiva eye photo")
        self.img_info.config(text="No file loaded", fg=TEXT_GRAY)
        self.drop_zone.config(highlightbackground=BORDER2)
        self.analyze_btn.config(state="disabled")
        self.reset_results()
        self.log("🗑  Image cleared.")

    def reset_results(self):
        self.result_icon.config(text="⬡", fg=TEXT_DIM)
        self.result_main.config(text="AWAITING INPUT", fg=TEXT_DIM)
        self.result_sub.config(text="Upload an eye image to begin analysis", fg=TEXT_GRAY)
        self.conf_pct.config(text="0%", fg=TEXT_GRAY)
        self.bar_fill.config(bg=TEXT_DIM, width=0)
        self.risk_label.config(text="Risk Level: —", fg=TEXT_GRAY)
        self.risk_badge.config(text="UNKNOWN", bg=BG_CARD3, fg=TEXT_GRAY)
        for name, (r, c, d, _) in self.model_widgets.items():
            r.config(text="—", fg=TEXT_DIM)
            c.config(text="")

    # ══════════════════════════════════════════
    #  PREDICTION
    # ══════════════════════════════════════════
    def run_prediction(self):
        if not self.current_image_path or self.is_analyzing:
            return
        if not ML_AVAILABLE:
            self.log("✗  TensorFlow not installed.")
            return
        if not self.models_loaded:
            self.log("⟳  Models still loading, please wait...")
            return

        self.is_analyzing = True
        self.analyze_btn.config(state="disabled", text="  ⟳  Analyzing...")
        self.set_status("⟳  Analyzing image...", ACCENT_GOLD)
        self.log(f"🔍  Starting analysis: {os.path.basename(self.current_image_path)}")
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            print_section("PREDICTION")
            term_print(f"  │  File: {self.current_image_path}", TERM_DIM)
            term_print(f"  │  Time: {datetime.datetime.now().strftime('%H:%M:%S')}", TERM_DIM)
            print()

            def load(path, size, fn):
                img = Image.open(path).convert("RGB").resize(size)
                arr = np.array(img, dtype=np.float32)
                return np.expand_dims(fn(arr), 0)

            img_d = load(self.current_image_path, (224,224), densenet_pre)
            img_v = load(self.current_image_path, (224,224), vgg_pre)
            img_i = load(self.current_image_path, (299,299), inception_pre)

            t0 = time.time()
            p_d = float(self.model_densenet.predict(img_d,  verbose=0)[0][0])
            p_v = float(self.model_vgg.predict(img_v,       verbose=0)[0][0])
            p_i = float(self.model_inception.predict(img_i, verbose=0)[0][0])
            elapsed = time.time() - t0

            p_e = (p_d + p_v + p_i) / 3.0

            def lc(p):
                if p > 0.5:
                    return "ANEMIC", p*100, True
                return "NON-ANEMIC", (1-p)*100, False

            lbl_d, conf_d, an_d = lc(p_d)
            lbl_v, conf_v, an_v = lc(p_v)
            lbl_i, conf_i, an_i = lc(p_i)
            lbl_e, conf_e, an_e = lc(p_e)

            # Terminal output
            print_model_result("DenseNet121",  lbl_d, conf_d, an_d)
            print_model_result("VGG16",        lbl_v, conf_v, an_v)
            print_model_result("InceptionV3",  lbl_i, conf_i, an_i)
            print()
            term_print(f"  │  {'─'*52}", TERM_BLUE)
            final_color = TERM_RED if an_e else TERM_GREEN
            icon = "🔴" if an_e else "🟢"
            term_print(f"  │  {icon}  FINAL DECISION : ", TERM_BLUE, end="")
            term_print(f"{lbl_e}", final_color + TERM_BOLD, end="")
            term_print(f"  ({conf_e:.1f}% confidence)", TERM_DIM)
            term_print(f"  │  ⏱  Inference time : {elapsed:.2f}s", TERM_DIM)
            term_print(f"  └{'─'*56}", TERM_BLUE)
            print()

            self.root.after(0, lambda: self._update_ui(
                p_d, p_v, p_i, p_e,
                lbl_d, lbl_v, lbl_i, lbl_e,
                conf_d, conf_v, conf_i, conf_e,
                an_d, an_v, an_i, an_e, elapsed
            ))
        except Exception as ex:
            self.root.after(0, lambda: self._show_error(str(ex)))

    def _update_ui(self, p_d, p_v, p_i, p_e,
                   lbl_d, lbl_v, lbl_i, lbl_e,
                   conf_d, conf_v, conf_i, conf_e,
                   an_d, an_v, an_i, an_e, elapsed):

        col_e = ACCENT_RED if an_e else ACCENT_GREEN
        icon  = "🔴" if an_e else "🟢"

        # Main result
        self.result_icon.config(text=icon, fg=col_e)
        self.result_main.config(text=lbl_e, fg=col_e)
        self.result_sub.config(
            text=f"Inference time: {elapsed:.2f}s  •  Ensemble of 3 models",
            fg=TEXT_GRAY)
        self.conf_pct.config(text=f"{conf_e:.1f}%", fg=col_e)

        # Progress bar
        self.bar_bg.update_idletasks()
        w = self.bar_bg.winfo_width()
        self.bar_fill.config(bg=col_e, width=int(w * conf_e / 100))

        # Risk badge
        if conf_e >= 90:
            risk, rbg = ("HIGH RISK" if an_e else "LOW RISK"), col_e
        elif conf_e >= 70:
            risk, rbg = "MODERATE", ACCENT_GOLD
        else:
            risk, rbg = "UNCERTAIN", TEXT_GRAY
        self.risk_label.config(text="Risk Level:", fg=TEXT_LIGHT)
        self.risk_badge.config(text=risk, bg=rbg, fg=BG_DARK if rbg != TEXT_GRAY else TEXT_GRAY)

        # Model rows
        for name, lbl, conf, an in [
            ("DenseNet121", lbl_d, conf_d, an_d),
            ("VGG16",       lbl_v, conf_v, an_v),
            ("InceptionV3", lbl_i, conf_i, an_i),
        ]:
            col = ACCENT_RED if an else ACCENT_GREEN
            ico = "🔴" if an else "🟢"
            r, c, d, _ = self.model_widgets[name]
            r.config(text=f"{ico} {lbl}", fg=col)
            c.config(text=f"{conf:.1f}%", fg=col)

        # Stats
        self.history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "file": os.path.basename(self.current_image_path),
            "densenet": (lbl_d, conf_d, an_d),
            "vgg": (lbl_v, conf_v, an_v),
            "inception": (lbl_i, conf_i, an_i),
            "final": (lbl_e, conf_e, an_e),
        })
        total  = len(self.history)
        anemic = sum(1 for h in self.history if h["final"][2])
        self.stat_total.config(text=str(total))
        self.stat_anemic.config(text=str(anemic))
        self.stat_normal.config(text=str(total - anemic))

        # Update history tab
        self._add_history_row(self.history[-1], total)

        self.log(f"✓  Result: {lbl_e} ({conf_e:.1f}%) in {elapsed:.2f}s")
        self.analyze_btn.config(state="normal", text="  🔍  ANALYZE FOR ANEMIA")
        self.set_status(f"✓  {lbl_e} — {conf_e:.1f}%", col_e)
        self.is_analyzing = False

    def _add_history_row(self, entry, row_num):
        # Clear placeholder
        for w in self.history_frame.winfo_children():
            if isinstance(w, tk.Label) and "No analyses" in (w.cget("text") or ""):
                w.destroy()

        bg = BG_CARD2 if row_num % 2 == 0 else BG_CARD
        row = tk.Frame(self.history_frame, bg=bg)
        row.pack(fill="x")

        cols_data = [
            str(row_num),
            entry["time"],
            entry["file"][:20],
            entry["densenet"][0][:3],
            entry["vgg"][0][:3],
            entry["inception"][0][:3],
            entry["final"][0],
            f"{entry['final'][1]:.1f}%",
        ]
        widths = [3, 10, 22, 10, 10, 10, 12, 10]
        colors = [TEXT_GRAY, TEXT_GRAY, TEXT_LIGHT,
                  ACCENT_RED if entry["densenet"][2] else ACCENT_GREEN,
                  ACCENT_RED if entry["vgg"][2]      else ACCENT_GREEN,
                  ACCENT_RED if entry["inception"][2] else ACCENT_GREEN,
                  ACCENT_RED if entry["final"][2]     else ACCENT_GREEN,
                  ACCENT_GOLD]
        for val, w, col in zip(cols_data, widths, colors):
            tk.Label(row, text=val, font=("Segoe UI", 8), bg=bg,
                     fg=col, width=w, anchor="w", padx=8, pady=5).pack(side="left")

    def _show_error(self, msg):
        self.log(f"✗  Error: {msg}")
        self.result_main.config(text="ERROR", fg=ACCENT_RED)
        self.result_sub.config(text=msg[:60], fg=ACCENT_RED)
        self.analyze_btn.config(state="normal", text="  🔍  ANALYZE FOR ANEMIA")
        self.set_status("✗  Error occurred", ACCENT_RED)
        self.is_analyzing = False
        term_print(f"  ERROR: {msg}", TERM_RED)

    def clear_history(self):
        self.history.clear()
        for w in self.history_frame.winfo_children():
            w.destroy()
        tk.Label(self.history_frame, text="No analyses yet — go to Analyze tab to begin",
                 font=("Segoe UI", 10), bg=BG_CARD, fg=TEXT_GRAY,
                 pady=40).pack()
        self.stat_total.config(text="0")
        self.stat_anemic.config(text="0")
        self.stat_normal.config(text="0")
        self.log("🗑  History cleared.")


if __name__ == "__main__":
    root = tk.Tk()
    app  = AnemiaAppPro(root)
    root.mainloop()
