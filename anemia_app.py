"""
anemia_app.py
-------------
Beautiful UI for Anemia Detection System.
Run with: python anemia_app.py
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys

# ── Try importing ML libraries ──
try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageTk
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

# ── Colors ──
BG_DARK      = "#0A0F1E"
BG_CARD      = "#111827"
BG_CARD2     = "#1A2235"
ACCENT_BLUE  = "#3B82F6"
ACCENT_CYAN  = "#06B6D4"
ACCENT_GREEN = "#10B981"
ACCENT_RED   = "#EF4444"
ACCENT_GOLD  = "#F59E0B"
TEXT_WHITE   = "#F9FAFB"
TEXT_GRAY    = "#9CA3AF"
TEXT_LIGHT   = "#E5E7EB"
BORDER       = "#1F2D45"

class AnemiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anemia Detection System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG_DARK)

        self.models_loaded = False
        self.current_image_path = None
        self.photo_img = None

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 900) // 2
        y = (self.root.winfo_screenheight() - 700) // 2
        self.root.geometry(f"900x700+{x}+{y}")

        self.build_ui()

        # Load models in background
        if ML_AVAILABLE:
            threading.Thread(target=self.load_models, daemon=True).start()
        else:
            self.set_status("⚠ Install tensorflow & pillow to use predictions", ACCENT_GOLD)

    # ─────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────
    def build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=BG_CARD, height=70)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="🔬", font=("Segoe UI Emoji", 22),
                 bg=BG_CARD, fg=ACCENT_CYAN).pack(side="left", padx=(20, 8), pady=15)

        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side="left", pady=12)
        tk.Label(title_frame, text="ANEMIA DETECTION SYSTEM",
                 font=("Segoe UI", 15, "bold"), bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w")
        tk.Label(title_frame, text="DenseNet121  •  VGG16  •  InceptionV3  Ensemble",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")

        # Status badge top right
        self.status_badge = tk.Label(header, text="⟳  Loading models...",
                                     font=("Segoe UI", 9), bg="#1C2B45",
                                     fg=ACCENT_CYAN, padx=12, pady=4)
        self.status_badge.pack(side="right", padx=20, pady=20)

        # ── Main content ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=20, pady=15)

        # Left panel — image upload
        left = tk.Frame(main, bg=BG_CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(left, text="EYE IMAGE", font=("Segoe UI", 9, "bold"),
                 bg=BG_CARD, fg=ACCENT_CYAN).pack(anchor="w", padx=15, pady=(15, 5))

        # Drop zone
        self.drop_frame = tk.Frame(left, bg=BG_CARD2, bd=0,
                                   highlightthickness=2, highlightbackground=BORDER)
        self.drop_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.image_label = tk.Label(self.drop_frame, bg=BG_CARD2,
                                    text="📷\n\nNo image selected\n\nClick Browse to upload\na conjunctiva eye photo",
                                    font=("Segoe UI", 10), fg=TEXT_GRAY,
                                    justify="center")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Browse button
        browse_btn = tk.Button(left, text="  📂  Browse Image",
                               font=("Segoe UI", 11, "bold"),
                               bg=ACCENT_BLUE, fg=TEXT_WHITE,
                               activebackground="#2563EB", activeforeground=TEXT_WHITE,
                               bd=0, padx=20, pady=10, cursor="hand2",
                               command=self.browse_image)
        browse_btn.pack(fill="x", padx=15, pady=(0, 8))

        # Analyze button
        self.analyze_btn = tk.Button(left, text="  🔍  Analyze for Anemia",
                                     font=("Segoe UI", 11, "bold"),
                                     bg=ACCENT_CYAN, fg=BG_DARK,
                                     activebackground="#0891B2", activeforeground=BG_DARK,
                                     bd=0, padx=20, pady=10, cursor="hand2",
                                     state="disabled",
                                     command=self.run_prediction)
        self.analyze_btn.pack(fill="x", padx=15, pady=(0, 15))

        # Right panel — results
        right = tk.Frame(main, bg=BG_DARK, width=340)
        right.pack(side="right", fill="both", padx=(10, 0))
        right.pack_propagate(False)

        # Result card
        self.result_card = tk.Frame(right, bg=BG_CARD, bd=0,
                                    highlightthickness=1, highlightbackground=BORDER)
        self.result_card.pack(fill="x", pady=(0, 12))

        tk.Label(self.result_card, text="DIAGNOSIS RESULT",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_CARD, fg=ACCENT_CYAN).pack(anchor="w", padx=15, pady=(15, 8))

        # Big result label
        self.result_icon = tk.Label(self.result_card, text="—",
                                    font=("Segoe UI Emoji", 36),
                                    bg=BG_CARD, fg=TEXT_GRAY)
        self.result_icon.pack(pady=(5, 0))

        self.result_label = tk.Label(self.result_card, text="No Analysis Yet",
                                     font=("Segoe UI", 18, "bold"),
                                     bg=BG_CARD, fg=TEXT_GRAY)
        self.result_label.pack()

        self.confidence_label = tk.Label(self.result_card, text="Upload an image to begin",
                                         font=("Segoe UI", 10),
                                         bg=BG_CARD, fg=TEXT_GRAY)
        self.confidence_label.pack(pady=(2, 15))

        # Confidence bar
        bar_frame = tk.Frame(self.result_card, bg=BG_CARD)
        bar_frame.pack(fill="x", padx=15, pady=(0, 15))
        tk.Label(bar_frame, text="Confidence", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=TEXT_GRAY).pack(anchor="w")
        self.progress_bg = tk.Frame(bar_frame, bg=BG_CARD2, height=8)
        self.progress_bg.pack(fill="x", pady=4)
        self.progress_fill = tk.Frame(self.progress_bg, bg=BORDER, height=8, width=0)
        self.progress_fill.place(x=0, y=0, relheight=1)

        # Individual models card
        models_card = tk.Frame(right, bg=BG_CARD, bd=0,
                               highlightthickness=1, highlightbackground=BORDER)
        models_card.pack(fill="x", pady=(0, 12))

        tk.Label(models_card, text="INDIVIDUAL MODELS",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_CARD, fg=ACCENT_CYAN).pack(anchor="w", padx=15, pady=(15, 8))

        self.model_rows = {}
        for name in ["DenseNet121", "VGG16", "InceptionV3"]:
            row = tk.Frame(models_card, bg=BG_CARD)
            row.pack(fill="x", padx=15, pady=3)
            tk.Label(row, text=name, font=("Segoe UI", 9, "bold"),
                     bg=BG_CARD, fg=TEXT_LIGHT, width=12, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="—", font=("Segoe UI", 9),
                           bg=BG_CARD, fg=TEXT_GRAY)
            lbl.pack(side="left", padx=6)
            conf = tk.Label(row, text="", font=("Segoe UI", 9),
                            bg=BG_CARD, fg=TEXT_GRAY)
            conf.pack(side="right")
            self.model_rows[name] = (lbl, conf)

        tk.Frame(models_card, bg=BG_CARD, height=10).pack()

        # Tips card
        tips_card = tk.Frame(right, bg=BG_CARD, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        tips_card.pack(fill="x")

        tk.Label(tips_card, text="💡  TIPS FOR BEST RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_CARD, fg=ACCENT_GOLD).pack(anchor="w", padx=15, pady=(15, 8))

        tips = [
            "Pull lower eyelid down gently",
            "Take close-up of inner pink tissue",
            "Use bright lighting",
            "90%+ confidence = reliable result",
            "Always confirm with a doctor",
        ]
        for tip in tips:
            tk.Label(tips_card, text=f"• {tip}", font=("Segoe UI", 8),
                     bg=BG_CARD, fg=TEXT_GRAY, anchor="w",
                     wraplength=280, justify="left").pack(anchor="w", padx=15, pady=1)
        tk.Frame(tips_card, bg=BG_CARD, height=12).pack()

        # ── Footer ──
        footer = tk.Frame(self.root, bg=BG_CARD, height=30)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer,
                 text="⚠  For educational purposes only. Always consult a qualified medical professional.",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_GRAY).pack(pady=7)

    # ─────────────────────────────────────────
    # LOGIC
    # ─────────────────────────────────────────
    def load_models(self):
        try:
            self.model_densenet  = tf.keras.models.load_model(DENSENET_MODEL)
            self.model_vgg       = tf.keras.models.load_model(VGG_MODEL)
            self.model_inception = tf.keras.models.load_model(INCEPTION_MODEL)
            self.models_loaded = True
            self.root.after(0, lambda: self.set_status("✓  Models Ready", ACCENT_GREEN))
        except Exception as e:
            self.root.after(0, lambda: self.set_status("✗  Models not found — train first", ACCENT_RED))

    def set_status(self, text, color):
        self.status_badge.config(text=text, fg=color)

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Eye Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.current_image_path = path
            self.show_preview(path)
            self.analyze_btn.config(state="normal")
            self.reset_results()

    def show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            # Fit image into drop zone
            img.thumbnail((340, 260))
            self.photo_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo_img, text="")
        except Exception:
            self.image_label.config(text="Could not preview image", image="")

    def reset_results(self):
        self.result_icon.config(text="—", fg=TEXT_GRAY)
        self.result_label.config(text="Ready to Analyze", fg=TEXT_GRAY)
        self.confidence_label.config(text="Click Analyze to get result", fg=TEXT_GRAY)
        self.progress_fill.config(bg=BORDER, width=0)
        for name, (lbl, conf) in self.model_rows.items():
            lbl.config(text="—", fg=TEXT_GRAY)
            conf.config(text="", fg=TEXT_GRAY)
        self.result_card.config(highlightbackground=BORDER)

    def run_prediction(self):
        if not self.current_image_path:
            return
        if not ML_AVAILABLE:
            self.result_label.config(text="TensorFlow not installed", fg=ACCENT_RED)
            return
        if not self.models_loaded:
            self.result_label.config(text="Models still loading...", fg=ACCENT_GOLD)
            return

        self.analyze_btn.config(state="disabled", text="  ⟳  Analyzing...")
        self.set_status("⟳  Analyzing...", ACCENT_CYAN)
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            def load_img(path, size, pre_fn):
                img = Image.open(path).convert("RGB").resize(size)
                arr = np.array(img, dtype=np.float32)
                return np.expand_dims(pre_fn(arr), 0)

            img_d = load_img(self.current_image_path, (224, 224), densenet_pre)
            img_v = load_img(self.current_image_path, (224, 224), vgg_pre)
            img_i = load_img(self.current_image_path, (299, 299), inception_pre)

            p_d = float(self.model_densenet.predict(img_d,  verbose=0)[0][0])
            p_v = float(self.model_vgg.predict(img_v,       verbose=0)[0][0])
            p_i = float(self.model_inception.predict(img_i, verbose=0)[0][0])
            p_e = (p_d + p_v + p_i) / 3.0

            self.root.after(0, lambda: self._show_results(p_d, p_v, p_i, p_e))
        except Exception as ex:
            self.root.after(0, lambda: self._show_error(str(ex)))

    def _show_results(self, p_d, p_v, p_i, p_e):
        def label_conf(p):
            if p > 0.5:
                return "ANEMIC", p * 100, ACCENT_RED, "🔴"
            else:
                return "NON-ANEMIC", (1 - p) * 100, ACCENT_GREEN, "🟢"

        lbl_d, conf_d, col_d, ico_d = label_conf(p_d)
        lbl_v, conf_v, col_v, ico_v = label_conf(p_v)
        lbl_i, conf_i, col_i, ico_i = label_conf(p_i)
        lbl_e, conf_e, col_e, ico_e = label_conf(p_e)

        # Update individual model rows
        for (name, p, lbl, col, ico) in [
            ("DenseNet121", p_d, lbl_d, col_d, ico_d),
            ("VGG16",       p_v, lbl_v, col_v, ico_v),
            ("InceptionV3", p_i, lbl_i, col_i, ico_i),
        ]:
            row_lbl, row_conf = self.model_rows[name]
            row_lbl.config(text=f"{ico}  {lbl}", fg=col)
            row_conf.config(text=f"{(p*100 if p>0.5 else (1-p)*100):.1f}%", fg=col)

        # Update main result
        self.result_icon.config(text=ico_e, fg=col_e)
        self.result_label.config(text=lbl_e, fg=col_e)
        self.confidence_label.config(
            text=f"Confidence: {conf_e:.1f}%", fg=col_e
        )
        self.result_card.config(highlightbackground=col_e)

        # Update progress bar
        self.progress_bg.update_idletasks()
        bar_w = self.progress_bg.winfo_width()
        fill_w = int(bar_w * conf_e / 100)
        self.progress_fill.config(bg=col_e, width=fill_w)

        self.analyze_btn.config(state="normal", text="  🔍  Analyze for Anemia")
        status = "✓  ANEMIC detected" if "ANEMIC" in lbl_e and "NON" not in lbl_e else "✓  NON-ANEMIC"
        self.set_status(status, col_e)

    def _show_error(self, msg):
        self.result_label.config(text="Error occurred", fg=ACCENT_RED)
        self.confidence_label.config(text=msg[:50], fg=ACCENT_RED)
        self.analyze_btn.config(state="normal", text="  🔍  Analyze for Anemia")
        self.set_status("✗  Error", ACCENT_RED)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnemiaApp(root)
    root.mainloop()
