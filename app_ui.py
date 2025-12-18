import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageColor
import joblib
import numpy as np
import os
import sys
import time
import math

# --- Import Logic ---
try:
    from features import extract_features
except ImportError:
    messagebox.showerror("Critical Error", "Missing 'features.py'. Please place it in the same folder.")
    sys.exit(1)

# --- THEME CONFIGURATION ---
ctk.set_appearance_mode("Dark")

COLOR_BG_DEEP = "#050505"     
COLOR_PANEL_BG = "#0f0f12"    
COLOR_ACCENT = "#6a5cff"      
COLOR_ACCENT_HOVER = "#8275ff" 
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_SUB = "#a1a1aa"
COLOR_BORDER = "#27272a"      

class AgriProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # =========================
        # WINDOW SETUP
        # =========================
        self.title("AgriVision Pro")
        self.geometry("1200x800")
        self.configure(fg_color=COLOR_BG_DEEP)
        
        self.after(0, lambda: self.state("zoomed"))

        self.model_path = "plant_disease_rf_model.pkl"
        self.model = self.load_model()
        self.current_image_path = None

        # =========================
        # OPTIMIZED BACKGROUND (CANVAS)
        # =========================
        # We use a Canvas for the background because moving items inside
        # a canvas is much smoother than moving a Label widget.
        self.bg_canvas = tk.Canvas(
            self, 
            bg=COLOR_BG_DEEP, 
            highlightthickness=0, 
            bd=0
        )
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Generate Spotlight Image
        self.spotlight_size = 1400
        self.spotlight_image = self.create_spotlight_image(
            self.spotlight_size, COLOR_ACCENT
        )
        
        # Add Image to Canvas
        # We save the ID to move it later
        self.spotlight_id = self.bg_canvas.create_image(
            -2000, -2000, image=self.spotlight_image, anchor="center"
        )

        # Animation Variables
        self.last_mouse_move = time.time()
        self.last_frame_time = 0  # For throttling FPS
        self.idle_angle = 0

        # Bindings
        self.bind("<Motion>", self.on_mouse_move)
        self.after(50, self.idle_float_animation)

        # =========================
        # MAIN UI CONTAINER
        # =========================
        # Note: We pack this into the root. Because the Canvas is .placed 
        # behind it, this sits on top naturally.
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=60, pady=60)

        # =========================
        # TOP PANEL: IMAGE DISPLAY
        # =========================
        self.frame_top = ctk.CTkFrame(
            self.main_container,
            fg_color=COLOR_PANEL_BG,
            corner_radius=24,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        self.frame_top.pack(side="top", fill="both", expand=True, pady=(0, 20))

        self.lbl_image = tk.Label(
            self.frame_top,
            text="DRAG & DROP IMAGE\nOR CLICK OPEN",
            bg=COLOR_PANEL_BG,
            fg="#555",
            font=("Segoe UI", 20, "bold"),
            justify="center"
        )
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")

        # =========================
        # BOTTOM PANEL: CONTROLS
        # =========================
        self.frame_bottom = ctk.CTkFrame(
            self.main_container,
            fg_color=COLOR_PANEL_BG,
            corner_radius=24,
            border_width=1,
            border_color=COLOR_BORDER,
            height=250
        )
        self.frame_bottom.pack(side="bottom", fill="x")
        self.frame_bottom.pack_propagate(False) 

        self.lbl_result = ctk.CTkLabel(
            self.frame_bottom,
            text="SYSTEM READY",
            font=("Segoe UI", 32, "bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        self.lbl_result.pack(pady=(30, 5))

        self.lbl_conf = ctk.CTkLabel(
            self.frame_bottom,
            text="Waiting for image input...",
            font=("Segoe UI", 14),
            text_color=COLOR_TEXT_SUB,
        )
        self.lbl_conf.pack(pady=(0, 25))

        self.btn_group = ctk.CTkFrame(self.frame_bottom, fg_color="transparent")
        self.btn_group.pack(fill="x", padx=100, pady=(0, 30))

        self.btn_upload = ctk.CTkButton(
            self.btn_group,
            text="OPEN IMAGE",
            font=("Segoe UI", 13, "bold"),
            height=55,
            corner_radius=12,
            fg_color="#222225",
            hover_color="#333336",
            border_width=1,
            border_color=COLOR_BORDER,
            command=self.upload_image,
        )
        self.btn_upload.pack(side="left", fill="x", expand=True, padx=(0, 15))

        self.btn_run = ctk.CTkButton(
            self.btn_group,
            text="RUN DIAGNOSIS",
            font=("Segoe UI", 13, "bold"),
            height=55,
            corner_radius=12,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="white",
            state="disabled",
            command=self.run_diagnosis,
        )
        self.btn_run.pack(side="right", fill="x", expand=True, padx=(15, 0))

        ctk.CTkLabel(
            self.frame_bottom,
            text="Designed by Haroon (23108124) | BS AI Fall 2025",
            font=("Segoe UI", 10),
            text_color="#444",
        ).place(relx=0.5, rely=0.92, anchor="center")

    # =========================
    # OPTIMIZED ANIMATION
    # =========================
    def create_spotlight_image(self, size, color):
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        center = size // 2
        max_radius = size // 2

        for r in range(max_radius, 0, -15):
            alpha = int(180 * (r / max_radius)) 
            draw.ellipse(
                (center - r, center - r, center + r, center + r),
                fill=(*ImageColor.getrgb(color), alpha),
            )
        image = image.filter(ImageFilter.GaussianBlur(120))
        return ImageTk.PhotoImage(image)

    def on_mouse_move(self, event):
        """Moves spotlight with THROTTLING to prevent flickering"""
        current_time = time.time()
        
        # Limit updates to ~60 FPS (every 0.016s)
        # This prevents the app from choking on too many events
        if current_time - self.last_frame_time < 0.015:
            return

        self.last_frame_time = current_time
        self.last_mouse_move = current_time
        
        # Use canvas.coords to move the image. This is much faster/smoother
        # than destroying/recreating a Label widget.
        self.bg_canvas.coords(self.spotlight_id, event.x, event.y)

    def idle_float_animation(self):
        """Idle float logic"""
        idle_time = time.time() - self.last_mouse_move

        if idle_time > 1.0:
            self.idle_angle += 0.02
            radius = 50 

            cx = self.winfo_width() // 2
            cy = self.winfo_height() // 2

            x = cx + math.cos(self.idle_angle) * radius
            y = cy + math.sin(self.idle_angle) * radius

            self.bg_canvas.coords(self.spotlight_id, x, y)

        self.after(30, self.idle_float_animation)

    # =========================
    # CORE LOGIC
    # =========================
    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.current_image_path = file_path

        try:
            pil_img = Image.open(file_path)
            self.update_idletasks()
            w, h = self.frame_top.winfo_width(), self.frame_top.winfo_height()
            if w < 100: w, h = 800, 500

            ratio = min(w / pil_img.width, h / pil_img.height)
            new_size = (
                int(pil_img.width * ratio * 0.90),
                int(pil_img.height * ratio * 0.90),
            )

            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(pil_img)

            self.lbl_image.config(image=self.tk_image, text="", bg=COLOR_PANEL_BG)
            self.lbl_result.configure(text="IMAGE LOADED", text_color=COLOR_TEXT_MAIN)
            self.lbl_conf.configure(text="Ready to analyze")
            self.btn_run.configure(state="normal", fg_color=COLOR_ACCENT)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def run_diagnosis(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return

        self.lbl_result.configure(text="ANALYZING...", text_color=COLOR_ACCENT)
        self.btn_run.configure(state="disabled")
        self.update()
        self.after(500) 

        try:
            features = extract_features(self.current_image_path)
            if features is None: raise ValueError("Extraction Failed")
            features = features.reshape(1, -1)

            pred = self.model.predict(features)[0]
            probs = self.model.predict_proba(features)
            conf = np.max(probs) * 100
            formatted_pred = pred.replace("_", " ").upper()
            
            result_color = "#4ade80" if "HEALTHY" in formatted_pred else COLOR_ACCENT
            
            self.lbl_result.configure(text=formatted_pred, text_color=result_color)
            self.lbl_conf.configure(text=f"CONFIDENCE SCORE: {conf:.2f}%")

        except Exception as e:
            self.lbl_result.configure(text="ERROR", text_color="#ef4444")
            self.lbl_conf.configure(text="Could not process image")
            print(e)

        self.btn_run.configure(state="normal")

if __name__ == "__main__":
    app = AgriProApp()
    app.mainloop()