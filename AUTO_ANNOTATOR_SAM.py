# (Full script — updated tracking selection/behavior + segmentation, rotation,
# class shortcuts, video-splitter, class-filter dialog, refined tracking rules)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import glob
import shutil
import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
import time

import json

def build_chunks(mapping_dict, chunk_size):
    chunks = []
    current_chunk = []
    
    for base_cls, syns in mapping_dict.items():
        group = [base_cls] + syns
        if len(current_chunk) + len(group) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            
        for item in group:
            current_chunk.append((item, base_cls))
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# Explicit imports to force PyInstaller to bundle them
try:
    import einops
    import timm
except ImportError:
    pass

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional ultralytics YOLOv8
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# =========================== LABEL HELPERS ===========================
def save_yolo_labels(label_path, boxes, img_w, img_h, save_format="bbox"):
    """
    Save axis-aligned YOLO labels (class cx cy w h normalized).
    If save_format is "polygon" and polygon data exists, saves normalized polygon points directly.
    Additionally, segmentation/polygon/rotation info is saved to a separate JSON
    with extension .seg.json containing polygons and per-box meta if present (when in bbox mode).
    """
    lines = []
    seg_data = []
    for b in boxes:
        cls = int(b.get("class", 0))
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        has_poly = "poly" in b and b["poly"]

        if save_format == "polygon" and has_poly:
            poly_norm = []
            for px, py in b["poly"]:
                poly_norm.append(f"{px/img_w:.6f}")
                poly_norm.append(f"{py/img_h:.6f}")
            lines.append(f"{cls} " + " ".join(poly_norm))
        else:
            # yolo format: class cx_norm cy_norm w_norm h_norm
            lines.append(f"{cls} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")

        # collect segmentation/rotation info if present
        meta = {}
        if has_poly:
            meta["poly"] = [[float(x), float(y)] for (x, y) in b["poly"]]
        if "seg" in b and b["seg"]:
            meta["seg"] = [[float(x), float(y)] for (x, y) in b["seg"]]
        if "rot_angle" in b:
            try:
                meta["rot_angle"] = float(b["rot_angle"])
                meta["rot_center"] = [float(b["rot_center"][0]), float(b["rot_center"][1])]
            except Exception:
                pass
        if meta:
            meta["class"] = int(cls)
            meta["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
            seg_data.append(meta)

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    seg_path = os.path.splitext(label_path)[0] + ".seg.json"
    if seg_data and save_format == "bbox":
        with open(seg_path, "w") as sf:
            json.dump(seg_data, sf, indent=2)
    else:
        # remove if exists and no seg data or in polygon mode
        if os.path.exists(seg_path):
            try:
                os.remove(seg_path)
            except Exception:
                pass


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h
            boxes.append(
                {
                    "class": int(cls),
                    "x1": cx - w / 2,
                    "y1": cy - h / 2,
                    "x2": cx + w / 2,
                    "y2": cy + h / 2,
                    # selection/tracking flags default false
                    "selected": False,
                    "track": False,
                }
            )
    # load seg extras if available
    seg_path = os.path.splitext(label_path)[0] + ".seg.json"
    if os.path.exists(seg_path):
        try:
            with open(seg_path, "r") as sf:
                seg_data = json.load(sf)
            # naive mapping: try to match by bbox and class — best effort
            for meta in seg_data:
                mcls = meta.get("class", None)
                mbbox = meta.get("bbox", None)
                mpoly = meta.get("poly", None) or meta.get("seg", None)
                mangle = meta.get("rot_angle", None)
                mcenter = meta.get("rot_center", None)
                if mbbox:
                    bx1, by1, bx2, by2 = mbbox
                    for b in boxes:
                        ix1 = max(bx1, b["x1"])
                        iy1 = max(by1, b["y1"])
                        ix2 = min(bx2, b["x2"])
                        iy2 = min(by2, b["y2"])
                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        inter = iw * ih
                        a1 = max(1e-6, (bx2 - bx1) * (by2 - by1))
                        a2 = max(1e-6, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
                        if inter > 0 and inter >= 0.4 * min(a1, a2):
                            if mpoly:
                                b["poly"] = [(float(x), float(y)) for x, y in mpoly]
                                xs = [p[0] for p in b["poly"]]
                                ys = [p[1] for p in b["poly"]]
                                b["x1"], b["y1"], b["x2"], b["y2"] = min(xs), min(ys), max(xs), max(ys)
                            if mangle is not None:
                                try:
                                    b["rot_angle"] = float(mangle)
                                    b["rot_center"] = (
                                        (float(mcenter[0]), float(mcenter[1]))
                                        if mcenter
                                        else ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
                                    )
                                except Exception:
                                    pass
                            break
        except Exception:
            pass
    return boxes

class SplashScreen(tk.Toplevel):
    def __init__(self, master, title_text="JEZT_ANNOTATOR"):
        super().__init__(master)

        self.running = True  # <── important

        self.overrideredirect(True)
        self.configure(bg="#0b0f14")

        w, h = 460, 340
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

        frame = tk.Frame(self, bg="#0b0f14")
        frame.place(relx=0.5, rely=0.5, anchor="center")

        self.title_label = tk.Label(
            frame, text=title_text,
            fg="#22e0a3", bg="#0b0f14",
            font=("Segoe UI", 22, "bold")
        )
        self.title_label.pack(pady=(30, 10))

        self.anim_label = tk.Label(
            frame, text="",
            fg="white", bg="#0b0f14",
            font=("Segoe UI", 14)
        )
        self.anim_label.pack(pady=10)

        self.status_label = tk.Label(
            frame, text="Initializing…",
            fg="#9aa3ad", bg="#0b0f14",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(pady=(10, 20))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#0b0f14",
                        bordercolor="#0b0f14",
                        background="#22e0a3",
                        lightcolor="#22e0a3",
                        darkcolor="#22e0a3")

        self.pb = ttk.Progressbar(
            frame,
            orient="horizontal",
            mode="indeterminate",
            length=260,
            style="green.Horizontal.TProgressbar"
        )
        self.pb.pack(pady=10)
        self.pb.start(22)

        self.frames = ["Loading   ", "Loading.  ", "Loading.. ", "Loading..."]
        self.idx = 0
        self.animate()

    def animate(self):
        if not self.running:
            return
        self.anim_label.config(text=self.frames[self.idx])
        self.idx = (self.idx + 1) % len(self.frames)
        self.after(180, self.animate)

    def update_status(self, text):
        if self.running:
            self.status_label.config(text=text)
            self.update_idletasks()

    def close(self):
        self.running = False
        self.destroy()


# =========================== MODEL WRAPPER ===========================
class ModelWrapper:
    """
    Tries to load YOLOv8 via ultralytics.YOLO first.
    If that fails and torch is available, tries YOLOv5 via torch.hub as a fallback.
    Provides get_class_names() to surface model class names when available.
    """
    def __init__(self, model_path=None, device="gpu" if torch.cuda.is_available() else "cpu"):
        print("GPU available: torch.cuda.is_available()")
        self.model = None
        self.ready = False
        self.device = device
        self.type = None  # 'v5' or 'v8'
        self.model_path = model_path

        if model_path:
            # Try ultralytics YOLO first (handles v8, v9, v10, v11)
            if ULTRALYTICS_AVAILABLE:
                try:
                    print("Loading YOLO model via ultralytics...")
                    self.model = YOLO(model_path)
                    self.type = 'v8+'
                    self.ready = True
                    print("YOLO model loaded successfully.")
                    return
                except Exception as e:
                    print("YOLO load failed:", e)

            # Fallback: Try YOLOv5 if torch is available
            if TORCH_AVAILABLE:
                try:
                    print("Loading YOLOv5 model (fallback)...")
                    self.model = torch.hub.load(
                        "ultralytics/yolov5", "custom", path=model_path, verbose=False
                    )
                    try:
                        self.model.to(device)
                    except Exception:
                        pass
                    self.type = 'v5'
                    self.ready = True
                    print("YOLOv5 model loaded successfully.")
                except Exception as e:
                    print("YOLOv5 load failed:", e)
                    print("Note: if you have a YOLOv8-format .pt, use ultralytics YOLO to load it.")
            else:
                print("Neither ultralytics (v8) nor torch (v5) are available to load the model.")

    def predict(self, pil_image, conf_thresh=0.25):
        """
        pil_image: PIL.Image
        returns list of boxes in original image coordinates:
          {"class": int, "x1": float, "y1": float, "x2": float, "y2": float, "score": float}
        """
        if not self.ready or self.model is None:
            return []

        preds = []
        if self.type == 'v5':
            img = np.array(pil_image)[:, :, ::-1]  # PIL RGB -> BGR numpy
            results = self.model(img)
            try:
                arr = results.xyxy[0].cpu().numpy()
                for *xyxy, conf, cls in arr:
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = xyxy
                    preds.append({"class": int(cls), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(conf)})
            except Exception:
                for r in results:
                    boxes = getattr(r, 'boxes', None)
                    if boxes is None:
                        continue
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                            if conf < conf_thresh:
                                continue
                            preds.append({"class": int(cls), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(conf)})
                    except Exception:
                        for box in boxes:
                            try:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                if conf < conf_thresh:
                                    continue
                                preds.append({"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": conf})
                            except Exception:
                                pass

        elif self.type == 'v8+':
            results = self.model(pil_image)
            for r in results:
                try:
                    if hasattr(r, 'boxes') and getattr(r.boxes, 'xyxy', None) is not None:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                            if conf < conf_thresh:
                                continue
                            preds.append({"class": int(cls), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(conf)})
                    else:
                        if hasattr(r, 'boxes') and r.boxes is not None:
                            for box in r.boxes:
                                try:
                                    coords = box.xyxy[0].tolist()
                                    x1, y1, x2, y2 = coords
                                    conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                                    cls = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                                    if conf < conf_thresh:
                                        continue
                                    preds.append({"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": conf})
                                except Exception:
                                    pass
                except Exception:
                    try:
                        for box in getattr(r, 'boxes', []):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            if conf < conf_thresh:
                                continue
                            preds.append({"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": conf})
                    except Exception:
                        pass

        # Attach default flags for selection/tracking (so later code can rely on keys)
        for p in preds:
            p.setdefault("selected", False)
            p.setdefault("track", False)

        return preds

    def get_class_names(self):
        """
        Try to infer model class names. Return list of strings.
        """
        if not self.ready or self.model is None:
            return []
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                names = self.model.model.names
                return [str(n) for n in names]
            if hasattr(self.model, 'names'):
                names = self.model.names
                return [str(n) for n in names]
            if hasattr(self.model, 'names'):
                return [str(n) for n in self.model.names]
        except Exception:
            pass
        return []

# =========================== TRACKING FUNCTION ===========================
TEMPLATE_MATCH_THRESH = 0.5  # tweak this (0.4-0.6) depending on dataset

def clamp_box(b, w, h):
    b['x1'] = max(0, min(b['x1'], w-1))
    b['x2'] = max(0, min(b['x2'], w-1))
    b['y1'] = max(0, min(b['y1'], h-1))
    b['y2'] = max(0, min(b['y2'], h-1))
    if b['x2'] < b['x1']:
        b['x1'], b['x2'] = b['x2'], b['x1']
    if b['y2'] < b['y1']:
        b['y1'], b['y2'] = b['y2'], b['y1']

def match_box_via_template(prev_img_gray, curr_img_gray, box, search_margin=1.5):
    """
    Match a previous-box template into the current image, but restrict search to a local window
    around the previous box to reduce false matches. Returns (matched_box, score).
    """
    ih, iw = prev_img_gray.shape
    x1, y1, x2, y2 = map(int, (box['x1'], box['y1'], box['x2'], box['y2']))

    x1 = max(0, min(x1, iw-1))
    x2 = max(0, min(x2, iw-1))
    y1 = max(0, min(y1, ih-1))
    y2 = max(0, min(y2, ih-1))

    tmpl = prev_img_gray[y1:y2, x1:x2]
    if tmpl.size == 0 or tmpl.shape[0] < 4 or tmpl.shape[1] < 4:
        return None, 0.0

    tw, th = tmpl.shape[1], tmpl.shape[0]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = int(max(tw * search_margin, tw + 10))
    half_h = int(max(th * search_margin, th + 10))
    sx1 = max(0, cx - half_w)
    sy1 = max(0, cy - half_h)
    sx2 = min(curr_img_gray.shape[1], cx + half_w)
    sy2 = min(curr_img_gray.shape[0], cy + half_h)

    search_region = curr_img_gray[sy1:sy2, sx1:sx2]
    if search_region.size == 0 or search_region.shape[0] < tmpl.shape[0] or search_region.shape[1] < tmpl.shape[1]:
        try:
            res = cv2.matchTemplate(curr_img_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            new_x1, new_y1 = max_loc
        except Exception:
            return None, 0.0
    else:
        res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        new_x1, new_y1 = sx1 + max_loc[0], sy1 + max_loc[1]

    new_x2 = new_x1 + tw
    new_y2 = new_y1 + th

    matched = {'x1': float(new_x1), 'y1': float(new_y1), 'x2': float(new_x2), 'y2': float(new_y2), 'class': box['class']}
    return matched, float(max_val)

# =========================== ANNOTATION APP ===========================
class AnnotatorApp:
    def __init__(self, master, image_folder, img_out, label_out, classes, model_wrapper=None, yolo_save_format="bbox", sam3_config=None):
        self.image_folder = image_folder
        self.img_out = img_out
        self.label_out = label_out
        self.classes = classes
        self.model = model_wrapper
        self.yolo_save_format = yolo_save_format
        self.sam3_config = sam3_config
        self.tracking_mode = False

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
        self.image_paths = []
        for e in exts:
            self.image_paths.extend(sorted(glob.glob(os.path.join(image_folder, e))))
        if not self.image_paths:
            messagebox.showerror("Error", "No images found in selected folder.")
            sys.exit(1)

        self.index = 0
        self.boxes = []

        # segmentation / polygon state
        self.segmentation_mode = False
        self.current_polygon = []

        # rotate state
        self.rotate_mode = False
        self.rotate_box_idx = None
        self.rotate_initial_angle = 0.0

        # model-class filtering
        self.model_class_names = []
        self.selected_model_classes = set()
        
        from collections import OrderedDict
        self.histories = OrderedDict()
        self.resize_mode = 'none'
        
        self.global_resize_active = False
        self.global_resize_mult = 1.0

        if self.model and self.model.ready:
            try:
                self.model_class_names = self.model.get_class_names() or []
            except Exception:
                self.model_class_names = []
            for name in self.model_class_names:
                if name in self.classes:
                    self.selected_model_classes.add(name)
        else:
            self.model_class_names = []

        self.root = tk.Toplevel(master)
        self.root.title("Auto Annotator + Tracking")

        self.top_label = tk.Label(self.root, text="", anchor="w")
        self.top_label.pack(fill=tk.X)

        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=3)

        self.class_var = tk.IntVar(value=0)
        self.class_menu = tk.OptionMenu(ctrl_frame, self.class_var, *list(range(len(classes))))
        self.class_menu.pack(side=tk.LEFT, anchor="w")

        self.class_select_btn = tk.Button(ctrl_frame, text="Select Auto-Annotate Classes", command=self.open_class_selection_dialog)
        self.class_select_btn.pack(side=tk.LEFT, padx=6)
        
        self.florence_btn = tk.Button(ctrl_frame, text="Annotate with LLM (Florence)", command=self.open_florence_dialog)
        self.florence_btn.pack(side=tk.LEFT, padx=6)

        self.canvas = tk.Canvas(self.root, bg="black", width=1280, height=720, cursor="tcross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("n", self.next_image)
        self.root.bind("p", self.prev_image)
        self.root.bind("a", self.auto_annotate)
        self.root.bind("d", self.delete_box)
        self.root.bind("<Control-s>", self.save_labels)
        self.root.bind("t", self.toggle_tracking)
        self.root.bind("g", self.toggle_segmentation)
        self.root.bind("r", self.toggle_rotate_mode)
        self.root.bind("z", self.undo_action)
        self.root.bind("h", self.cycle_resize_mode)
        
        self.root.bind("<Up>", lambda e: self.move_boxes(0, -2))
        self.root.bind("<Down>", lambda e: self.move_boxes(0, 2))
        self.root.bind("<Left>", lambda e: self.move_boxes(-2, 0))
        self.root.bind("<Right>", lambda e: self.move_boxes(2, 0))

        for i in range(10):
            self.root.bind(str(i), lambda e, v=i: self.set_class_number(v))
        self.root.bind("s", self.cycle_class)

        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-4>", self.mouse_wheel_linux)
        self.canvas.bind("<Button-5>", self.mouse_wheel_linux)

        self.root.bind("<Return>", self.finish_polygon)
        self.root.bind("<Escape>", self.cancel_modes)
        self.root.bind("c", self.finish_polygon)

        self.drag_start = None
        self.new_rect = None

        self.awaiting_track_selection = False
        self.prev_image_cv = None
        self.scale = 1.0
        
        if self.sam3_config:
            self.root.after(100, self.run_sam3_bulk)
        else:
            self._safe_load_image(0)
            
        self.root.mainloop()

    def run_sam3_bulk(self):
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Open Vocabulary Labeling")
        prog_win.geometry("450x180")
        
        lbl = tk.Label(prog_win, text="Initializing SAM3...", pady=10, font=("Segoe UI", 10))
        lbl.pack()
        
        progress = ttk.Progressbar(prog_win, orient="horizontal", length=350, mode="determinate")
        progress.pack(pady=10)
        
        def task():
            try:
                import torch
                model_path = self.sam3_config.get("model_path", "sam3.pt")
                if not model_path.endswith('.pt') and not model_path.endswith('.pth') and not model_path.endswith('.engine'):
                    model_path += '.pt'
                quant = self.sam3_config.get("quantization", "float16")
                model_type = self.sam3_config.get("model_type", "SAM3")
                
                lbl.config(text=f"Loading {model_path} into memory...")
                prog_win.update()
                
                if model_type == "YOLOE":
                    from ultralytics import YOLO
                    predictor = YOLO(model_path)
                else:
                    from ultralytics.models.sam import SAM3SemanticPredictor
                    overrides = {
                        "model": model_path,
                        "task": "segment",
                        "mode": "predict",
                        "conf": 0.5,
                        "imgsz": 644,
                        "half": (quant == "float16")
                    }
                    predictor = SAM3SemanticPredictor(overrides=overrides)
                
                total = len(self.image_paths)
                progress["maximum"] = total
                
                target_classes_str = self.sam3_config.get("target_classes", ",".join(self.classes))
                try:
                    import json
                    mapping_dict = json.loads(target_classes_str)
                except Exception:
                    mapping_dict = {c.strip(): [] for c in target_classes_str.split(",") if c.strip()}
                    
                chunk_sz = int(self.sam3_config.get("chunk_size", "4"))
                chunks = build_chunks(mapping_dict, chunk_sz)
                
                for i, p in enumerate(self.image_paths):
                    lbl.config(text=f"Labeling {i+1} / {total}\n{os.path.basename(p)}")
                    img = Image.open(p).convert("RGB")
                    
                    w, h = img.size
                    boxes = []
                    
                    for chunk in chunks:
                        prompts = [pmt for pmt, base_cls in chunk]
                        with torch.no_grad():
                            if model_type == "YOLOE":
                                predictor.set_classes(prompts)
                                res = predictor(img, verbose=True)
                            else:
                                res = predictor(img, text=prompts)
                            
                        for r in res:
                            has_masks = r.masks is not None
                            if r.boxes:
                                bxyxy = r.boxes.xyxy.cpu().numpy()
                                bcls = r.boxes.cls.cpu().numpy()
                                segments = []
                                if has_masks:
                                    segments = r.masks.xy
                                    
                                for idx in range(len(bxyxy)):
                                    local_id = int(bcls[idx])
                                    if local_id < len(chunk):
                                        prompt, base_cls = chunk[local_id]
                                        c = self.classes.index(base_cls) if base_cls in self.classes else -1
                                        if c != -1:
                                            x1, y1, x2, y2 = float(bxyxy[idx][0]), float(bxyxy[idx][1]), float(bxyxy[idx][2]), float(bxyxy[idx][3])
                                            
                                            mult = float(self.sam3_config.get("global_resize_mult", 1.0))
                                            shift_x = float(self.sam3_config.get("global_shift_x", 0.0))
                                            shift_y = float(self.sam3_config.get("global_shift_y", 0.0))
                                            
                                            if mult != 1.0 or shift_x != 0.0 or shift_y != 0.0:
                                                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                                bw, bh = (x2 - x1) * mult, (y2 - y1) * mult
                                                cx += shift_x
                                                cy += shift_y
                                                x1, y1, x2, y2 = cx - bw/2.0, cy - bh/2.0, cx + bw/2.0, cy + bh/2.0
                                                
                                            box_data = {
                                                "class": c,
                                                "x1": x1, "y1": y1,
                                                "x2": x2, "y2": y2
                                            }
                                            
                                            if has_masks and idx < len(segments):
                                                if self.yolo_save_format == "polygon":
                                                    poly = segments[idx].tolist()
                                                    if poly:
                                                        box_data["poly"] = poly
                                                
                                            boxes.append(box_data)
                                
                    base = os.path.splitext(os.path.basename(p))[0]
                    lpath = os.path.join(self.label_out, base + ".txt")
                    
                    save_yolo_labels(lpath, boxes, w, h, save_format=self.yolo_save_format)
                    shutil.copy2(p, os.path.join(self.img_out, os.path.basename(p)))
                    
                    progress["value"] = i + 1
                    prog_win.update()
                    
                lbl.config(text="Done!")
                time.sleep(0.5)
            except Exception as e:
                messagebox.showerror("SAM3 Error", str(e))
            finally:
                prog_win.destroy()
                
                # Cleanup VRAM
                if 'predictor' in locals():
                    del predictor
                if 'torch' in sys.modules:
                    sys.modules['torch'].cuda.empty_cache()
                
                self.root.after(0, lambda: self._safe_load_image(0))
                
        threading.Thread(target=task, daemon=True).start()

    # ------------------- State / History -------------------
    def save_state(self):
        import copy
        p = self.image_paths[self.index]
        if p not in self.histories:
            self.histories[p] = []
            if len(self.histories) > 100:
                self.histories.popitem(last=False)
                
        self.histories[p].append(copy.deepcopy(self.boxes))
        if len(self.histories[p]) > 30:
            self.histories[p].pop(0)

    def undo_action(self, event=None):
        p = self.image_paths[self.index]
        if p in self.histories and self.histories[p]:
            self.boxes = self.histories[p].pop()
            self.redraw()

    # ------------------- Helpers -------------------
    def _safe_load_image(self, idx, initial_boxes=None, retries=10):
      if not self.canvas.winfo_exists():
        return

      canvas_w = self.canvas.winfo_width()
      canvas_h = self.canvas.winfo_height()

      if (canvas_w < 10 or canvas_h < 10) and retries > 0:
        self.root.after(50, lambda: self._safe_load_image(idx, initial_boxes, retries-1))
        return

      self.load_image(idx, initial_boxes)

    def current_label_path(self):
        base = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0]
        return os.path.join(self.label_out, base + ".txt")

    def current_image_out(self):
        return os.path.join(self.img_out, os.path.basename(self.image_paths[self.index]))

    def update_top_label(self):
        tracking_status = "ON" if self.tracking_mode else "OFF"
        awaiting = " (select boxes then press 't' to start tracking)" if self.awaiting_track_selection else ""
        seg_status = "ON" if self.segmentation_mode else "OFF"
        rotate_status = "ON" if self.rotate_mode else "OFF"
        resize_status = self.resize_mode
        if self.model and self.model.ready and self.model_class_names:
            if self.selected_model_classes:
                sel = ", ".join(sorted(self.selected_model_classes))
                model_info = f"Auto-classes: [{sel}]"
            else:
                model_info = "Auto-classes: [none]"
        else:
            model_info = "Auto-classes: [model missing or names unknown]"
        self.top_label.config(
            text=f"[{self.index+1}/{len(self.image_paths)}] {os.path.basename(self.image_paths[self.index])} | Tracking: {tracking_status}{awaiting} | Seg: {seg_status} | Rotate: {rotate_status} | Resize: {resize_status} | {model_info}\nn=next, p=prev, a=auto, d=delete, Ctrl+S=save, t=toggle tracking, g=toggle seg, r=rotate, h=resize mode, z=undo, arrows=move, 0-9=set class, s=cycle class"
        )

    # ------------------- Drawing -------------------
    def draw_box(self, box):
        x1 = box["x1"] * self.scale
        y1 = box["y1"] * self.scale
        x2 = box["x2"] * self.scale
        y2 = box["y2"] * self.scale
        
        palette = ["lime", "cyan", "yellow", "dodger blue", "magenta", "orange", "spring green", "deep sky blue", "gold", "medium purple"]
        cls_color = palette[box["class"] % len(palette)]
        
        if box.get("track", False):
            color = "blue"
            text_color = "blue"
        elif box.get("selected", False):
            color = "red"
            text_color = "red"
        else:
            color = cls_color
            text_color = cls_color

        if "poly" in box and box["poly"]:
            coords = []
            for (px, py) in box["poly"]:
                coords.extend([px * self.scale, py * self.scale])
            poly_id = self.canvas.create_polygon(coords, outline=color, fill="", width=2)
            name = self.classes[box["class"]] if box["class"] < len(self.classes) else str(box["class"])
            first_x = box["poly"][0][0] * self.scale
            first_y = box["poly"][0][1] * self.scale
            self.canvas.create_text(first_x + 5, first_y + 5, anchor="nw", text=name, fill=text_color)
            return poly_id
        else:
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            cls = box["class"]
            name = self.classes[cls] if cls < len(self.classes) else str(cls)
            self.canvas.create_text(x1 + 5, y1 + 5, anchor="nw", text=name, fill=text_color)
            return rect

    def redraw(self):
        self.canvas.delete("all")
        if hasattr(self, "tkimg") and self.tkimg is not None:
            if hasattr(self, "tkimg") and self.tkimg and self.canvas.winfo_exists():
             self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        else:
            self.canvas.create_rectangle(0, 0, 10, 10, fill="black")

        for b in self.boxes:
            b["id"] = self.draw_box(b)

        if self.segmentation_mode and self.current_polygon:
            pts = []
            for (px, py) in self.current_polygon:
                sx, sy = px * self.scale, py * self.scale
                self.canvas.create_oval(sx-3, sy-3, sx+3, sy+3, fill="cyan", outline="white")
                pts.extend([sx, sy])
            if len(self.current_polygon) > 1:
                self.canvas.create_line(pts, fill="cyan", width=2)

    # ------------------- Mouse -------------------
    def mouse_down(self, event):
        ix, iy = event.x / self.scale, event.y / self.scale
        self.drag_start = (ix, iy)

        if self.segmentation_mode:
            self.save_state()
            self.current_polygon.append((ix, iy))
            self.redraw()
            return

        self.new_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x + 1, event.y + 1, outline="red", dash=(2, 2)
        )

    def mouse_drag(self, event):
        if self.new_rect:
            self.canvas.coords(
                self.new_rect,
                self.drag_start[0] * self.scale,
                self.drag_start[1] * self.scale,
                event.x,
                event.y,
            )

    def mouse_up(self, event):
        ix, iy = event.x / self.scale, event.y / self.scale

        if self.segmentation_mode:
            return

        if not self.new_rect:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x / self.scale, event.y / self.scale

        if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
            click_x, click_y = x2, y2
            toggled = False
            for b in reversed(self.boxes):
                if "poly" in b and b["poly"]:
                    if self.point_in_poly((click_x, click_y), b["poly"]):
                        b["selected"] = not b.get("selected", False)
                        if self.tracking_mode:
                            b["track"] = bool(b.get("selected", False))
                        toggled = True
                        break
                else:
                    if b["x1"] <= click_x <= b["x2"] and b["y1"] <= click_y <= b["y2"]:
                        b["selected"] = not b.get("selected", False)
                        if self.tracking_mode:
                            b["track"] = bool(b.get("selected", False))
                        toggled = True
                        break
            self.canvas.delete(self.new_rect)
            self.new_rect = None
            self.redraw()
            return

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            self.save_state()
            cls = int(self.class_var.get())
            box = {"class": cls, "x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)}
            box["selected"] = False
            box["track"] = False
            box["id"] = self.draw_box(box)
            self.boxes.append(box)

        self.canvas.delete(self.new_rect)
        self.new_rect = None

    # ------------------- Rotation helpers (FIXED) -------------------
    def _apply_rotation_delta(self, delta_angle):
        if not self.rotate_mode or self.rotate_box_idx is None:
            return
        b = self.boxes[self.rotate_box_idx]

        # Keep a copy of the original axis-aligned box for stable rotation
        if "base_x1" not in b:
            b["base_x1"], b["base_y1"], b["base_x2"], b["base_y2"] = b["x1"], b["y1"], b["x2"], b["y2"]

        cx = (b["base_x1"] + b["base_x2"]) / 2.0
        cy = (b["base_y1"] + b["base_y2"]) / 2.0
        b.setdefault("rot_center", (cx, cy))
        b.setdefault("rot_angle", 0.0)

        new_angle = b["rot_angle"] + delta_angle

        # rotate from the original, not from last poly
        corners = [
            (b["base_x1"], b["base_y1"]),
            (b["base_x2"], b["base_y1"]),
            (b["base_x2"], b["base_y2"]),
            (b["base_x1"], b["base_y2"]),
        ]
        rot_corners = []
        cos_a = math.cos(new_angle)
        sin_a = math.sin(new_angle)
        for (px, py) in corners:
            dx = px - cx
            dy = py - cy
            rdx = dx * cos_a - dy * sin_a
            rdy = dx * sin_a + dy * cos_a
            rx, ry = cx + rdx, cy + rdy
            # clamp to image
            rx = max(0, min(rx, self.img_w - 1))
            ry = max(0, min(ry, self.img_h - 1))
            rot_corners.append((rx, ry))

        b["poly"] = rot_corners
        b["rot_angle"] = new_angle
        b["rot_center"] = (cx, cy)

        xs = [p[0] for p in rot_corners]
        ys = [p[1] for p in rot_corners]
        b["x1"], b["y1"], b["x2"], b["y2"] = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

        # final clamp for bbox
        b["x1"] = max(0, min(b["x1"], self.img_w - 1))
        b["x2"] = max(0, min(b["x2"], self.img_w - 1))
        b["y1"] = max(0, min(b["y1"], self.img_h - 1))
        b["y2"] = max(0, min(b["y2"], self.img_h - 1))

        self.redraw()

    def cycle_resize_mode(self, event=None):
        modes = ['none', 'height', 'width', 'both']
        idx = modes.index(self.resize_mode)
        self.resize_mode = modes[(idx + 1) % len(modes)]
        self.update_top_label()

    def _apply_resize_delta(self, delta_amount):
        selected_indices = [i for i, b in enumerate(self.boxes) if b.get("selected", False)]
        if not selected_indices:
            return
            
        for idx in selected_indices:
            b = self.boxes[idx]
            if "poly" in b and b["poly"]:
                continue
                
            cx = (b["x1"] + b["x2"]) / 2.0
            cy = (b["y1"] + b["y2"]) / 2.0
            w = b["x2"] - b["x1"]
            h = b["y2"] - b["y1"]
            
            new_w = w
            new_h = h
            
            if self.resize_mode in ['width', 'both']:
                new_w = max(5, w + delta_amount)
            if self.resize_mode in ['height', 'both']:
                new_h = max(5, h + delta_amount)
                
            b["x1"] = cx - new_w / 2.0
            b["x2"] = cx + new_w / 2.0
            b["y1"] = cy - new_h / 2.0
            b["y2"] = cy + new_h / 2.0
            
            clamp_box(b, self.img_w, self.img_h)
            
        self.redraw()

    def mouse_wheel(self, event):
        if self.global_resize_active:
            if event.delta > 0:
                self._apply_global_resize(1.02)
            elif event.delta < 0:
                self._apply_global_resize(1 / 1.02)
            return

        if self.rotate_mode:
            step = math.radians(2)
            if event.delta > 0:
                self._apply_rotation_delta(step)
            elif event.delta < 0:
                self._apply_rotation_delta(-step)
        elif self.resize_mode != 'none':
            if event.delta > 0:
                self._apply_resize_delta(4)
            elif event.delta < 0:
                self._apply_resize_delta(-4)

    def mouse_wheel_linux(self, event):
        if self.global_resize_active:
            if event.num == 4:
                self._apply_global_resize(1.02)
            elif event.num == 5:
                self._apply_global_resize(1 / 1.02)
            return

        if self.rotate_mode:
            step = math.radians(2)
            if event.num == 4:
                self._apply_rotation_delta(step)
            elif event.num == 5:
                self._apply_rotation_delta(-step)
        elif self.resize_mode != 'none':
            if event.num == 4:
                self._apply_resize_delta(4)
            elif event.num == 5:
                self._apply_resize_delta(-4)

    def _apply_global_resize(self, mult):
        self.global_resize_mult *= mult
        for b in self.boxes:
            if b.get("poly"): continue
            cx = (b["x1"] + b["x2"]) / 2.0
            cy = (b["y1"] + b["y2"]) / 2.0
            bw = (b["x2"] - b["x1"]) * mult
            bh = (b["y2"] - b["y1"]) * mult
            b["x1"] = cx - bw / 2.0
            b["x2"] = cx + bw / 2.0
            b["y1"] = cy - bh / 2.0
            b["y2"] = cy + bh / 2.0
            clamp_box(b, self.img_w, self.img_h)
        self.redraw()

    # ------------------- Geometric helpers -----------------------------
    def finish_polygon(self, event=None):
        if not self.segmentation_mode or not self.current_polygon:
            return
        xs = [p[0] for p in self.current_polygon]
        ys = [p[1] for p in self.current_polygon]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cls = int(self.class_var.get())
        box = {"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        box["seg"] = list(self.current_polygon)
        box["poly"] = list(self.current_polygon)
        box["selected"] = False
        box["track"] = False
        self.save_state()
        self.boxes.append(box)
        self.current_polygon = []
        self.redraw()

    def cancel_modes(self, event=None):
        if self.segmentation_mode and self.current_polygon:
            self.current_polygon = []
            self.redraw()
            return
        if self.rotate_mode:
            self.toggle_rotate_mode()
            return

    def point_in_poly(self, point, poly):
        x, y = point
        inside = False
        n = len(poly)
        if n < 3:
            return False
        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # ------------------- Actions -------------------
    def save_labels(self, event=None):
        to_save = []
        for b in self.boxes:
            b_copy = b.copy()
            if "poly" in b_copy and b_copy["poly"]:
                xs = [p[0] for p in b_copy["poly"]]
                ys = [p[1] for p in b_copy["poly"]]
                b_copy["x1"], b_copy["y1"], b_copy["x2"], b_copy["y2"] = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            to_save.append(b_copy)
        save_yolo_labels(self.current_label_path(), to_save, self.img_w, self.img_h, save_format=self.yolo_save_format)
        shutil.copy2(self.image_paths[self.index], self.current_image_out())
        print("Saved:", self.current_label_path())

    def filter_predictions_by_selected_classes(self, preds):
        if not preds:
            return []
        if self.model and self.model.ready:
            model_names = self.model_class_names
            if model_names:
                allowed_idxs = set()
                for idx, name in enumerate(model_names):
                    if name in self.selected_model_classes:
                        allowed_idxs.add(idx)
                filtered = [p for p in preds if int(p.get("class", -1)) in allowed_idxs]
                return filtered
            else:
                return [p for p in preds if int(p.get("class", -1)) < len(self.classes)]
        else:
            return []

    def load_image(self, idx, initial_boxes=None):
        self.index = max(0, min(len(self.image_paths) - 1, idx))
        path = self.image_paths[self.index]

        self.image = Image.open(path).convert("RGB")
        self.img_w, self.img_h = self.image.size

        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if not canvas_w or not canvas_h or canvas_w < 10 or canvas_h < 10:
            self.root.after(50, lambda: self.load_image(idx, initial_boxes))
            return

        self.scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        resized_w, resized_h = int(self.img_w * self.scale), int(self.img_h * self.scale)
        resized_w = max(1, resized_w)
        resized_h = max(1, resized_h)

        try:
            self.tkimg = ImageTk.PhotoImage(self.image.resize((resized_w, resized_h), Image.Resampling.LANCZOS))
            self.canvas.image = self.tkimg
        except Exception:
            self.tkimg = ImageTk.PhotoImage(self.image.resize((resized_w, resized_h)))

        self.canvas.delete("all")
        if hasattr(self, "tkimg") and self.tkimg:
         self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        self.update_top_label()

        tracked_passed = None
        if initial_boxes is not None:
            tracked_passed = []
            for b in initial_boxes:
                b.setdefault("selected", False)
                b["track"] = True
                tracked_passed.append(b)

        saved = load_yolo_labels(self.current_label_path(), self.img_w, self.img_h)
        for b in saved:
            b.setdefault("selected", False)
            b.setdefault("track", False)

        autos = []
        if self.model and self.model.ready:
            if self.tracking_mode:
                autos = self.model.predict(self.image)
            else:
                if not saved:
                    autos = self.model.predict(self.image)
        else:
            autos = []

        autos_filtered = self.filter_predictions_by_selected_classes(autos)

        autos_proc = []
        for p in autos_filtered:
            p.setdefault("selected", False)
            p.setdefault("track", False)
            autos_proc.append(p)

        if saved and not self.tracking_mode and tracked_passed is None:
            self.boxes = saved
        else:
            base = []
            if autos_proc:
                base.extend(autos_proc)
            elif not saved:
                base = []
            self.boxes = base

        if tracked_passed:
            for tb in tracked_passed:
                tb.setdefault("selected", False)
                tb["track"] = True
                self.boxes.append(tb)

        for b in self.boxes:
            b.setdefault("selected", False)
            b.setdefault("track", False)

        self.prev_image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        self.redraw()

    def next_image(self, event=None):
        self.save_labels()

        if self.index >= len(self.image_paths) - 1:
            messagebox.showinfo("Done", "Reached last image.")
            return

        new_boxes = None
        if self.tracking_mode and self.prev_image_cv is not None and self.index < len(self.image_paths) - 1:
            prev_gray = cv2.cvtColor(self.prev_image_cv, cv2.COLOR_BGR2GRAY)

            next_path = self.image_paths[self.index + 1]
            next_img = cv2.imread(next_path)
            if next_img is None:
                print("Warning: couldn't open next image for tracking:", next_path)
                next_gray = None
            else:
                next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
                candidate_boxes = []
                boxes_to_track = [b for b in self.boxes if b.get("track", False)]
                if boxes_to_track:
                    for b in boxes_to_track:
                        matched_box, score = match_box_via_template(prev_gray, next_gray, b, search_margin=1.8)
                        if matched_box and score >= TEMPLATE_MATCH_THRESH:
                            clamp_box(matched_box, next_img.shape[1], next_img.shape[0])
                            matched_box["class"] = b["class"]
                            matched_box["selected"] = False
                            matched_box["track"] = True
                            candidate_boxes.append(matched_box)
                new_boxes = candidate_boxes

        self.index += 1
        self.load_image(self.index, initial_boxes=new_boxes)

    def prev_image(self, event=None):
        if self.index > 0:
            self.save_labels()
            self.index -= 1
            self.load_image(self.index)

    def delete_box(self, event=None):
        selected_indices = [i for i, b in enumerate(self.boxes) if b.get("selected", False)]
        if selected_indices:
            self.save_state()
            for i in sorted(selected_indices, reverse=True):
                self.boxes.pop(i)
        else:
            if self.boxes:
                self.save_state()
                self.boxes.pop()
        self.redraw()

    def move_boxes(self, dx, dy):
        selected_indices = [i for i, b in enumerate(self.boxes) if b.get("selected", False)]
        if not selected_indices:
            return
            
        self.save_state()
        for idx in selected_indices:
            b = self.boxes[idx]
            if "poly" in b and b["poly"]:
                new_poly = []
                for px, py in b["poly"]:
                    nx = max(0, min(px + dx, self.img_w - 1))
                    ny = max(0, min(py + dy, self.img_h - 1))
                    new_poly.append((nx, ny))
                b["poly"] = new_poly
                xs = [p[0] for p in new_poly]
                ys = [p[1] for p in new_poly]
                b["x1"], b["y1"], b["x2"], b["y2"] = min(xs), min(ys), max(xs), max(ys)
            else:
                w = b["x2"] - b["x1"]
                h = b["y2"] - b["y1"]
                b["x1"] = max(0, min(b["x1"] + dx, self.img_w - 1 - w))
                b["y1"] = max(0, min(b["y1"] + dy, self.img_h - 1 - h))
                b["x2"] = b["x1"] + w
                b["y2"] = b["y1"] + h
        self.redraw()

    def auto_annotate(self, event=None):
        if not self.model or not self.model.ready:
            print("No model loaded. Skipping auto-annotation.")
            return
        preds = self.model.predict(self.image)
        preds = self.filter_predictions_by_selected_classes(preds)
        self.boxes = []
        for p in preds:
            p.setdefault("selected", False)
            p.setdefault("track", False)
            self.boxes.append(p)
        for p in self.boxes:
            p["id"] = self.draw_box(p)
        print(f"Auto-annotated {len(preds)} boxes (filtered).")

    def toggle_tracking(self, event=None):
        if not self.tracking_mode and not self.awaiting_track_selection:
            self.boxes = [b for b in self.boxes if not b.get("track", False)]
            for b in self.boxes:
                b["track"] = False
            self.awaiting_track_selection = True
            messagebox.showinfo("Select boxes", "Select boxes you want to track (multi-select). When ready, press 't' again to start tracking. If you press 't' without selecting anything, tracking will start but no boxes will be tracked until you select them.")
            self.update_top_label()
            self.redraw()
            return

        if not self.tracking_mode and self.awaiting_track_selection:
            selected_any = any(b.get("selected", False) for b in self.boxes)
            if selected_any:
                for b in self.boxes:
                    b["track"] = bool(b.get("selected", False))
                    b["selected"] = False
            else:
                for b in self.boxes:
                    b["track"] = False
            self.tracking_mode = True
            self.awaiting_track_selection = False
            messagebox.showinfo("Tracking", "Tracking started. Select boxes to mark them tracked (blue).")
            self.update_top_label()
            self.redraw()
            return

        if self.tracking_mode:
            self.boxes = [b for b in self.boxes if not b.get("track", False)]
            for b in self.boxes:
                b["track"] = False
            self.tracking_mode = False
            self.awaiting_track_selection = False
            messagebox.showinfo("Tracking", "Tracking stopped. Tracked boxes removed.")
            self.update_top_label()
            self.redraw()
            return

    # ------------------- Segmentation -------------------
    def toggle_segmentation(self, event=None):
        self.segmentation_mode = not self.segmentation_mode
        if not self.segmentation_mode:
            self.current_polygon = []
        self.update_top_label()
        self.redraw()

    # ------------------- Rotation -------------------
    def toggle_rotate_mode(self, event=None):
        if not self.rotate_mode:
            selected_indices = [i for i, b in enumerate(self.boxes) if b.get("selected", False)]
            if not selected_indices:
                messagebox.showinfo("Rotate", "Please select a box (click it) and press 'r' again to start rotating.")
                self.rotate_mode = False
                return
            idx = selected_indices[0]
            self.rotate_mode = True
            self.rotate_box_idx = idx
            b = self.boxes[idx]

            # store base box for rotation
            b.setdefault("base_x1", b["x1"])
            b.setdefault("base_y1", b["y1"])
            b.setdefault("base_x2", b["x2"])
            b.setdefault("base_y2", b["y2"])

            cx = (b["base_x1"] + b["base_x2"]) / 2.0
            cy = (b["base_y1"] + b["base_y2"]) / 2.0
            b.setdefault("rot_center", (cx, cy))
            b.setdefault("rot_angle", 0.0)
            if "poly" not in b or not b["poly"]:
                corners = [
                    (b["base_x1"], b["base_y1"]),
                    (b["base_x2"], b["base_y1"]),
                    (b["base_x2"], b["base_y2"]),
                    (b["base_x1"], b["base_y2"]),
                ]
                b["poly"] = corners
            self.rotate_initial_angle = float(b.get("rot_angle", 0.0))
            messagebox.showinfo("Rotate", "Rotate mode ON. Use the mouse scroll wheel to rotate the selected box. Press 'r' or Esc to exit rotate mode.")
            self.update_top_label()
            self.redraw()
        else:
            self.rotate_mode = False
            self.rotate_box_idx = None
            self.rotate_initial_angle = 0.0
            messagebox.showinfo("Rotate", "Rotate mode OFF.")
            self.update_top_label()
            self.redraw()

    # ------------------- Class shortcuts -------------------
    def set_class_number(self, n):
        if n < len(self.classes):
            self.class_var.set(n)
            
            selected_any = False
            for b in self.boxes:
                if b.get("selected", False):
                    if not selected_any:
                        self.save_state()
                    b["class"] = n
                    selected_any = True
            
            if selected_any:
                self.redraw()
            self.update_top_label()

    def cycle_class(self, event=None):
        selected_indices = [i for i, b in enumerate(self.boxes) if b.get("selected", False)]
        if selected_indices:
            self.save_state()
            for idx in selected_indices:
                b = self.boxes[idx]
                b["class"] = (b["class"] + 1) % max(1, len(self.classes))
            self.redraw()
        else:
            cur = int(self.class_var.get())
            nxt = (cur + 1) % max(1, len(self.classes))
            self.class_var.set(nxt)
            self.update_top_label()

    # ------------------- Class selection dialog -------------------
    def open_class_selection_dialog(self):
        if not self.model or not self.model.ready:
            messagebox.showwarning("No model", "No model loaded or model not ready. Please load a model in the startup GUI.")
            return
        model_names = self.model.get_class_names()
        if not model_names:
            resp = messagebox.askyesno("Model names unavailable", "Model did not expose class names. Auto-annotation will fall back to index-based filtering (class index < number of startup classes). Do you still want to open a selection dialog?")
            if not resp:
                return
            model_names = [str(i) for i in range(max(10, len(self.classes)))]

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Auto-Annotate Classes")
        dialog.geometry("400x400")

        tk.Label(dialog, text="Check the model classes you want to use for auto-annotation:", anchor="w").pack(fill=tk.X, padx=8, pady=6)

        container = tk.Frame(dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        var_map = {}
        for name in model_names:
            v = tk.IntVar(value=1 if (name in self.selected_model_classes) else 0)
            chk = tk.Checkbutton(scroll_frame, text=name, variable=v, anchor="w")
            chk.pack(fill=tk.X, anchor="w")
            var_map[name] = v

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=6)

        def on_ok():
            newset = set()
            for nm, vv in var_map.items():
                if vv.get():
                    newset.add(nm)
            self.selected_model_classes = newset
            dialog.destroy()
            self.update_top_label()
            self.redraw()

        def on_cancel():
            dialog.destroy()

        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=8)

    def open_florence_dialog(self):
        FlorenceDialog(self)

# =========================== STARTUP GUI (with Video Splitter page) ===========================
class VideoSplitterGUI:
    def __init__(self, master=None):
        self.root = tk.Toplevel(master) if master else tk.Tk()
        self.root.title("Video Splitter")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.mode = tk.StringVar(value="gap")

        self.total_frames = 0
        self.preview_image = None

        self.make_widgets()

    # ---------------- UI ----------------
    def make_widgets(self):
        def path_row(label, var, cmd):
            f = tk.Frame(self.root)
            f.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(f, text=label, width=15, anchor="w").pack(side=tk.LEFT)
            e = tk.Entry(f, textvariable=var, width=50)
            e.pack(side=tk.LEFT, padx=5)
            tk.Button(f, text="Browse", command=cmd).pack(side=tk.LEFT)
            return e

        path_row("Video file:", self.video_path, self.browse_video)
        path_row("Output folder:", self.output_dir, self.browse_output)

        lf = tk.LabelFrame(self.root, text="Extraction Mode")
        lf.pack(fill=tk.X, padx=10, pady=5)

        tk.Radiobutton(lf, text="Frame Gap", variable=self.mode, value="gap",
                       command=self.update_count).pack(anchor="w")

        f1 = tk.Frame(lf)
        f1.pack(anchor="w", padx=20)
        tk.Label(f1, text="Every Nth frame:").pack(side=tk.LEFT)
        self.frame_gap_entry = tk.Entry(f1, width=8)
        self.frame_gap_entry.insert(0, "10")
        self.frame_gap_entry.pack(side=tk.LEFT)
        self.frame_gap_entry.bind("<KeyRelease>", lambda e: self.update_count())

        tk.Radiobutton(lf, text="Target Images", variable=self.mode, value="target",
                       command=self.update_count).pack(anchor="w")

        f2 = tk.Frame(lf)
        f2.pack(anchor="w", padx=20)
        tk.Label(f2, text="Total images:").pack(side=tk.LEFT)
        self.target_entry = tk.Entry(f2, width=8)
        self.target_entry.insert(0, "500")
        self.target_entry.pack(side=tk.LEFT)
        self.target_entry.bind("<KeyRelease>", lambda e: self.update_count())

        btns = tk.Frame(self.root)
        btns.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(btns, text="Analyze Video", command=self.analyze_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="Run Extraction", command=self.run_extraction).pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(self.root, text="No video loaded", justify="left")
        self.info_label.pack(fill=tk.X, padx=10, pady=5)

        # 🔹 Preview playback
        self.preview_label = tk.Label(self.root, bg="#111")
        self.preview_label.pack(padx=10, pady=5)

        # 🔹 Progress + ETA
        self.progress = ttk.Progressbar(self.root, length=420, mode="determinate")
        self.progress.pack(padx=10, pady=4)

        self.eta_label = tk.Label(self.root, text="ETA: --:--")
        self.eta_label.pack(padx=10, pady=2)

    # ---------------- Browse ----------------
    def browse_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm")]
        )
        if p:
            self.video_path.set(p)
            self.analyze_video()

    def browse_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)

    # ---------------- Analyze ----------------
    def analyze_video(self):
        print("### ANALYZE VIDEO CALLED ###")

        cap = cv2.VideoCapture(self.video_path.get())
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video.")
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.update_count()

        self.info_label.config(
            text=f"Total frames: {self.total_frames}\nFPS: {fps:.2f}\nFrames to extract: {len(self.get_selected_frames())}"
        )

        self.show_preview_frame()

    # ---------------- Frame logic ----------------
    def get_selected_frames(self):
        if self.total_frames <= 0:
            return []

        try:
            if self.mode.get() == "gap":
                gap = max(1, int(self.frame_gap_entry.get()))
                return list(range(0, self.total_frames, gap))
            else:
                n = min(self.total_frames, int(self.target_entry.get()))
                return np.linspace(0, self.total_frames - 1, n, dtype=int).tolist()
        except ValueError:
            return []

    def update_count(self):
        frames = self.get_selected_frames()
        self.info_label.config(
            text=f"Total frames: {self.total_frames}\nFrames to extract: {len(frames)}"
        )

    # ---------------- Preview playback ----------------
    def show_preview_frame(self):
        frames = self.get_selected_frames()
        if not frames:
            return

        cap = cv2.VideoCapture(self.video_path.get())
        frame_no = int(np.random.choice(frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (380, 210))

        img = ImageTk.PhotoImage(Image.fromarray(frame), master=self.root)
        self.preview_image = img
        self.preview_label.config(image=img)

    # ---------------- Extraction ----------------
    def run_extraction(self):
        frames = self.get_selected_frames()
        if not frames:
            messagebox.showwarning("No frames", "Nothing to extract.")
            return

        out = self.output_dir.get()
        if not out:
            messagebox.showerror("Error", "Select output folder.")
            return

        os.makedirs(out, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path.get())
        total = len(frames)

        self.progress["maximum"] = total
        self.progress["value"] = 0

        start_time = time.time()

        for i, frame_no in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(out, f"frame_{frame_no:06d}.jpg"), frame)

            # 🔹 ETA
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (total - i - 1)
            mins, secs = divmod(int(remaining), 60)
            self.eta_label.config(text=f"ETA: {mins:02d}:{secs:02d}")

            # 🔹 Live playback
            if i % max(1, total // 30) == 0:
                self.show_preview_frame()

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        cap.release()
        self.eta_label.config(text="ETA: 00:00")

        messagebox.showinfo("Done", f"Extracted {total} frames.")


class StartupGUI:
    def __init__(self, master):
        self.root = master
        self.root.title("Auto Annotator + Tracking")


        tk.Label(self.root, text="JEZT Auto Annotation Tool", font=("Arial", 16, "bold")).pack(pady=10)

        self.img_dir = tk.StringVar()
        self.out_img_dir = tk.StringVar()
        self.out_label_dir = tk.StringVar()
        self.model_path = tk.StringVar()
        self.classes_str = tk.StringVar(value="person,car,dog")

        self.make_path_input("Input Images Folder", self.img_dir, self.browse_images)
        self.make_path_input("Output Images Folder", self.out_img_dir, self.browse_out_images)
        self.make_path_input("Output Labels Folder", self.out_label_dir, self.browse_out_labels)
        self.make_path_input("Model (.pt) File (YOLOv8/v11)", self.model_path, self.browse_model)

        tk.Label(self.root, text="Class names (comma separated):").pack(anchor="w", padx=10)
        tk.Entry(self.root, textvariable=self.classes_str, width=50).pack(padx=10, pady=5)

        self.use_sam3_var = tk.BooleanVar(value=False)
        self.sam3_config = None
        tk.Checkbutton(self.root, text="Annotate with Open Vocabulary (SAM3 & YOLOE)", variable=self.use_sam3_var, command=self.on_sam3_check, font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)

        tk.Button(self.root, text="Start Annotation", bg="#4CAF50", fg="white",
                  command=self.launch, font=("Arial", 12, "bold")).pack(pady=10)

        tk.Button(self.root, text="Open Video Splitter", bg="#2196F3", fg="white",
                  command=self.open_video_splitter, font=("Arial", 10)).pack(pady=5)

        self.root.mainloop()

    def on_sam3_check(self):
        if self.use_sam3_var.get():
            OpenVocabDialog(self)
        else:
            self.sam3_config = None

    def make_path_input(self, text, var, cmd):
        f = tk.Frame(self.root)
        f.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f, text=text, width=25, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(f, text="Browse", command=cmd).pack(side=tk.LEFT)

    def browse_images(self):
        path = filedialog.askdirectory(title="Select Input Image Folder")
        if path: self.img_dir.set(path)

    def browse_out_images(self):
        path = filedialog.askdirectory(title="Select Output Image Folder")
        if path: self.out_img_dir.set(path)

    def browse_out_labels(self):
        path = filedialog.askdirectory(title="Select Label Output Folder")
        if path: self.out_label_dir.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("Models", "*.pt *.pth *.engine"), ("All", "*.*")])
        if path: self.model_path.set(path)

    def open_video_splitter(self):
        VideoSplitterGUI(master=self.root)

    def launch(self):
        images = self.img_dir.get().strip()
        out_img = self.out_img_dir.get().strip()
        out_label = self.out_label_dir.get().strip()
        model_p = self.model_path.get().strip()
        classes = [c.strip() for c in self.classes_str.get().split(",") if c.strip()]

        if not images or not out_img or not out_label:
            messagebox.showerror("Error", "Please select all required folders.")
            return

        yolo_save_format = "bbox"
        model_wrapper = None
        if self.use_sam3_var.get() and self.sam3_config:
            yolo_save_format = self.sam3_config.get("save_format", "bbox")
        elif model_p:
            model_wrapper = ModelWrapper(model_p)
            if not model_wrapper.ready:
                messagebox.showwarning("Warning", "Failed to load model. Running without auto-annotation.")

        self.root.withdraw()
        AnnotatorApp(self.root, images, out_img, out_label, classes, model_wrapper, yolo_save_format=yolo_save_format, sam3_config=self.sam3_config)

# =========================== FLORENCE AND GLOBAL ADJUSTER ===========================
import random

try:
    import transformers
    import transformers.dynamic_module_utils
    transformers.dynamic_module_utils.check_imports = lambda filename: []
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class GlobalBoxAdjuster(tk.Toplevel):
    def __init__(self, annotator):
        super().__init__(annotator.root)
        self.annotator = annotator
        self.title("Interactive Global BBox Adjuster")
        self.geometry("350x150")
        
        self.annotator.global_resize_active = True
        self.annotator.global_resize_mult = 1.0
        self.annotator.save_state()
        
        tk.Label(self, text="Scroll your mouse wheel on the image behind\nthis window to interactively scale all boxes.\n\nClick Apply to use this scaling on all images.", pady=10).pack()
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Apply to All Images", command=self.apply, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.cancel, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=10)
        
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        
    def cancel(self):
        self.annotator.global_resize_active = False
        self.annotator.global_resize_mult = 1.0
        self.annotator.undo_action()
        self.destroy()
        
    def apply(self):
        self.annotator.global_resize_active = False
        mult = self.annotator.global_resize_mult
        
        if abs(mult - 1.0) < 0.001:
            messagebox.showinfo("Done", "No scaling applied.")
            self.destroy()
            return
            
        count = 0
        for p in self.annotator.image_paths:
            if p == self.annotator.image_paths[self.annotator.index]:
                self.annotator.save_labels()
                continue
                
            base = os.path.splitext(os.path.basename(p))[0]
            lpath = os.path.join(self.annotator.label_out, base + ".txt")
            if os.path.exists(lpath):
                img = Image.open(p)
                w, h = img.size
                boxes = load_yolo_labels(lpath, w, h)
                changed = False
                for b in boxes:
                    if "poly" in b and b["poly"]:
                        continue # ignore polys
                    cx = (b["x1"] + b["x2"]) / 2.0
                    cy = (b["y1"] + b["y2"]) / 2.0
                    bw = (b["x2"] - b["x1"]) * mult
                    bh = (b["y2"] - b["y1"]) * mult
                    b["x1"] = cx - bw / 2.0
                    b["x2"] = cx + bw / 2.0
                    b["y1"] = cy - bh / 2.0
                    b["y2"] = cy + bh / 2.0
                    clamp_box(b, w, h)
                    changed = True
                if changed:
                    save_yolo_labels(lpath, boxes, w, h)
                    count += 1
                
        self.annotator.load_image(self.annotator.index) # reload
        messagebox.showinfo("Done", f"Adjusted boxes in {count + 1} labeled images.")
        self.destroy()

class FlorenceTestViewer(tk.Toplevel):
    def __init__(self, master, results, classes=None):
        super().__init__(master)
        self.classes = classes or []
        self.title("Florence-2 Test Results")
        self.geometry("1000x800")
        
        self.results = results
        self.current_idx = 0
        
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(fill=tk.X, pady=5)
        self.label = tk.Label(self.top_frame, text="", font=("Arial", 12))
        self.label.pack()
        
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.bind("n", self.next_img)
        self.bind("p", self.prev_img)
        self.bind("<Right>", self.next_img)
        self.bind("<Left>", self.prev_img)
        
        self.tk_img = None
        
        self.resize_frame = tk.Frame(self)
        self.resize_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.resize_frame, text="Global BBox Adjuster:").pack(side=tk.LEFT, padx=10)
        
        init_mult = getattr(master, 'global_resize_mult', 1.0)
        init_shift_x = getattr(master, 'global_shift_x', 0.0)
        init_shift_y = getattr(master, 'global_shift_y', 0.0)
            
        self.resize_mult_var = tk.DoubleVar(value=init_mult)
        self.shift_x_var = tk.DoubleVar(value=init_shift_x)
        self.shift_y_var = tk.DoubleVar(value=init_shift_y)
        
        def on_val_change(val):
            if hasattr(self.master, 'global_resize_mult'):
                self.master.global_resize_mult = self.resize_mult_var.get()
                self.master.global_shift_x = self.shift_x_var.get()
                self.master.global_shift_y = self.shift_y_var.get()
            self.load_current()
            
        self.resize_scale = tk.Scale(self.resize_frame, label="Scale", variable=self.resize_mult_var, from_=0.5, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, command=on_val_change)
        self.resize_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.shift_x_scale = tk.Scale(self.resize_frame, label="Shift X", variable=self.shift_x_var, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, command=on_val_change)
        self.shift_x_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.shift_y_scale = tk.Scale(self.resize_frame, label="Shift Y", variable=self.shift_y_var, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, command=on_val_change)
        self.shift_y_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.load_current()
        
    def load_current(self):
        if not self.results: return
        p, boxes = self.results[self.current_idx]
        self.label.config(text=f"Image {self.current_idx + 1} of {len(self.results)} - {os.path.basename(p)}")
        
        img = Image.open(p).convert("RGB")
        
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        palette = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255), (0, 255, 127), (255, 191, 0), (0, 215, 255), (211, 0, 148)]
        for b in boxes:
            c_idx = b.get('class', 0)
            color = palette[c_idx % len(palette)]
            
            x1_o, y1_o, x2_o, y2_o = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
            mult = getattr(self, 'resize_mult_var', None)
            mult = mult.get() if mult else 1.0
            
            shift_x = getattr(self, 'shift_x_var', None)
            shift_x = shift_x.get() if shift_x else 0.0
            
            shift_y = getattr(self, 'shift_y_var', None)
            shift_y = shift_y.get() if shift_y else 0.0
            
            cx, cy = (x1_o + x2_o) / 2.0, (y1_o + y2_o) / 2.0
            bw, bh = (x2_o - x1_o) * mult, (y2_o - y1_o) * mult
            cx += shift_x
            cy += shift_y
            
            x1, y1, x2, y2 = int(cx - bw/2.0), int(cy - bh/2.0), int(cx + bw/2.0), int(cy + bh/2.0)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
            
            name = self.classes[c_idx] if c_idx < len(self.classes) else str(c_idx)
            cv2.putText(cv_img, name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if 'poly' in b and b['poly']:
                pts = np.array(b['poly'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(cv_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        w, h = Image.fromarray(cv_img).size
        scale = 800 / w if w > 800 else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h))
        
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv_img))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        
    def next_img(self, e=None):
        if self.current_idx < len(self.results) - 1:
            self.current_idx += 1
            self.load_current()
            
    def prev_img(self, e=None):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current()

class FlorenceDialog(tk.Toplevel):
    def __init__(self, annotator):
        super().__init__(annotator.root)
        self.annotator = annotator
        self.title("Annotate with Florence-2")
        self.geometry("650x550")
        
        if not TRANSFORMERS_AVAILABLE:
            messagebox.showerror("Missing Dependency", "Please install transformers, einops, and timm.")
            self.destroy()
            return
            
        self.cache_file = ".florence_prompts_cache.json"
        self.history = self.load_history()
        
        self.processor = None
        self.model = None
        self.current_model_id = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.make_widgets()
        
    def load_history(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []
        
    def save_history(self, prompt):
        if prompt not in self.history:
            self.history.append(prompt)
            with open(self.cache_file, "w") as f:
                json.dump(self.history, f)
                
    def make_widgets(self):
        f1 = tk.Frame(self)
        f1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f1, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="Select Model...")
        self.model_opts = [
            "microsoft/Florence-2-base-ft", 
            "microsoft/Florence-2-large-ft",
            "microsoft/Florence-2-base",
            "microsoft/Florence-2-large"
        ]
        self.model_menu = tk.OptionMenu(f1, self.model_var, *self.model_opts, command=self.on_model_select)
        self.model_menu.pack(side=tk.LEFT, padx=5)
        
        self.eta_label = tk.Label(f1, text="ETA: --", fg="blue")
        self.eta_label.pack(side=tk.LEFT, padx=10)
        
        f2 = tk.Frame(self)
        f2.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f2, text="Text Prompt:").pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar()
        tk.Entry(f2, textvariable=self.prompt_var, width=35).pack(side=tk.LEFT, padx=5)
        
        tk.Label(f2, text="Class ID:").pack(side=tk.LEFT)
        self.target_class_var = tk.StringVar(value="0")
        tk.Entry(f2, textvariable=self.target_class_var, width=5).pack(side=tk.LEFT, padx=5)
        
        f3 = tk.Frame(self)
        f3.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f3, text="Previous Prompts:").pack(side=tk.LEFT)
        self.hist_var = tk.StringVar(value="Select...")
        hist_opts = self.history if self.history else ["(no history)"]
        self.hist_menu = tk.OptionMenu(f3, self.hist_var, *hist_opts, command=self.on_hist_select)
        self.hist_menu.pack(side=tk.LEFT, padx=5)
        
        f4 = tk.LabelFrame(self, text="Test Prompt")
        f4.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(f4, text="Random Frames:").pack(side=tk.LEFT, padx=5)
        self.test_n_var = tk.StringVar(value="3")
        tk.Entry(f4, textvariable=self.test_n_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Button(f4, text="Run Test", command=self.run_test).pack(side=tk.LEFT, padx=10)
        
        tk.Button(self, text="Open Global BBox Adjuster", command=lambda: GlobalBoxAdjuster(self.annotator)).pack(pady=5)
        
        self.progress = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(padx=10, pady=10)
        
        self.lbl_progress = tk.Label(self, text="Ready")
        self.lbl_progress.pack(pady=2)
        
        tk.Button(self, text="Annotate Full Dataset", command=self.annotate_dataset, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)
        
    def on_hist_select(self, val):
        if val != "(no history)":
            self.prompt_var.set(val)
            
    def on_model_select(self, val):
        self.eta_label.config(text="Loading model & benchmarking...")
        self.update_idletasks()
        threading.Thread(target=self._load_and_benchmark, args=(val,), daemon=True).start()
        
    def _load_and_benchmark(self, model_id):
        if self.current_model_id != model_id:
            try:
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(self.device)
                self.current_model_id = model_id
            except Exception as e:
                self.eta_label.config(text="Failed to load model")
                print("Error loading florence:", e)
                return
                
        if self.annotator.image_paths:
            img = Image.open(self.annotator.image_paths[0]).convert("RGB")
            start = time.time()
            self._run_inference(img, "test")
            elapsed = time.time() - start
            
            total_images = len(self.annotator.image_paths)
            est_total = elapsed * total_images
            mins, secs = divmod(int(est_total), 60)
            self.eta_label.config(text=f"ETA for {total_images} imgs: {mins}m {secs}s (approx {elapsed:.2f}s/img)")
            
    def _run_inference(self, pil_image, prompt_text, task="<CAPTION_TO_PHRASE_GROUNDING>"):
        if not self.model or not self.processor: return []
        prompt = task + prompt_text
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task, image_size=pil_image.size)
        
        boxes = []
        if task in parsed_answer:
            result = parsed_answer[task]
            if isinstance(result, dict):
                if 'bboxes' in result:
                    for bbox in result.get('bboxes', []):
                        x1, y1, x2, y2 = bbox
                        boxes.append({'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)})
                elif 'polygons' in result:
                    for polys, label in zip(result.get('polygons', []), result.get('labels', [])):
                        for poly in polys:
                            xs = poly[0::2]
                            ys = poly[1::2]
                            if not xs or not ys: continue
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)
                            boxes.append({'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)})
        return boxes
        
    def run_test(self):
        prompt = self.prompt_var.get().strip()
        if not prompt: return
        if not self.model: 
            messagebox.showerror("Error", "Select a model first")
            return
        
        try: n = int(self.test_n_var.get())
        except: return
        
        self.save_history(prompt)
        
        paths = random.sample(self.annotator.image_paths, min(n, len(self.annotator.image_paths)))
        
        prog_win = tk.Toplevel(self)
        prog_win.title("Testing...")
        prog_win.geometry("300x100")
        lbl = tk.Label(prog_win, text=f"Running inference on {len(paths)} images...", pady=20)
        lbl.pack()
        prog_win.update()
        
        results = []
        for i, p in enumerate(paths):
            lbl.config(text=f"Running inference {i+1}/{len(paths)}...")
            prog_win.update()
            img = Image.open(p).convert("RGB")
            boxes = self._run_inference(img, prompt)
            results.append((p, boxes))
            
        prog_win.destroy()
        viewer = FlorenceTestViewer(self, results, self.annotator.classes)
            
    def annotate_dataset(self):
        prompt = self.prompt_var.get().strip()
        if not prompt: return
        if not self.model: 
            messagebox.showerror("Error", "Select a model first")
            return
        
        self.save_history(prompt)
        target_cls = int(self.target_class_var.get()) if self.target_class_var.get().isdigit() else 0
        
        threading.Thread(target=self._annotate_task, args=(prompt, target_cls), daemon=True).start()
        
    def _annotate_task(self, prompt, target_cls):
        total = len(self.annotator.image_paths)
        self.progress["maximum"] = total
        self.progress["value"] = 0
        
        start_time = time.time()
        
        for i, p in enumerate(self.annotator.image_paths):
            img = Image.open(p).convert("RGB")
            w, h = img.size
            new_boxes = self._run_inference(img, prompt)
            
            base = os.path.splitext(os.path.basename(p))[0]
            lpath = os.path.join(self.annotator.label_out, base + ".txt")
            
            boxes = load_yolo_labels(lpath, w, h)
            for nb in new_boxes:
                nb['class'] = target_cls
                boxes.append(nb)
                
            save_yolo_labels(lpath, boxes, w, h)
            shutil.copy2(p, os.path.join(self.annotator.img_out, os.path.basename(p)))
            
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            rem = avg * (total - i - 1)
            mins, secs = divmod(int(rem), 60)
            
            self.progress["value"] = i + 1
            self.lbl_progress.config(text=f"Processed {i+1}/{total} - ETA: {mins}m {secs}s")
            
        self.lbl_progress.config(text="Done!")
        self.annotator.load_image(self.annotator.index)

class OpenVocabDialog(tk.Toplevel):
    def __init__(self, startup_gui):
        super().__init__(startup_gui.root)
        self.startup_gui = startup_gui
        self.title("Open Vocabulary Configuration")
        self.geometry("500x350")
        
        self.transient(startup_gui.root)
        self.grab_set()
        
        f0 = tk.Frame(self)
        f0.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f0, text="Model Architecture:").pack(side=tk.LEFT)
        self.model_type_var = tk.StringVar(value="YOLOE")
        def on_model_type_change():
            if self.model_type_var.get() == "YOLOE":
                self.quant_var.set("float32")
            else:
                self.quant_var.set("float16")
                
        tk.Radiobutton(f0, text="YOLOE", variable=self.model_type_var, value="YOLOE", command=on_model_type_change).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(f0, text="SAM3", variable=self.model_type_var, value="SAM3", command=on_model_type_change).pack(side=tk.LEFT, padx=5)

        f1 = tk.Frame(self)
        f1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f1, text="Model Path:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=r"C:\Users\JEZT\Downloads\Quick Share\sam3.pt")
        tk.Entry(f1, textvariable=self.model_var, width=30).pack(side=tk.LEFT, padx=5)
        tk.Button(f1, text="Browse", command=self.browse_model).pack(side=tk.LEFT)
        
        f2 = tk.Frame(self)
        f2.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f2, text="YOLO Save Format:").pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value="bbox")
        tk.Radiobutton(f2, text="Bounding Box (cx,cy,w,h)", variable=self.format_var, value="bbox").pack(side=tk.LEFT)
        tk.Radiobutton(f2, text="Polygon (x1,y1,x2,y2...)", variable=self.format_var, value="polygon").pack(side=tk.LEFT)
        
        f3 = tk.Frame(self)
        f3.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f3, text="Quantization:").pack(side=tk.LEFT)
        self.quant_var = tk.StringVar(value="float32" if self.model_type_var.get() == "YOLOE" else "float16")
        tk.OptionMenu(f3, self.quant_var, "bfloat16", "float16",  "int8", "float32").pack(side=tk.LEFT, padx=5)
        
        f3_5 = tk.Frame(self)
        f3_5.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f3_5, text="Target Classes:").pack(side=tk.LEFT)
        self.target_classes_var = tk.StringVar(value=startup_gui.classes_str.get())
        tk.Entry(f3_5, textvariable=self.target_classes_var, width=30, state="readonly").pack(side=tk.LEFT, padx=5)
        tk.Button(f3_5, text="Select Classes", command=self.open_class_selection).pack(side=tk.LEFT)
        

        f3_6 = tk.Frame(self)
        f3_6.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f3_6, text="Chunk Size:").pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value="5")
        tk.Spinbox(f3_6, from_=1, to=20, textvariable=self.chunk_size_var, width=5).pack(side=tk.LEFT, padx=5)
        self.global_resize_mult = 1.0
        self.global_shift_x = 0.0
        self.global_shift_y = 0.0
        f4 = tk.Frame(self)
        f4.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f4, text="Random Frames:").pack(side=tk.LEFT, padx=5)
        self.test_n_var = tk.StringVar(value="3")
        tk.Entry(f4, textvariable=self.test_n_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Button(f4, text="Test Inference", command=self.test_inference).pack(side=tk.LEFT)
        
        self.eta_label = tk.Label(self, text="ETA: --", fg="blue")
        self.eta_label.pack(pady=10)
        
        tk.Button(self, text="Save & Close", command=self.save_close, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=10)
        
    def open_class_selection(self):
        main_classes_str = self.startup_gui.classes_str.get()
        main_classes = [c.strip() for c in main_classes_str.split(",") if c.strip()]
        if not main_classes:
            messagebox.showwarning("Error", "No classes defined in startup GUI.")
            return
            
        try:
            import json
            current_mapping = json.loads(self.target_classes_var.get())
        except Exception:
            current_targets = [c.strip() for c in self.target_classes_var.get().split(",") if c.strip()]
            current_mapping = {c: [] for c in current_targets}
            
        dialog = tk.Toplevel(self)
        dialog.title("Select Target Classes & Synonyms")
        dialog.geometry("450x500")
        dialog.transient(self)
        dialog.grab_set()
        
        tk.Label(dialog, text="Check classes and add synonyms (comma-separated):", anchor="w").pack(fill=tk.X, padx=8, pady=6)
        
        container = tk.Frame(dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        var_map = {}
        syn_map = {}
        for name in main_classes:
            f = tk.Frame(scroll_frame)
            f.pack(fill=tk.X, anchor="w", pady=2)
            
            v = tk.IntVar(value=1 if name in current_mapping else 0)
            chk = tk.Checkbutton(f, text=name, variable=v, anchor="w", width=15)
            chk.pack(side=tk.LEFT)
            
            tk.Label(f, text="Synonyms:").pack(side=tk.LEFT)
            syn_v = tk.StringVar(value=",".join(current_mapping.get(name, [])))
            tk.Entry(f, textvariable=syn_v, width=20).pack(side=tk.LEFT, padx=5)
            
            var_map[name] = v
            syn_map[name] = syn_v
            
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=6)
        
        def on_ok():
            import json
            mapping = {}
            for nm, vv in var_map.items():
                if vv.get():
                    syns = [s.strip() for s in syn_map[nm].get().split(',') if s.strip()]
                    mapping[nm] = syns
            self.target_classes_var.set(json.dumps(mapping))
            dialog.destroy()
            
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=8)
        
    def browse_model(self):
        path = filedialog.askopenfilename(parent=self, title="Select SAM3 Weights", filetypes=[("Models", "*.pt *.pth *.engine"), ("All", "*.*")])
        if path:
            self.model_var.set(path)
            
    def save_close(self):
        self.startup_gui.sam3_config = {
            "model_type": self.model_type_var.get(),
            "model_path": self.model_var.get(),
            "save_format": self.format_var.get(),
            "quantization": self.quant_var.get(),
            "target_classes": self.target_classes_var.get(),
            "chunk_size": self.chunk_size_var.get(),
            "global_resize_mult": getattr(self, 'global_resize_mult', 1.0),
            "global_shift_x": getattr(self, 'global_shift_x', 0.0),
            "global_shift_y": getattr(self, 'global_shift_y', 0.0)
        }
        self.destroy()
        
    def test_inference(self):
        img_dir = self.startup_gui.img_dir.get().strip()
        if not img_dir or not os.path.exists(img_dir):
            messagebox.showerror("Error", "Select input image folder first.")
            return
            
        images = []
        for e in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
            images.extend(glob.glob(os.path.join(img_dir, e)))
            
        if not images:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return
            
        try: n = int(self.test_n_var.get())
        except: n = 1
        
        img_paths = random.sample(images, min(n, len(images)))
        
        classes_str = self.startup_gui.classes_str.get()
        classes = [c.strip() for c in classes_str.split(",") if c.strip()]
        
        if not classes:
            messagebox.showerror("Error", "Provide class names.")
            return
            
        self.eta_label.config(text=f"Running test inference on {len(img_paths)} frames...")
        self.update_idletasks()
        
        def run():
            try:
                import torch
                model_path = self.model_var.get().strip()
                if not model_path.endswith('.pt') and not model_path.endswith('.pth') and not model_path.endswith('.engine'):
                    model_path += '.pt'
                quant = self.quant_var.get()
                model_type = self.model_type_var.get()
                if model_type == "YOLOE":
                    from ultralytics import YOLO
                    predictor = YOLO(model_path)
                else:
                    from ultralytics.models.sam import SAM3SemanticPredictor
                    overrides = {
                        "model": model_path,
                        "task": "segment",
                        "mode": "predict",
                        "conf": 0.5,
                        "imgsz": 644,
                        "half": (quant == "float16")
                    }
                    predictor = SAM3SemanticPredictor(overrides=overrides)
                
                total_elapsed = 0
                valid_elapsed_count = 0
                results = []
                
                target_classes_str = self.target_classes_var.get()
                try:
                    import json
                    mapping_dict = json.loads(target_classes_str)
                except Exception:
                    mapping_dict = {c.strip(): [] for c in target_classes_str.split(",") if c.strip()}
                    
                chunk_sz = int(self.chunk_size_var.get())
                chunks = build_chunks(mapping_dict, chunk_sz)
                
                for i, img_path in enumerate(img_paths):
                    img = Image.open(img_path).convert("RGB")
                    boxes = []
                    frame_elapsed = 0
                    
                    for chunk in chunks:
                        prompts = [p for p, b in chunk]
                        with torch.no_grad():
                            if model_type == "YOLOE":
                                predictor.set_classes(prompts)
                                res = predictor(img, verbose=False)
                            else:
                                res = predictor(img, text=prompts)
                            
                        if res and hasattr(res[0], 'speed') and isinstance(res[0].speed, dict):
                            frame_elapsed += sum(res[0].speed.values()) / 1000.0
                            
                        for r in res:
                            has_masks = r.masks is not None
                            if r.boxes:
                                bxyxy = r.boxes.xyxy.cpu().numpy()
                                bcls = r.boxes.cls.cpu().numpy()
                                segments = r.masks.xy if has_masks else []
                                for idx in range(len(bxyxy)):
                                    local_id = int(bcls[idx])
                                    if local_id < len(chunk):
                                        prompt, base_cls = chunk[local_id]
                                        c = classes.index(base_cls) if base_cls in classes else -1
                                        if c != -1:
                                            x1, y1, x2, y2 = bxyxy[idx]
                                            box_data = {'class': c, 'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                                            if has_masks and idx < len(segments):
                                                poly = segments[idx].tolist()
                                                if poly:
                                                    box_data['poly'] = poly
                                            boxes.append(box_data)
                                            
                    if len(img_paths) > 1 and i == 0:
                        pass
                    else:
                        total_elapsed += frame_elapsed
                        valid_elapsed_count += 1
                        
                    results.append((img_path, boxes))
                
                avg_time = total_elapsed / max(1, valid_elapsed_count)
                total_imgs = len(images)
                est_total = avg_time * total_imgs
                mins, secs = divmod(int(est_total), 60)
                
                self.eta_label.config(text=f"Test took {total_elapsed:.2f}s (ignoring 1st frame warmup). Est. total for {total_imgs} images: {mins}m {secs}s")
                
                # Cleanup VRAM
                del predictor
                if 'torch' in sys.modules:
                    sys.modules['torch'].cuda.empty_cache()
                    
                self.after(0, lambda: FlorenceTestViewer(self, results, classes))
            except Exception as e:
                self.eta_label.config(text=f"Error: {e}")
                
        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    StartupGUI(root)
    root.mainloop()




