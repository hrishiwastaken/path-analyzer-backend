import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import PanedWindow
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 1: THE BACKSTAGE CREW (Core Image Processing Functions)
# These functions are the unsung heroes. They don't wear capes, but they
# wrestle pixels into submission before the GUI even wakes up.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def skeletonize_image(image):
    """
    Puts an image on a strict diet until it's just a 1-pixel-wide skeleton.
    It's the digital equivalent of "I have a beach body to get ready for!"
    Reference: Zhang-Suen Thinning Algorithm (conceptually similar).
    """
    skeleton = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return skeleton

def preprocess_and_extract_path(image_path, lower_bound, upper_bound):
    """
    The CSI "Enhance!" function. Takes a crime scene (the image), finds the
    evidence (the colored path), and dusts for prints (extracts the coordinates).
    """
    img = cv2.imread(image_path)
    if img is None: return None, None, None, 1.0
    
    # If the image is a behemoth, we shrink it to avoid making the computer cry.
    original_shape = img.shape
    scale_factor = 1.0
    if original_shape[1] > 1000:
        scale_percent = 50
        scale_factor = scale_percent / 100.0
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # This 'morphologyEx' is like cleaning up a messy room. It fills in gaps
    # and removes little specks of noise. Very satisfying.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=7)
    
    skeleton = skeletonize_image(mask)
    rows, cols = np.where(skeleton > 0)
    
    if len(rows) < 2: return cv2.imread(image_path), None, skeleton, scale_factor
    
    all_points_scaled = np.column_stack((cols, rows))
    return cv2.imread(image_path), all_points_scaled, skeleton, scale_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: THE SUPPORTING CAST (GUI Helper Windows)
# These are the specialized pop-up windows. Each has one job and does it with flair.
# They are defined here so our Main Application knows who to call for help.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MaskViewer(tk.Toplevel):
    """
    A simple "Look at this!" window.
    For when you absolutely, positively have to see the raw, unfiltered
    black-and-white mask we created. No frills, just pixels.
    """
    def __init__(self, parent, mask_image):
        super().__init__(parent)
        self.title("Full Detected Mask")
        self.geometry("600x600")
        self.mask_image = mask_image
        self.mask_label = tk.Label(self)
        self.mask_label.pack(fill=tk.BOTH, expand=True)
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        container_w = self.winfo_width()
        container_h = self.winfo_height()
        if container_w < 2 or container_h < 2: return
        img_h, img_w = self.mask_image.shape
        scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(self.mask_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask)
            self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)

class ColorTunerWindow(tk.Toplevel):
    """
    The "Goldilocks" window. Lets the user fiddle with sliders until the
    color detection is juuuust right. Not too much, not too little.
    """
    def __init__(self, parent, image_path, current_lower, current_upper, on_apply_callback):
        super().__init__(parent)
        self.title("Color Tuner")
        self.geometry("800x700")
        self.on_apply_callback = on_apply_callback
        img = cv2.imread(image_path)
        img_for_hsv = img.copy()
        if img.shape[1] > 1000:
            scale_percent = 50
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img_for_hsv = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        self.hsv = cv2.cvtColor(img_for_hsv, cv2.COLOR_BGR2HSV)
        controls_container = tk.Frame(self)
        controls_container.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
        self.mask_label = tk.Label(self, bg="black")
        self.mask_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sliders_frame = tk.Frame(controls_container)
        sliders_frame.pack(fill=tk.X)
        self.h_min = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Min (0-179)", command=self.update_mask); self.h_min.set(current_lower[0]); self.h_min.pack(fill=tk.X)
        self.s_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Min (0-255)", command=self.update_mask); self.s_min.set(current_lower[1]); self.s_min.pack(fill=tk.X)
        self.v_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Min (0-255)", command=self.update_mask); self.v_min.set(current_lower[2]); self.v_min.pack(fill=tk.X)
        self.h_max = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Max (0-179)", command=self.update_mask); self.h_max.set(current_upper[0]); self.h_max.pack(fill=tk.X)
        self.s_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Max (0-255)", command=self.update_mask); self.s_max.set(current_upper[1]); self.s_max.pack(fill=tk.X)
        self.v_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Max (0-255)", command=self.update_mask); self.v_max.set(current_upper[2]); self.v_max.pack(fill=tk.X)
        btn_frame = tk.Frame(controls_container, pady=5); btn_frame.pack()
        tk.Button(btn_frame, text="Apply", command=self.apply_and_close).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        self.after(100, self.update_mask)
    def update_mask(self, val=None):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        mask = cv2.inRange(self.hsv, lower, upper)
        container_w = self.mask_label.winfo_width(); container_h = self.mask_label.winfo_height()
        if container_w < 2 or container_h < 2: self.after(100, self.update_mask); return
        img_h, img_w = mask.shape; scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)
    def apply_and_close(self):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        self.on_apply_callback(lower, upper)
        self.destroy()

class InteractiveSlopeWindow(tk.Toplevel):
    """
    Behold, the Slope-o-Meter 3000!
    This window is for the true connoisseurs of curves. It lets you slide
    a point along the path and see the exact slope and angle at that spot,
    complete with a snazzy tangent line. Your high school calculus teacher
    would be so proud.
    """
    def __init__(self, parent, path_points, tck, u_params):
        super().__init__(parent)
        self.title("Interactive Slope Analysis")
        self.geometry("950x700")
        self.path_points = path_points
        self.tck = tck
        self.u_params = u_params
        
        # Is this path a "long boi" or a "tall boi"? We rotate it for a better view.
        x_range = np.ptp(self.path_points[:, 0])
        y_range = np.ptp(self.path_points[:, 1])
        self.is_rotated_for_display = y_range > x_range

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_frame = tk.Frame(main_frame, padx=10, pady=10, relief=tk.SUNKEN, borderwidth=2)
        data_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        tk.Label(data_frame, text="Slope Data", font=("Helvetica", 14, "bold")).pack(pady=(0, 15))
        self.slope_var = tk.StringVar(value="Slope (dy/dx): -")
        self.int_slope_var = tk.StringVar(value="Integer Slope: -")
        self.angle_var = tk.StringVar(value="Angle: -")
        tk.Label(data_frame, textvariable=self.slope_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)
        tk.Label(data_frame, textvariable=self.int_slope_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)
        tk.Label(data_frame, textvariable=self.angle_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)

        self.slider = tk.Scale(plot_frame, from_=0, to=len(self.path_points) - 1,
                               orient=tk.HORIZONTAL, command=self._update_plot,
                               label="Position Along Path")
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self._draw_initial_plot()
        self.after(100, lambda: self.slider.set(len(self.path_points) // 2))

    def _draw_initial_plot(self):
        self.ax.clear()
        if self.is_rotated_for_display:
            self.ax.plot(self.path_points[:, 1], self.path_points[:, 0], color='red', linewidth=2)
            self.ax.set_xlabel("Secondary Axis (pixels)")
            self.ax.set_ylabel("Primary Axis (pixels)")
        else:
            self.ax.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2)
            self.ax.set_xlabel("Primary Axis (pixels)")
            self.ax.set_ylabel("Secondary Axis (pixels)")

        self.point_on_curve, = self.ax.plot([], [], 'o', color='magenta', markersize=12, markeredgecolor='black', zorder=10)
        self.tangent_line, = self.ax.plot([], [], color='blue', linestyle='--', linewidth=2, zorder=9)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range_plot = xlim[1] - xlim[0]
        y_range_plot = ylim[1] - ylim[0]
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        max_range = max(x_range_plot, y_range_plot)
        plot_dimension = max_range * 1.1 
        self.ax.set_xlim(x_center - plot_dimension / 2, x_center + plot_dimension / 2)
        self.ax.set_ylim(y_center - plot_dimension / 2, y_center + plot_dimension / 2)
        
        self.ax.set_title("Slope along the Fitted Path")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis()
        self.fig.tight_layout()
        self.canvas.draw()

    def _update_plot(self, val):
        index = int(val)
        current_point = self.path_points[index]
        u_val = self.u_params[index]
        dx_dt, dy_dt = splev(u_val, self.tck, der=1)
        angle_deg = np.degrees(np.arctan2(dy_dt, dx_dt))

        if abs(dx_dt) < 1e-6:
            self.slope_var.set("Slope (dy/dx): Infinity")
            self.int_slope_var.set("Integer Slope: N/A")
        else:
            slope_val = dy_dt / dx_dt
            self.slope_var.set(f"Slope (dy/dx): {slope_val:.2f}")
            self.int_slope_var.set(f"Integer Slope: {int(round(slope_val))}")
        
        self.angle_var.set(f"Angle: {angle_deg:.1f}°")

        # Time to draw that tangent line! It's like having a tiny Isaac Newton
        # living in your computer. "A body in motion stays in motion... along this blue line."
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        if magnitude > 1e-6:
            ux, uy = dx_dt / magnitude, dy_dt / magnitude
            plot_width = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            line_length = plot_width * 0.15
            cx, cy = current_point[0], current_point[1]
            x_coords = [cx - ux * line_length / 2, cx + ux * line_length / 2]
            y_coords = [cy - uy * line_length / 2, cy + uy * line_length / 2]
            
            if self.is_rotated_for_display:
                self.tangent_line.set_data(y_coords, x_coords)
            else:
                self.tangent_line.set_data(x_coords, y_coords)
        else:
            self.tangent_line.set_data([], [])

        if self.is_rotated_for_display:
            self.point_on_curve.set_data([current_point[1]], [current_point[0]])
        else:
            self.point_on_curve.set_data([current_point[0]], [current_point[1]])

        self.canvas.draw_idle()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: THE MAIN CHARACTER (The PathAnalyzerApp Class)
# This is Mission Control. The big cheese. The class that orchestrates the
# entire operation, from loading the image to launching the analysis windows.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PathAnalyzerApp:
    """Our main application window. It's like a digital LEGO set, with panes,
    buttons, and plots all snapped together to build something cool."""
    def __init__(self, root):
        self.root = root
        self.root.title("Path Analyzer")
        self.root.geometry("1200x800")
        
        # This is our state management, the app's "brain".
        self.image_path = None; self.original_cv_img = None; self.final_mask = None
        self.all_ordered_points = None; self.displayed_nodes = None
        self.smooth_path_points = None; self.start_node = None
        self.hsv_lower = np.array([0, 70, 50]); self.hsv_upper = np.array([179, 255, 255])
        self.tck = None; self.u_params = None
        
        # --- GUI Layout Construction ---
        self.main_paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)
        self.left_pane = tk.Frame(self.main_paned_window)
        self.main_paned_window.add(self.left_pane, weight=3)
        self.right_paned_window = PanedWindow(self.main_paned_window, orient=tk.VERTICAL)
        self.main_paned_window.add(self.right_paned_window, weight=1)
        
        # Left side: The big, beautiful image display.
        self.fig = Figure(); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Right side: A control panel on top, math plot on the bottom.
        controls_frame = tk.Frame(self.right_paned_window, height=280)
        self.right_paned_window.add(controls_frame, weight=0)
        math_plot_frame = tk.Frame(self.right_paned_window)
        self.right_paned_window.add(math_plot_frame, weight=1)
        controls_frame.pack_propagate(False)
        
        # The buttons and widgets that make the magic happen.
        self.btn_load = tk.Button(controls_frame, text="Load Image", command=self.load_image); self.btn_load.pack(fill=tk.X, padx=10, pady=5)
        self.btn_tuner = tk.Button(controls_frame, text="Tune Color Range...", command=self.open_color_tuner, state=tk.DISABLED); self.btn_tuner.pack(fill=tk.X, padx=10)
        self.btn_analyze = tk.Button(controls_frame, text="Analyze Path", command=self.analyze_path, state=tk.DISABLED); self.btn_analyze.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(controls_frame, text="Number of Nodes to Display:").pack(padx=10)
        self.nodes_var = tk.IntVar(value=75)
        self.nodes_slider = tk.Scale(controls_frame, from_=2, to=1000, orient=tk.HORIZONTAL, variable=self.nodes_var, command=self.update_node_display); self.nodes_slider.pack(fill=tk.X, padx=10)
        self.btn_show_mask = tk.Button(controls_frame, text="Show Full Mask", command=self.show_full_mask, state=tk.DISABLED); self.btn_show_mask.pack(fill=tk.X, padx=10, pady=5)
        self.btn_slope = tk.Button(controls_frame, text="Interactive Slope Analysis", command=self.open_slope_visualizer, state=tk.DISABLED); self.btn_slope.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(controls_frame, text="Please load an image to begin."); self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # The math plot, for seeing the path without the distracting image.
        tk.Label(math_plot_frame, text="Mathematical 2D Plot").pack()
        self.math_fig = Figure(); self.math_ax = self.math_fig.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_fig, master=math_plot_frame)
        self.math_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

    def load_image(self):
        """Wipes the slate clean and loads a new image to investigate."""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path: return
        self.image_path = path
        self.original_cv_img = cv2.imread(self.image_path)
        if self.original_cv_img is None: messagebox.showerror("Error", "Failed to load image."); return
        
        # Reset all the data from the previous run.
        self.all_ordered_points = None; self.displayed_nodes = None
        self.smooth_path_points = None; self.start_node = None
        self.tck = None; self.u_params = None
        
        self.display_image(self.original_cv_img, "Original Image Loaded")
        self.update_math_plot()
        
        # Guide the user to the next logical step.
        self.btn_tuner.config(state=tk.NORMAL)
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_show_mask.config(state=tk.DISABLED)
        self.btn_slope.config(state=tk.DISABLED)
        self.status_label.config(text="Image loaded. Please tune color range.")

    def on_tuner_apply(self, new_lower, new_upper):
        """Callback for when the user is done playing with the color sliders."""
        self.hsv_lower = new_lower
        self.hsv_upper = new_upper
        self.status_label.config(text="New color range applied. Ready to analyze.")
        self.btn_analyze.config(state=tk.NORMAL)

    def open_color_tuner(self):
        if self.image_path: ColorTunerWindow(self.root, self.image_path, self.hsv_lower, self.hsv_upper, self.on_tuner_apply)

    def analyze_path(self):
        """This is where the real number-crunching begins. It's a multi-stage process..."""
        if self.image_path is None: return
        
        # Stage 1: Extraction. Call our backstage crew to get the raw points.
        self.status_label.config(text="Extracting path points..."); self.root.update_idletasks()
        self.original_cv_img, raw_points, self.final_mask, scale_factor = preprocess_and_extract_path(self.image_path, self.hsv_lower, self.hsv_upper)
        if raw_points is None or len(raw_points) < 20:
            messagebox.showerror("Error", "Could not detect a sufficient path."); self.status_label.config(text="Analysis failed."); return
        
        # Stage 2: Ordering. The points are a jumbled mess, so we sort them.
        all_path_points = (raw_points / scale_factor).astype(int)
        x_range = np.ptp(all_path_points[:, 0]); y_range = np.ptp(all_path_points[:, 1])
        sorted_indices = np.argsort(all_path_points[:, 0] if x_range > y_range else all_path_points[:, 1])
        self.all_ordered_points = all_path_points[sorted_indices]
        
        # Stage 3: Pre-smoothing. We run a simple filter to calm down the jitters.
        self.status_label.config(text="Pre-smoothing path data..."); self.root.update_idletasks()
        window_size = 21 
        if len(self.all_ordered_points) < window_size:
            points_for_spline = self.all_ordered_points
        else:
            weights = np.repeat(1.0, window_size) / window_size
            x_smooth = np.convolve(self.all_ordered_points[:, 0], weights, 'valid')
            y_smooth = np.convolve(self.all_ordered_points[:, 1], weights, 'valid')
            points_for_spline = np.column_stack((x_smooth, y_smooth))
        
        # Stage 4: Spline Fitting! This gives the path a Hollywood makeover,
        # turning our jagged points into a movie-star smooth curve.
        self.status_label.config(text="Fitting final smooth curve..."); self.root.update_idletasks()
        try:
            if len(points_for_spline) < 4: raise ValueError("Not enough points for spline fitting.")
            x, y = points_for_spline[:, 0], points_for_spline[:, 1]
            smoothing_factor = len(points_for_spline) * 0.1
            spline_weights = np.ones(len(points_for_spline))
            spline_weights[0] = 100.0; spline_weights[-1] = 100.0 # Pin the endpoints
            
            self.tck, u = splprep([x, y], w=spline_weights, s=smoothing_factor, k=3)
            
            if self.tck:
                num_smooth_points = len(self.all_ordered_points) * 2
                self.u_params = np.linspace(u.min(), u.max(), num_smooth_points)
                x_final, y_final = splev(self.u_params, self.tck)
                self.smooth_path_points = np.column_stack((x_final, y_final))
                self.btn_slope.config(state=tk.NORMAL)
            else:
                self.smooth_path_points = None; self.tck = None; self.u_params = None
        except Exception as e:
            messagebox.showerror("Spline Error", f"Math got weird. Could not fit curve.\nError: {e}"); self.smooth_path_points = None
        
        self.btn_show_mask.config(state=tk.NORMAL)
        self.update_node_display()
        self.status_label.config(text="Path analysis and curve fitting complete.")
    
    def update_node_display(self, val=None):
        """Updates the plots based on the 'Number of Nodes' slider."""
        if self.all_ordered_points is None: return
        num_nodes = self.nodes_var.get()
        indices = np.linspace(0, len(self.all_ordered_points) - 1, num_nodes, dtype=int)
        self.displayed_nodes = self.all_ordered_points[indices]
        self.start_node = self.all_ordered_points[0]
        self.display_graph()
        self.update_math_plot()
        self.status_label.config(text=f"Displaying {num_nodes} nodes and smooth curve.")

    def display_graph(self):
        """Draws the analysis results on top of the original image."""
        self.ax.clear()
        img_rgb = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img_rgb, aspect='equal')
        if self.smooth_path_points is not None: self.ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2.5, label='Fitted Smooth Path', zorder=4)
        if self.displayed_nodes is not None: self.ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=35, zorder=5, edgecolors='black', label='Detected Nodes')
        if hasattr(self, 'start_node') and self.start_node is not None: self.ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=150, zorder=6, edgecolors='black', label='Start Node')
        self.ax.legend(); self.ax.set_title("Path Node Overlay"); self.ax.axis('off'); self.canvas.draw()
        
    def update_math_plot(self):
        """Draws the 'clean room' version of the path on the math plot."""
        self.math_ax.clear()
        if self.smooth_path_points is None:
            self.math_ax.set_title("Mathematical 2D Plot")
            self.math_ax.grid(True, linestyle='--', alpha=0.6)
            self.math_canvas.draw()
            return

        x_range_data = np.ptp(self.smooth_path_points[:, 0])
        y_range_data = np.ptp(self.smooth_path_points[:, 1])
        is_rotated = y_range_data > x_range_data

        if is_rotated:
            self.math_ax.plot(self.smooth_path_points[:, 1], self.smooth_path_points[:, 0], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 1], self.displayed_nodes[:, 0], c='cyan', s=20, edgecolors='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[1], self.start_node[0], c='lime', s=100, label='Start Node', edgecolors='black')
            self.math_ax.set_xlabel("Secondary Axis (pixels)"); self.math_ax.set_ylabel("Primary Axis (pixels)")
        else:
            self.math_ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=20, edgecolors='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=100, label='Start Node', edgecolors='black')
            self.math_ax.set_xlabel("Primary Axis (pixels)"); self.math_ax.set_ylabel("Secondary Axis (pixels)")

        xlim = self.math_ax.get_xlim(); ylim = self.math_ax.get_ylim()
        x_range_plot = xlim[1] - xlim[0]; y_range_plot = ylim[1] - ylim[0]
        x_center = (xlim[0] + xlim[1]) / 2; y_center = (ylim[0] + ylim[1]) / 2
        max_range = max(x_range_plot, y_range_plot)
        plot_dimension = max_range * 1.15
        self.math_ax.set_xlim(x_center - plot_dimension / 2, x_center + plot_dimension / 2)
        self.math_ax.set_ylim(y_center - plot_dimension / 2, y_center + plot_dimension / 2)
        
        self.math_ax.set_title("Mathematical 2D Plot"); self.math_ax.grid(True, linestyle='--', alpha=0.6)
        self.math_ax.set_aspect('equal', adjustable='box'); self.math_ax.legend(); self.math_ax.invert_yaxis()
        self.math_fig.tight_layout(); self.math_canvas.draw()
    
    def open_slope_visualizer(self):
        if self.smooth_path_points is None or self.tck is None:
            messagebox.showinfo("Info", "Gotta analyze a path first, partner."); return
        InteractiveSlopeWindow(self.root, self.smooth_path_points, self.tck, self.u_params)

    def display_image(self, cv_img, title=""):
        self.ax.clear(); img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal'); self.ax.set_title(title); self.ax.axis('off'); self.canvas.draw()
    
    def show_full_mask(self):
        if self.final_mask is not None: MaskViewer(self.root, self.final_mask)
        else: messagebox.showinfo("Info", "No mask has been generated yet.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: THE GRAND FINALE (Execution Block)
# The sacred incantation that summons the GUI from the digital ether.
# This is where the magic begins. Lights, camera, analyze!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()