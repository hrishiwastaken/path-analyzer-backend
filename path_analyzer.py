import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import PanedWindow
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 1: CORE IMAGE PROCESSING (Unchanged)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def skeletonize_image(image):
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
    img = cv2.imread(image_path)
    if img is None: return None, None, None, 1.0
    original_shape = img.shape
    scale_factor = 1.0
    if original_shape[1] > 1000:
        scale_percent = 50; scale_factor = scale_percent / 100.0
        width = int(img.shape[1] * scale_factor); height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return cv2.imread(image_path), None, mask, scale_factor
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    erosion_kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)
    skeleton = skeletonize_image(eroded_mask)
    rows, cols = np.where(skeleton > 0)
    if len(rows) < 2: return cv2.imread(image_path), None, skeleton, scale_factor
    all_points_scaled = np.column_stack((cols, rows))
    return cv2.imread(image_path), all_points_scaled, clean_mask, scale_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: GUI HELPER WINDOWS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MaskViewer(tk.Toplevel):
    def __init__(self, parent, mask_image):
        super().__init__(parent)
        self.title("Full Detected Mask"); self.geometry("600x600")
        self.mask_image = mask_image
        self.mask_label = tk.Label(self); self.mask_label.pack(fill=tk.BOTH, expand=True)
        self.bind("<Configure>", self.on_resize)
    def on_resize(self, event=None):
        container_w, container_h = self.winfo_width(), self.winfo_height()
        if container_w < 2 or container_h < 2: return
        img_h, img_w = self.mask_image.shape
        scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(self.mask_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)

class ColorTunerWindow(tk.Toplevel):
    def __init__(self, parent, image_path, current_lower, current_upper, on_apply_callback):
        super().__init__(parent)
        self.title("Color Tuner"); self.geometry("800x700")
        self.on_apply_callback = on_apply_callback
        img = cv2.imread(image_path); img_for_hsv = img.copy()
        if img.shape[1] > 1000:
            scale_percent = 50; width = int(img.shape[1] * scale_percent / 100); height = int(img.shape[0] * scale_percent / 100)
            img_for_hsv = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        self.hsv = cv2.cvtColor(img_for_hsv, cv2.COLOR_BGR2HSV)
        controls_container = tk.Frame(self); controls_container.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
        self.mask_label = tk.Label(self, bg="black"); self.mask_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sliders_frame = tk.Frame(controls_container); sliders_frame.pack(fill=tk.X)
        self.h_min = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Min", command=self.update_mask); self.h_min.set(current_lower[0]); self.h_min.pack(fill=tk.X)
        self.s_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Min", command=self.update_mask); self.s_min.set(current_lower[1]); self.s_min.pack(fill=tk.X)
        self.v_min = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Min", command=self.update_mask); self.v_min.set(current_lower[2]); self.v_min.pack(fill=tk.X)
        self.h_max = tk.Scale(sliders_frame, from_=0, to=179, orient=tk.HORIZONTAL, label="H Max", command=self.update_mask); self.h_max.set(current_upper[0]); self.h_max.pack(fill=tk.X)
        self.s_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="S Max", command=self.update_mask); self.s_max.set(current_upper[1]); self.s_max.pack(fill=tk.X)
        self.v_max = tk.Scale(sliders_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="V Max", command=self.update_mask); self.v_max.set(current_upper[2]); self.v_max.pack(fill=tk.X)
        btn_frame = tk.Frame(controls_container, pady=5); btn_frame.pack()
        tk.Button(btn_frame, text="Apply", command=self.apply_and_close).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        self.after(100, self.update_mask)
    def update_mask(self, val=None):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        mask = cv2.inRange(self.hsv, lower, upper)
        container_w, container_h = self.mask_label.winfo_width(), self.mask_label.winfo_height()
        if container_w < 2 or container_h < 2: self.after(100, self.update_mask); return
        img_h, img_w = mask.shape; scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)
    def apply_and_close(self):
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        self.on_apply_callback(lower, upper); self.destroy()

class InteractiveSlopeWindow(tk.Toplevel):
    def __init__(self, parent, original_cv_image, path_points, tck, u_params):
        super().__init__(parent)
        self.title("Interactive Slope Analysis (Side-by-Side)"); self.geometry("1300x700")
        self.original_cv_image = original_cv_image
        self.path_points, self.tck, self.u_params = path_points, tck, u_params
        x_range = np.ptp(self.path_points[:, 0]); y_range = np.ptp(self.path_points[:, 1])
        self.is_math_plot_rotated = y_range > x_range
        plots_frame = tk.Frame(self); plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = tk.Frame(self, pady=10); controls_frame.pack(fill=tk.X, expand=False, padx=10)
        plot_paned_window = PanedWindow(plots_frame, orient=tk.HORIZONTAL); plot_paned_window.pack(fill=tk.BOTH, expand=True)
        img_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(img_plot_frame, weight=1)
        self.fig_img = Figure(); self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_plot_frame); self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        math_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(math_plot_frame, weight=1)
        self.fig_math = Figure(); self.ax_math = self.fig_math.add_subplot(111)
        self.canvas_math = FigureCanvasTkAgg(self.fig_math, master=math_plot_frame); self.canvas_math.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        data_frame = tk.Frame(controls_frame); data_frame.pack(side=tk.LEFT, padx=(0, 20))
        self.slope_var = tk.StringVar(value="Slope (dy/dx): -"); self.int_slope_var = tk.StringVar(value="Integer Slope: -"); self.angle_var = tk.StringVar(value="Angle: -")
        tk.Label(data_frame, textvariable=self.slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.int_slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.angle_var, font=("Helvetica", 11)).pack(anchor="w")
        slider_frame = tk.Frame(controls_frame); slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.slider = tk.Scale(slider_frame, from_=0, to=len(self.u_params) - 1, orient=tk.HORIZONTAL, command=self._update_plots, label="Position Along Path"); self.slider.pack(fill=tk.X, expand=True)
        self._draw_initial_plots()
        self.after(100, lambda: self.slider.set(len(self.path_points) // 2))

    def _draw_initial_plots(self):
        img_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(img_rgb, aspect='equal')
        self.ax_img.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2.5, alpha=0.8)
        self.point_on_img, = self.ax_img.plot([], [], 'o', color='magenta', markersize=12, markeredgecolor='black', zorder=10)
        self.tangent_on_img, = self.ax_img.plot([], [], color='yellow', linestyle='--', linewidth=2.5, zorder=9)
        self.ax_img.set_title("Image Overlay View"); self.ax_img.axis('off'); self.fig_img.tight_layout()
        if self.is_math_plot_rotated:
            self.ax_math.plot(self.path_points[:, 1], self.path_points[:, 0], color='red', linewidth=2)
            self.ax_math.set_xlabel("Secondary Axis (pixels)"); self.ax_math.set_ylabel("Primary Axis (pixels)")
        else:
            self.ax_math.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2)
            self.ax_math.set_xlabel("Primary Axis (pixels)"); self.ax_math.set_ylabel("Secondary Axis (pixels)")
        self.point_on_math, = self.ax_math.plot([], [], 'o', color='magenta', markersize=10, markeredgecolor='black', zorder=10)
        self.tangent_on_math, = self.ax_math.plot([], [], color='blue', linestyle='--', linewidth=2, zorder=9)
        self.ax_math.set_aspect('equal', adjustable='box'); self.ax_math.invert_yaxis(); self.ax_math.grid(True, linestyle='--', alpha=0.6)
        self.ax_math.set_title("Mathematical View"); self.fig_math.tight_layout()
        self.canvas_img.draw(); self.canvas_math.draw()

    def _update_plots(self, val):
        index = int(val); u_val = self.u_params[index]
        current_point = splev(u_val, self.tck, der=0)
        dx_dt, dy_dt = splev(u_val, self.tck, der=1)
        cx, cy = current_point[0], current_point[1]
        angle_deg = np.degrees(np.arctan2(dy_dt, dx_dt))
        if abs(dx_dt) < 1e-6:
            self.slope_var.set("Slope (dy/dx): Infinity"); self.int_slope_var.set("Integer Slope: N/A")
        else:
            slope_val = dy_dt / dx_dt
            self.slope_var.set(f"Slope (dy/dx): {slope_val:.2f}"); self.int_slope_var.set(f"Integer Slope: {int(round(slope_val))}")
        self.angle_var.set(f"Angle: {angle_deg:.1f}°")
        magnitude = np.hypot(dx_dt, dy_dt)
        line_length = 50
        if magnitude > 1e-6:
            ux, uy = dx_dt / magnitude, dy_dt / magnitude
            x_coords = [cx - ux * line_length, cx + ux * line_length]; y_coords = [cy - uy * line_length, cy + uy * line_length]
        else: x_coords, y_coords = [], []
        self.point_on_img.set_data([cx], [cy]); self.tangent_on_img.set_data(x_coords, y_coords)
        if self.is_math_plot_rotated:
            self.point_on_math.set_data([cy], [cx]); self.tangent_on_math.set_data(y_coords, x_coords)
        else:
            self.point_on_math.set_data([cx], [cy]); self.tangent_on_math.set_data(x_coords, y_coords)
        self.canvas_img.draw_idle(); self.canvas_math.draw_idle()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: THE MAIN APPLICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PathAnalyzerApp:
    def __init__(self, root):
        self.root = root; self.root.title("Path Analyzer"); self.root.geometry("1200x800")
        self.image_path = None; self.original_cv_img = None; self.final_mask = None
        self.raw_skeleton_points = None; self.all_ordered_points = None; self.active_ordered_points = None
        self.displayed_nodes = None; self.smooth_path_points = None; self.start_node = None
        self.tck = None; self.u_params = None
        self.hsv_lower = np.array([0, 70, 50]); self.hsv_upper = np.array([179, 255, 255])
        self.selection_mode = "none"; self.manual_start_point = None; self.manual_end_point = None
        self.click_handler_id = None; self.motion_handler_id = None
        self.hovered_node_marker = None; self.currently_hovered_node_index = None
        self._build_gui()

    def _build_gui(self):
        self.main_paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)
        self.left_pane = tk.Frame(self.main_paned_window); self.main_paned_window.add(self.left_pane, weight=3)
        self.right_paned_window = PanedWindow(self.main_paned_window, orient=tk.VERTICAL); self.main_paned_window.add(self.right_paned_window, weight=1)
        self.fig = Figure(); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- FIX: Stable layout for control panel ---
        controls_frame = tk.Frame(self.right_paned_window)
        math_plot_frame = tk.Frame(self.right_paned_window)
        self.right_paned_window.add(controls_frame, weight=0) # weight=0 makes it a fixed size
        self.right_paned_window.add(math_plot_frame, weight=1) # weight=1 makes this expand

        workflow_frame = tk.LabelFrame(controls_frame, text="Workflow Steps", padx=10, pady=10)
        workflow_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        self.btn_load = tk.Button(workflow_frame, text="1. Load Image", command=self.load_image); self.btn_load.pack(fill=tk.X, pady=2)
        self.btn_tuner = tk.Button(workflow_frame, text="2. Tune Color Range...", command=self.open_color_tuner, state=tk.DISABLED); self.btn_tuner.pack(fill=tk.X, pady=2)
        self.btn_analyze = tk.Button(workflow_frame, text="3. Analyze & Select Endpoints", command=self.analyze_path, state=tk.DISABLED); self.btn_analyze.pack(fill=tk.X, pady=(2, 0))

        tuning_frame = tk.LabelFrame(controls_frame, text="Path Visualization Tuning", padx=10, pady=5)
        tuning_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(tuning_frame, text="Path Endpoint Control (Fine-tuning):").pack()
        self.path_length_var = tk.IntVar(value=100)
        self.path_length_slider = tk.Scale(tuning_frame, from_=2, to=100, orient=tk.HORIZONTAL, variable=self.path_length_var, command=self._update_path_length, state=tk.DISABLED, showvalue=0); self.path_length_slider.pack(fill=tk.X)
        tk.Label(tuning_frame, text="Number of Nodes to Display:").pack()
        self.nodes_var = tk.IntVar(value=150)
        self.nodes_slider = tk.Scale(tuning_frame, from_=2, to=1000, orient=tk.HORIZONTAL, variable=self.nodes_var, command=self.update_node_display_during_selection, showvalue=0); self.nodes_slider.pack(fill=tk.X)
        
        export_frame = tk.LabelFrame(controls_frame, text="Export for Vehicle Control", padx=10, pady=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(export_frame, text="Export Path Resolution (Number of Steps):").pack()
        self.export_nodes_var = tk.IntVar(value=100)
        self.export_nodes_slider = tk.Scale(export_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.export_nodes_var, state=tk.DISABLED); self.export_nodes_slider.pack(fill=tk.X, pady=(0, 5))
        self.btn_export = tk.Button(export_frame, text="4. Export Path to CSV", command=self.export_path_data, state=tk.DISABLED, bg="#ccffcc"); self.btn_export.pack(fill=tk.X)
        
        tools_frame = tk.LabelFrame(controls_frame, text="Diagnostic Tools", padx=10, pady=5)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_show_mask = tk.Button(tools_frame, text="Show Full Mask", command=self.show_full_mask, state=tk.DISABLED); self.btn_show_mask.pack(fill=tk.X, pady=2)
        self.btn_slope = tk.Button(tools_frame, text="Interactive Slope Analysis", command=self.open_slope_visualizer, state=tk.DISABLED); self.btn_slope.pack(fill=tk.X, pady=2)
        
        self.status_label = tk.Label(controls_frame, text="Please load an image to begin.", wraplength=350, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        
        tk.Label(math_plot_frame, text="Mathematical 2D Plot").pack()
        self.math_fig = Figure(); self.math_ax = self.math_fig.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_fig, master=math_plot_frame); self.math_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

    def open_slope_visualizer(self):
        if self.smooth_path_points is None or self.tck is None: messagebox.showinfo("Info", "No path analyzed yet."); return
        InteractiveSlopeWindow(self.root, self.original_cv_img, self.smooth_path_points, self.tck, self.u_params)

    def _disconnect_event_handlers(self):
        if self.click_handler_id: self.canvas.mpl_disconnect(self.click_handler_id); self.click_handler_id = None
        if self.motion_handler_id: self.canvas.mpl_disconnect(self.motion_handler_id); self.motion_handler_id = None
        self._remove_hover_marker()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path: return
        self.image_path = path; self.original_cv_img = cv2.imread(self.image_path)
        if self.original_cv_img is None: messagebox.showerror("Error", "Failed to load image."); return
        self.raw_skeleton_points = None; self.all_ordered_points = None; self.active_ordered_points = None
        self.display_image(self.original_cv_img, "Original Image Loaded")
        self.update_math_plot()
        self.btn_tuner.config(state=tk.NORMAL); self.btn_analyze.config(state=tk.DISABLED)
        self.btn_show_mask.config(state=tk.DISABLED); self.btn_slope.config(state=tk.DISABLED)
        self.path_length_slider.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
        self._disconnect_event_handlers()
        self.status_label.config(text="Image loaded. Please tune color range.")

    def analyze_path(self):
        self.status_label.config(text="Extracting path points..."); self.root.update_idletasks()
        self.btn_analyze.config(state=tk.DISABLED)
        self.original_cv_img, raw_points, self.final_mask, scale_factor = preprocess_and_extract_path(self.image_path, self.hsv_lower, self.hsv_upper)
        if raw_points is None or len(raw_points) < 20:
            messagebox.showerror("Error", "Could not detect a sufficient path."); self.status_label.config(text="Analysis failed.")
            self.btn_analyze.config(state=tk.NORMAL); return
        self.raw_skeleton_points = (raw_points / scale_factor).astype(int)
        self.smooth_path_points = None
        self.update_node_display_during_selection()
        self.selection_mode = "start"; self.manual_start_point, self.manual_end_point = None, None
        self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.motion_handler_id = self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.status_label.config(text="Path detected. Hover and click to select the START node.")

    def _on_canvas_motion(self, event):
        if event.inaxes != self.ax or self.displayed_nodes is None or len(self.displayed_nodes) == 0: self._remove_hover_marker(); return
        mouse_point = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.displayed_nodes - mouse_point, axis=1)
        closest_node_index = np.argmin(distances); min_dist = distances[closest_node_index]
        hover_threshold = 15
        if min_dist < hover_threshold:
            if self.currently_hovered_node_index != closest_node_index:
                self._remove_hover_marker()
                node_coords = self.displayed_nodes[closest_node_index]
                self.hovered_node_marker, = self.ax.plot(node_coords[0], node_coords[1], 'o', markersize=18, color='yellow', alpha=0.7, zorder=10)
                self.currently_hovered_node_index = closest_node_index; self.canvas.draw_idle()
        else: self._remove_hover_marker()
            
    def _remove_hover_marker(self):
        if self.hovered_node_marker:
            self.hovered_node_marker.remove(); self.hovered_node_marker = None
            self.currently_hovered_node_index = None; self.canvas.draw_idle()
            
    def _on_canvas_click(self, event):
        if self.selection_mode == "none" or self.currently_hovered_node_index is None: return
        selected_node = self.displayed_nodes[self.currently_hovered_node_index]
        if self.selection_mode == "start":
            self.manual_start_point = selected_node; self.selection_mode = "end"
            self.status_label.config(text="Start node set. Hover and click to select the END node.")
        elif self.selection_mode == "end":
            self.manual_end_point = selected_node; self.selection_mode = "none"
            self._disconnect_event_handlers()
            self.status_label.config(text="Endpoints selected. Processing...")
            self.root.after(50, self._process_manual_endpoints)
            
    def _process_manual_endpoints(self):
        self.status_label.config(text="Ordering path from selected start to end..."); self.root.update_idletasks()
        self.all_ordered_points = self.order_points_by_proximity(self.raw_skeleton_points, start_point=self.manual_start_point, end_point=self.manual_end_point)
        self.btn_show_mask.config(state=tk.NORMAL)
        num_total_points = len(self.all_ordered_points)
        self.path_length_slider.config(to=num_total_points, state=tk.NORMAL); self.path_length_var.set(num_total_points)
        self._update_path_length()
        self.status_label.config(text="Path analysis complete."); self.btn_analyze.config(state=tk.NORMAL)

    def order_points_by_proximity(self, points, start_point, end_point):
        if len(points) < 2: return points
        points_list = points.tolist()
        ordered_points = []
        dists = np.linalg.norm(np.array(points_list) - np.array(start_point), axis=1)
        start_index = np.argmin(dists)
        current_point = points_list.pop(start_index)
        ordered_points.append(current_point)
        if np.array_equal(current_point, end_point): return np.array(ordered_points)
        while points_list:
            distances_sq = np.sum((np.array(points_list) - current_point)**2, axis=1)
            closest_index = np.argmin(distances_sq)
            current_point = points_list.pop(closest_index)
            ordered_points.append(current_point)
            if np.array_equal(current_point, end_point): break
        return np.array(ordered_points)
        
    def _update_path_length(self, val=None):
        if self.all_ordered_points is None: return
        length = self.path_length_var.get()
        self.active_ordered_points = self.all_ordered_points[:length]
        if len(self.active_ordered_points) < 4:
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
            self.update_node_display_during_selection(); return
        window_size = min(21, len(self.active_ordered_points) -1 if len(self.active_ordered_points) % 2 == 0 else len(self.active_ordered_points) -2)
        if window_size < 3: window_size = 3
        points_for_spline = self.active_ordered_points
        if len(self.active_ordered_points) >= window_size:
            weights = np.repeat(1.0, window_size) / window_size
            x_smooth = np.convolve(self.active_ordered_points[:, 0], weights, 'valid'); y_smooth = np.convolve(self.active_ordered_points[:, 1], weights, 'valid')
            points_for_spline = np.column_stack((x_smooth, y_smooth))
        try:
            if len(points_for_spline) < 4: raise ValueError("Not enough points for spline fitting.")
            x, y = points_for_spline[:, 0], points_for_spline[:, 1]
            smoothing_factor = len(points_for_spline) * 0.1
            self.tck, u = splprep([x, y], s=smoothing_factor, k=3)
            if self.tck:
                num_smooth_points = len(self.active_ordered_points) * 2; self.u_params = np.linspace(u.min(), u.max(), num_smooth_points)
                x_final, y_final = splev(self.u_params, self.tck); self.smooth_path_points = np.column_stack((x_final, y_final))
                self.btn_slope.config(state=tk.NORMAL)
                self.btn_export.config(state=tk.NORMAL); self.export_nodes_slider.config(state=tk.NORMAL)
            else: raise ValueError("Spline creation failed.")
        except Exception:
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
        self.update_node_display_during_selection()

    def update_node_display_during_selection(self, val=None):
        points_to_display_from = self.active_ordered_points if self.active_ordered_points is not None else self.raw_skeleton_points
        if points_to_display_from is None or len(points_to_display_from) == 0:
            self.displayed_nodes = None; self.start_node = None
        else:
            num_points_in_path = len(points_to_display_from)
            num_nodes = self.nodes_var.get()
            if num_nodes > num_points_in_path: num_nodes = num_points_in_path
            num_nodes = max(2, num_nodes)
            indices = np.linspace(0, num_points_in_path - 1, num_nodes, dtype=int)
            self.displayed_nodes = points_to_display_from[indices]
            self.start_node = self.active_ordered_points[0] if self.active_ordered_points is not None else None
        self.display_graph(); self.update_math_plot()

    def display_graph(self):
        self.ax.clear(); img_rgb = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal')
        show_legend = self.smooth_path_points is not None
        if self.smooth_path_points is not None: self.ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2.5, label='Fitted Smooth Path', zorder=4)
        if self.displayed_nodes is not None: self.ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=35, zorder=5, edgecolors='black', label='Detected Nodes')
        if self.start_node is not None: self.ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=150, zorder=6, edgecolors='black', label='Start Node')
        if show_legend: self.ax.legend()
        self.ax.set_title("Path Node Overlay"); self.ax.axis('off'); self.canvas.draw()
        
    def update_math_plot(self):
        self.math_ax.clear(); self.math_ax.set_title("Mathematical 2D Plot"); self.math_ax.grid(True, linestyle='--', alpha=0.6)
        if self.smooth_path_points is None or len(self.smooth_path_points) == 0: self.math_canvas.draw(); return
        x_range_data = np.ptp(self.smooth_path_points[:, 0]); y_range_data = np.ptp(self.smooth_path_points[:, 1]); is_rotated = y_range_data > x_range_data
        if is_rotated:
            self.math_ax.plot(self.smooth_path_points[:, 1], self.smooth_path_points[:, 0], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 1], self.displayed_nodes[:, 0], c='cyan', s=20, ec='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[1], self.start_node[0], c='lime', s=100, label='Start Node', ec='black')
            self.math_ax.set_xlabel("Secondary Axis (pixels)"); self.math_ax.set_ylabel("Primary Axis (pixels)")
        else:
            self.math_ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=20, ec='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=100, label='Start Node', ec='black')
            self.math_ax.set_xlabel("Primary Axis (pixels)"); self.math_ax.set_ylabel("Secondary Axis (pixels)")
        self.math_ax.set_aspect('equal', adjustable='box'); self.math_ax.invert_yaxis()
        self.math_ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
        self.math_fig.subplots_adjust(right=0.75)
        self.math_canvas.draw()

    def on_tuner_apply(self, new_lower, new_upper):
        self.hsv_lower, self.hsv_upper = new_lower, new_upper
        self.status_label.config(text="New color range applied. Ready to analyze.")
        self.btn_analyze.config(state=tk.NORMAL)

    def open_color_tuner(self):
        if self.image_path: ColorTunerWindow(self.root, self.image_path, self.hsv_lower, self.hsv_upper, self.on_tuner_apply)

    def display_image(self, cv_img, title=""):
        self.ax.clear(); img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal'); self.ax.set_title(title); self.ax.axis('off'); self.canvas.draw()
    
    def show_full_mask(self):
        if self.final_mask is not None: MaskViewer(self.root, self.final_mask)
        else: messagebox.showinfo("Info", "No mask has been generated yet.")

    def export_path_data(self):
        import csv
        if self.tck is None or self.u_params is None:
            messagebox.showerror("Error", "No valid smooth path has been generated yet."); return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")], title="Save Path Data")
        if not file_path: return
        num_points = self.export_nodes_var.get()
        u_vals_export = np.linspace(self.u_params.min(), self.u_params.max(), num_points)
        points = np.column_stack(splev(u_vals_export, self.tck))
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'y', 'distance_segment', 'turning_angle_deg'])
                prev_heading_rad = 0.0
                for i in range(len(points)):
                    x, y = points[i]
                    if i == 0:
                        dist_segment, turn_angle_deg = 0.0, 0.0
                        if len(points) > 1:
                            dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
                            prev_heading_rad = np.arctan2(dy, dx)
                    else:
                        prev_x, prev_y = points[i-1]; dx, dy = x - prev_x, y - prev_y
                        dist_segment = np.hypot(dx, dy)
                        current_heading_rad = np.arctan2(dy, dx) if (dx**2 + dy**2) > 1e-12 else prev_heading_rad
                        turn_angle_rad = current_heading_rad - prev_heading_rad
                        turn_angle_rad = (turn_angle_rad + np.pi) % (2 * np.pi) - np.pi
                        turn_angle_deg = np.degrees(turn_angle_rad)
                        prev_heading_rad = current_heading_rad
                    writer.writerow([f"{x:.2f}", f"{y:.2f}", f"{dist_segment:.3f}", f"{turn_angle_deg:.3f}"])
            messagebox.showinfo("Success", f"Path data with {num_points} steps successfully exported to:\n{file_path}")
        except Exception as e: messagebox.showerror("Error", f"Failed to write to file: {e}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: EXECUTION BLOCK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()
