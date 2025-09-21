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
# SECTION 1: CORE IMAGE PROCESSING (Functionality Corrected Again)
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
    """A robust pipeline to extract a clean set of path coordinates from an image."""
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

    # --- FIX STARTS HERE ---
    # The previous MORPH_OPEN was too destructive. The original MORPH_CLOSE was too aggressive.
    # This is the correct balance: A MORPH_CLOSE operation with a small number of
    # iterations. It's strong enough to fill gaps in the line but not so strong
    # that it creates a giant blob.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # --- FIX ENDS HERE ---

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return cv2.imread(image_path), None, mask, scale_factor
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # A single erosion pass is sufficient to thin the line for skeletonization.
    # The previous value of 2 was also too destructive.
    erosion_kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)

    skeleton = skeletonize_image(eroded_mask)
    rows, cols = np.where(skeleton > 0)
    if len(rows) < 2: return cv2.imread(image_path), None, skeleton, scale_factor
    all_points_scaled = np.column_stack((cols, rows))
    return cv2.imread(image_path), all_points_scaled, clean_mask, scale_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: GUI HELPER WINDOWS (Unchanged)
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
        self.on_apply_callback(lower, upper)
        self.destroy()

class InteractiveSlopeWindow(tk.Toplevel):
    def __init__(self, parent, path_points, tck, u_params):
        super().__init__(parent)
        self.title("Interactive Slope Analysis"); self.geometry("950x700")
        self.path_points, self.tck, self.u_params = path_points, tck, u_params
        x_range = np.ptp(self.path_points[:, 0]); y_range = np.ptp(self.path_points[:, 1])
        self.is_rotated_for_display = y_range > x_range
        main_frame = tk.Frame(self); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_frame = tk.Frame(main_frame); plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_frame = tk.Frame(main_frame, padx=10, pady=10, relief=tk.SUNKEN, borderwidth=2); data_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.fig = Figure(); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tk.Label(data_frame, text="Slope Data", font=("Helvetica", 14, "bold")).pack(pady=(0, 15))
        self.slope_var = tk.StringVar(value="Slope (dy/dx): -"); self.int_slope_var = tk.StringVar(value="Integer Slope: -"); self.angle_var = tk.StringVar(value="Angle: -")
        tk.Label(data_frame, textvariable=self.slope_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)
        tk.Label(data_frame, textvariable=self.int_slope_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)
        tk.Label(data_frame, textvariable=self.angle_var, font=("Helvetica", 12)).pack(anchor="w", pady=5)
        self.slider = tk.Scale(plot_frame, from_=0, to=len(self.u_params) - 1, orient=tk.HORIZONTAL, command=self._update_plot, label="Position Along Path"); self.slider.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self._draw_initial_plot()
        self.after(100, lambda: self.slider.set(len(self.path_points) // 2))

    def _draw_initial_plot(self):
        self.ax.clear()
        if self.is_rotated_for_display:
            self.ax.plot(self.path_points[:, 1], self.path_points[:, 0], color='red', linewidth=2); self.ax.set_xlabel("Secondary Axis (pixels)"); self.ax.set_ylabel("Primary Axis (pixels)")
        else:
            self.ax.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2); self.ax.set_xlabel("Primary Axis (pixels)"); self.ax.set_ylabel("Secondary Axis (pixels)")
        self.point_on_curve, = self.ax.plot([], [], 'o', color='magenta', markersize=12, markeredgecolor='black', zorder=10)
        self.tangent_line, = self.ax.plot([], [], color='blue', linestyle='--', linewidth=2, zorder=9)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim(); x_range_plot, y_range_plot = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x_center, y_center = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        max_range = max(x_range_plot, y_range_plot); plot_dimension = max_range * 1.1
        self.ax.set_xlim(x_center - plot_dimension / 2, x_center + plot_dimension / 2); self.ax.set_ylim(y_center - plot_dimension / 2, y_center + plot_dimension / 2)
        self.ax.set_title("Slope along the Fitted Path"); self.ax.grid(True, linestyle='--', alpha=0.6); self.ax.set_aspect('equal', adjustable='box'); self.ax.invert_yaxis(); self.fig.tight_layout(); self.canvas.draw()

    def _update_plot(self, val):
        index = int(val); u_val = self.u_params[index]
        current_point = splev(u_val, self.tck, der=0)
        dx_dt, dy_dt = splev(u_val, self.tck, der=1)
        angle_deg = np.degrees(np.arctan2(dy_dt, dx_dt))
        if abs(dx_dt) < 1e-6:
            self.slope_var.set("Slope (dy/dx): Infinity"); self.int_slope_var.set("Integer Slope: N/A")
        else:
            slope_val = dy_dt / dx_dt; self.slope_var.set(f"Slope (dy/dx): {slope_val:.2f}"); self.int_slope_var.set(f"Integer Slope: {int(round(slope_val))}")
        self.angle_var.set(f"Angle: {angle_deg:.1f}°")
        magnitude = np.sqrt(dx_dt**2 + dy_dt**2)
        if magnitude > 1e-6:
            ux, uy = dx_dt / magnitude, dy_dt / magnitude; plot_width = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]; line_length = plot_width * 0.15
            cx, cy = current_point[0], current_point[1]; x_coords = [cx - ux * line_length / 2, cx + ux * line_length / 2]; y_coords = [cy - uy * line_length / 2, cy + uy * line_length / 2]
            if self.is_rotated_for_display: self.tangent_line.set_data(y_coords, x_coords)
            else: self.tangent_line.set_data(x_coords, y_coords)
        else: self.tangent_line.set_data([], [])
        if self.is_rotated_for_display: self.point_on_curve.set_data([current_point[1]], [current_point[0]])
        else: self.point_on_curve.set_data([current_point[0]], [current_point[1]])
        self.canvas.draw_idle()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: THE MAIN APPLICATION (Unchanged)
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
        self.main_paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL); self.main_paned_window.pack(fill=tk.BOTH, expand=True)
        self.left_pane = tk.Frame(self.main_paned_window); self.main_paned_window.add(self.left_pane, weight=3)
        self.right_paned_window = PanedWindow(self.main_paned_window, orient=tk.VERTICAL); self.main_paned_window.add(self.right_paned_window, weight=1)
        self.fig = Figure(); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        controls_frame = tk.Frame(self.right_paned_window, height=350); self.right_paned_window.add(controls_frame, weight=0)
        math_plot_frame = tk.Frame(self.right_paned_window); self.right_paned_window.add(math_plot_frame, weight=1)
        controls_frame.pack_propagate(False)

        self.btn_load = tk.Button(controls_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(fill=tk.X, padx=10, pady=5)
        self.btn_tuner = tk.Button(controls_frame, text="Tune Color Range...", command=self.open_color_tuner, state=tk.DISABLED)
        self.btn_tuner.pack(fill=tk.X, padx=10)
        # MODIFIED: Button text is more descriptive of the new workflow
        self.btn_analyze = tk.Button(controls_frame, text="Analyze Path & Select Endpoints", command=self.analyze_path, state=tk.DISABLED)
        self.btn_analyze.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(controls_frame, text="Path Endpoint Control (Fine-tuning):").pack(padx=10, pady=(5,0))
        self.path_length_var = tk.IntVar(value=100)
        self.path_length_slider = tk.Scale(controls_frame, from_=2, to=100, orient=tk.HORIZONTAL, variable=self.path_length_var, command=self._update_path_length, state=tk.DISABLED)
        self.path_length_slider.pack(fill=tk.X, padx=10)
        
        tk.Label(controls_frame, text="Number of Nodes to Display:").pack(padx=10)
        self.nodes_var = tk.IntVar(value=150)
        self.nodes_slider = tk.Scale(controls_frame, from_=2, to=1000, orient=tk.HORIZONTAL, variable=self.nodes_var, command=self.update_node_display_during_selection)
        self.nodes_slider.pack(fill=tk.X, padx=10)
        
        self.btn_show_mask = tk.Button(controls_frame, text="Show Full Mask", command=self.show_full_mask, state=tk.DISABLED)
        self.btn_show_mask.pack(fill=tk.X, padx=10, pady=5)
        self.btn_slope = tk.Button(controls_frame, text="Interactive Slope Analysis", command=self.open_slope_visualizer, state=tk.DISABLED)
        self.btn_slope.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(controls_frame, text="Please load an image to begin.", wraplength=350)
        self.status_label.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(math_plot_frame, text="Mathematical 2D Plot").pack()
        self.math_fig = Figure(); self.math_ax = self.math_fig.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_fig, master=math_plot_frame)
        self.math_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

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
        self._disconnect_event_handlers()
        self.status_label.config(text="Image loaded. Please tune color range.")

    def analyze_path(self):
        """MODIFIED: This function now kicks off the mandatory interactive selection process."""
        self.status_label.config(text="Extracting path points..."); self.root.update_idletasks()
        # Disable button to prevent re-clicks during selection
        self.btn_analyze.config(state=tk.DISABLED)

        self.original_cv_img, raw_points, self.final_mask, scale_factor = preprocess_and_extract_path(self.image_path, self.hsv_lower, self.hsv_upper)
        if raw_points is None or len(raw_points) < 20:
            messagebox.showerror("Error", "Could not detect a sufficient path."); self.status_label.config(text="Analysis failed.")
            self.btn_analyze.config(state=tk.NORMAL) # Re-enable button on failure
            return
        self.raw_skeleton_points = (raw_points / scale_factor).astype(int)
        
        # Prepare for selection: display raw nodes on the image
        self.smooth_path_points = None # Clear any previous smooth path
        self.update_node_display_during_selection()
        
        # Enter the interactive selection mode
        self.selection_mode = "start"
        self.manual_start_point, self.manual_end_point = None, None
        self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.motion_handler_id = self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.status_label.config(text="Path detected. Hover and click to select the START node.")

    def _on_canvas_motion(self, event):
        if event.inaxes != self.ax or self.displayed_nodes is None or len(self.displayed_nodes) == 0:
            self._remove_hover_marker(); return
        mouse_point = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.displayed_nodes - mouse_point, axis=1)
        closest_node_index = np.argmin(distances)
        min_dist = distances[closest_node_index]
        
        hover_threshold = 15 # pixels
        if min_dist < hover_threshold:
            if self.currently_hovered_node_index != closest_node_index:
                self._remove_hover_marker()
                node_coords = self.displayed_nodes[closest_node_index]
                self.hovered_node_marker, = self.ax.plot(node_coords[0], node_coords[1], 'o', markersize=18, color='yellow', alpha=0.7, zorder=10)
                self.currently_hovered_node_index = closest_node_index
                self.canvas.draw_idle()
        else:
            self._remove_hover_marker()
            
    def _remove_hover_marker(self):
        if self.hovered_node_marker:
            self.hovered_node_marker.remove(); self.hovered_node_marker = None
            self.currently_hovered_node_index = None
            self.canvas.draw_idle()
            
    def _on_canvas_click(self, event):
        if self.selection_mode == "none" or self.currently_hovered_node_index is None: return
        selected_node = self.displayed_nodes[self.currently_hovered_node_index]
        if self.selection_mode == "start":
            self.manual_start_point = selected_node
            self.selection_mode = "end"
            self.status_label.config(text="Start node set. Hover and click to select the END node.")
        elif self.selection_mode == "end":
            self.manual_end_point = selected_node
            self.selection_mode = "none"
            self._disconnect_event_handlers()
            self.status_label.config(text="Endpoints selected. Processing...")
            self.root.after(50, self._process_manual_endpoints)
            
    def _process_manual_endpoints(self):
        """The final stage of analysis, triggered after endpoint selection."""
        self.status_label.config(text="Ordering path from selected start to end..."); self.root.update_idletasks()
        self.all_ordered_points = self.order_points_by_proximity(self.raw_skeleton_points, start_point=self.manual_start_point, end_point=self.manual_end_point)
        
        # Enable final controls
        self.btn_show_mask.config(state=tk.NORMAL)
        num_total_points = len(self.all_ordered_points)
        self.path_length_slider.config(to=num_total_points, state=tk.NORMAL); self.path_length_var.set(num_total_points)
        
        # Perform the final smoothing and display
        self._update_path_length()
        self.status_label.config(text="Path analysis complete.")
        self.btn_analyze.config(state=tk.NORMAL) # Re-enable the main button

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
            else: raise ValueError("Spline creation failed.")
        except Exception: self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
        self.update_node_display_during_selection()

    def update_node_display_during_selection(self, val=None):
        """MODIFIED: A version of update_node_display that can work with raw or ordered points."""
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
            # Only set the start_node if the path has actually been ordered
            self.start_node = self.active_ordered_points[0] if self.active_ordered_points is not None else None
        self.display_graph(); self.update_math_plot()

    def display_graph(self):
        self.ax.clear(); img_rgb = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal')
        # Only show legend if the path is finalized
        show_legend = self.smooth_path_points is not None
        if self.smooth_path_points is not None: self.ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2.5, label='Fitted Smooth Path', zorder=4)
        if self.displayed_nodes is not None: self.ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=35, zorder=5, edgecolors='black', label='Detected Nodes')
        if self.start_node is not None: self.ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=150, zorder=6, edgecolors='black', label='Start Node')
        if show_legend: self.ax.legend()
        self.ax.set_title("Path Node Overlay"); self.ax.axis('off'); self.canvas.draw()
        
    def update_math_plot(self):
        self.math_ax.clear(); self.math_ax.set_title("Mathematical 2D Plot"); self.math_ax.grid(True, linestyle='--', alpha=0.6)
        if self.smooth_path_points is None or len(self.smooth_path_points) == 0: self.math_canvas.draw(); return
        x_range_data, y_range_data = np.ptp(self.smooth_path_points[:, 0]), np.ptp(self.smooth_path_points[:, 1]); is_rotated = y_range_data > x_range_data
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
        xlim, ylim = self.math_ax.get_xlim(), self.math_ax.get_ylim(); x_range_plot, y_range_plot = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x_center, y_center = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2; max_range = max(x_range_plot, y_range_plot)
        if max_range > 0:
            plot_dimension = max_range * 1.15; self.math_ax.set_xlim(x_center - plot_dimension / 2, x_center + plot_dimension / 2); self.math_ax.set_ylim(y_center - plot_dimension / 2, y_center + plot_dimension / 2)
        self.math_ax.set_aspect('equal', adjustable='box'); self.math_ax.legend(); self.math_ax.invert_yaxis(); self.math_fig.tight_layout(); self.math_canvas.draw()
    
    def on_tuner_apply(self, new_lower, new_upper):
        self.hsv_lower, self.hsv_upper = new_lower, new_upper
        self.status_label.config(text="New color range applied. Ready to analyze.")
        self.btn_analyze.config(state=tk.NORMAL)

    def open_color_tuner(self):
        if self.image_path: ColorTunerWindow(self.root, self.image_path, self.hsv_lower, self.hsv_upper, self.on_tuner_apply)
    
    def open_slope_visualizer(self):
        if self.smooth_path_points is None or self.tck is None: messagebox.showinfo("Info", "No path analyzed yet."); return
        InteractiveSlopeWindow(self.root, self.smooth_path_points, self.tck, self.u_params)

    def display_image(self, cv_img, title=""):
        self.ax.clear(); img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal'); self.ax.set_title(title); self.ax.axis('off'); self.canvas.draw()
    
    def show_full_mask(self):
        if self.final_mask is not None: MaskViewer(self.root, self.final_mask)
        else: messagebox.showinfo("Info", "No mask has been generated yet.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: EXECUTION BLOCK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()
