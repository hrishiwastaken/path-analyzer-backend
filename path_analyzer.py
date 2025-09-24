import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import PanedWindow
import cv2
import heapq
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import csv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 1: THE ALCHEMY LAB (CORE IMAGE PROCESSING)
# Here's where we take a perfectly good image and put it through a series of
# questionable experiments until it tells us its secrets.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def skeletonize_image(image):
    """
    Whittles a shape down to its bare bones. It's like image anorexia.
    We repeatedly shave off layers until only the spooky skeleton remains.
    """
    # Start with a blank canvas, a void, a black abyss.
    skeleton = np.zeros(image.shape, np.uint8)
    # This 'element' is our magical pixel-eating chisel.
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Keep chiseling away until there's nothing left of the original image.
    while True:
        # Erode! Shrink the shape.
        eroded = cv2.erode(image, element)
        # Dilate it back, but just to see what we've lost.
        temp = cv2.dilate(eroded, element)
        # The difference is the layer of "skin" we just peeled off.
        temp = cv2.subtract(image, temp)
        # Slap that skin onto our skeleton canvas.
        skeleton = cv2.bitwise_or(skeleton, temp)
        # The eroded image is now our new "original" for the next round.
        image = eroded.copy()

        # If the image is now just a sad, empty void... we're done.
        if cv2.countNonZero(image) == 0:
            break
    return skeleton

def preprocess_and_extract_path(image_path, lower_bound, upper_bound):
    """
    Our master plan:
    1. Load image.
    2. Put on our special color-goggles (HSV).
    3. Find the biggest, most interesting blob of color.
    4. Force it onto a diet until it's a skinny line (skeletonize).
    5. Kidnap all the pixels that make up that line.
    """
    img = cv2.imread(image_path)
    if img is None: return None, None, None, 1.0 # Oops, couldn't find the image. Let's not panic.

    # If the image is ridiculously huge, let's hit it with a shrink ray.
    # Nobody has time to process 8K images of a garden hose.
    scale_factor = 1.0
    if img.shape[1] > 1000:
        scale_percent = 50; scale_factor = scale_percent / 100.0
        width = int(img.shape[1] * scale_factor); height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Convert from boring BGR to magical HSV color space. It's better for finding colors. Trust me.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask that only shows the pixels within our chosen color rave.
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Mop up any stray pixel dust and fill in tiny holes. A little morphological housekeeping.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find all the distinct shapes (contours) in our mask. It's like cloud-gazing.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return cv2.imread(image_path), None, mask, scale_factor # Found nothing. Awkward.

    # We only care about the BIGGEST shape. The alpha contour. The Big Cheese.
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros(mask.shape, np.uint8)

    # ~~~~~~~~~~~~~~ BUG FIX STARTS HERE (aka "The Great Blob Incident") ~~~~~~~~~~~~~~
    # The original code used `thickness=cv2.FILLED`, which was a TERRIBLE idea.
    # It turned our beautiful, winding path into a solid, unhelpful blob.
    # Now, we draw it as a nice, reasonably thick line, which keeps its shape.
    # The skeletonizer is much happier this way.
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=5)
    # ~~~~~~~~~~~~~~~ BUG FIX ENDS HERE (Crisis Averted) ~~~~~~~~~~~~~~~

    # One last little erosion to make sure the line is thin and tidy.
    erosion_kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)

    # SKELETONIZE! (cue spooky music)
    skeleton = skeletonize_image(eroded_mask)

    # Find the coordinates of every single white pixel in our skeleton.
    rows, cols = np.where(skeleton > 0)
    if len(rows) < 2: return cv2.imread(image_path), None, skeleton, scale_factor # Not enough points to make a path. Sad.

    # Package up the points neatly and send them on their way.
    all_points_scaled = np.column_stack((cols, rows))
    return cv2.imread(image_path), all_points_scaled, clean_mask, scale_factor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: POP-UP WINDOWS OF WONDER
# Because the main window gets lonely.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MaskViewer(tk.Toplevel):
    """ A simple pop-up to show off the mask we made. "Behold, our creation!" """
    def __init__(self, parent, mask_image):
        super().__init__(parent)
        self.title("Full Detected Mask"); self.geometry("600x600")
        self.mask_image = mask_image
        self.mask_label = tk.Label(self); self.mask_label.pack(fill=tk.BOTH, expand=True)
        # This part makes sure the image resizes nicely, like a well-behaved child.
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event=None):
        # This is the logic for resizing the image without it looking like a funhouse mirror.
        container_w, container_h = self.winfo_width(), self.winfo_height()
        if container_w < 2 or container_h < 2: return # Don't bother if the window is microscopic.
        img_h, img_w = self.mask_image.shape
        scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(self.mask_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)

class ColorTunerWindow(tk.Toplevel):
    """ The Color DJ Booth! Twist the knobs, slide the faders, and find the perfect color vibe. """
    def __init__(self, parent, image_path, current_lower, current_upper, on_apply_callback):
        super().__init__(parent)
        self.title("Color Tuner"); self.geometry("800x700")
        self.on_apply_callback = on_apply_callback # This is how we phone home with the results.

        # Load the image and shrink it if it's a behemoth.
        img = cv2.imread(image_path); img_for_hsv = img.copy()
        if img.shape[1] > 1000:
            scale_percent = 50; width = int(img.shape[1] * scale_percent / 100); height = int(img.shape[0] * scale_percent / 100)
            img_for_hsv = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        self.hsv = cv2.cvtColor(img_for_hsv, cv2.COLOR_BGR2HSV)

        # Let's build the control panel.
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
        # The big "Apply" button and its nervous friend, "Cancel".
        tk.Button(btn_frame, text="Apply", command=self.apply_and_close).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        self.after(100, self.update_mask) # Give it a sec to wake up before the first update.

    def update_mask(self, val=None):
        # Every time a slider moves, we cook up a new mask and slap it on the screen.
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        mask = cv2.inRange(self.hsv, lower, upper)
        # A bunch of boilerplate to resize the preview nicely.
        container_w, container_h = self.mask_label.winfo_width(), self.mask_label.winfo_height()
        if container_w < 2 or container_h < 2: self.after(100, self.update_mask); return
        img_h, img_w = mask.shape; scale = min(container_w / img_w, container_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w > 0 and new_h > 0:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_pil = Image.fromarray(resized_mask); self.mask_photo = ImageTk.PhotoImage(image=img_pil)
            self.mask_label.config(image=self.mask_photo)

    def apply_and_close(self):
        # User is happy! Package up the chosen colors and send 'em back to the main app.
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()]); upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        self.on_apply_callback(lower, upper); self.destroy()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2.5: THE SLOPE-O-MATIC 5000 (INTERACTIVE SLOPE WINDOW)
# This is where we get REAL nerdy. We're gonna poke the path with a virtual
# stick (a tangent line) and see how steep it is at every point.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class InteractiveSlopeWindow(tk.Toplevel):
    def __init__(self, parent, original_cv_image, path_points, tck, u_params):
        super().__init__(parent)
        self.title("Interactive Slope Analysis (Side-by-Side)"); self.geometry("1300x700")
        self.original_cv_image = original_cv_image
        self.path_points, self.tck, self.u_params = path_points, tck, u_params
        
        # Is this path more of a "tall and skinny" or a "short and fat" situation?
        # This decides if we should rotate the math plot to see it better.
        x_range = np.ptp(self.path_points[:, 0]); y_range = np.ptp(self.path_points[:, 1])
        self.is_math_plot_rotated = y_range > x_range

        # Assembling the two-panel extravaganza.
        plots_frame = tk.Frame(self); plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        controls_frame = tk.Frame(self, pady=10); controls_frame.pack(fill=tk.X, expand=False, padx=10)
        plot_paned_window = PanedWindow(plots_frame, orient=tk.HORIZONTAL); plot_paned_window.pack(fill=tk.BOTH, expand=True)
        img_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(img_plot_frame, weight=1)
        self.fig_img = Figure(); self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_plot_frame); self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        math_plot_frame = tk.Frame(plot_paned_window); plot_paned_window.add(math_plot_frame, weight=1)
        self.fig_math = Figure(); self.ax_math = self.fig_math.add_subplot(111)
        self.canvas_math = FigureCanvasTkAgg(self.fig_math, master=math_plot_frame); self.canvas_math.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # The data readouts for our slope-o-matic.
        data_frame = tk.Frame(controls_frame); data_frame.pack(side=tk.LEFT, padx=(0, 20))
        self.slope_var = tk.StringVar(value="Slope: -"); self.int_slope_var = tk.StringVar(value="Integer Slope: -"); self.angle_var = tk.StringVar(value="Angle: -")
        tk.Label(data_frame, textvariable=self.slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.int_slope_var, font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(data_frame, textvariable=self.angle_var, font=("Helvetica", 11)).pack(anchor="w")
        
        # The big slider that lets you surf along the path.
        slider_frame = tk.Frame(controls_frame); slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.slider = tk.Scale(slider_frame, from_=0, to=len(self.u_params) - 1, orient=tk.HORIZONTAL, command=self._update_plots, label="Position Along Path"); self.slider.pack(fill=tk.X, expand=True)
        
        self._draw_initial_plots()
        # Start the slider in the middle, because why not?
        self.after(100, lambda: self.slider.set(len(self.path_points) // 2))

    def _draw_initial_plots(self):
        # Left panel: The original image with our path drawn on top. Pretty!
        img_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(img_rgb, aspect='equal')
        self.ax_img.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2.5, alpha=0.8)
        self.point_on_img, = self.ax_img.plot([], [], 'o', color='magenta', markersize=12, markeredgecolor='black', zorder=10)
        self.tangent_on_img, = self.ax_img.plot([], [], color='yellow', linestyle='--', linewidth=2.5, zorder=9)
        self.ax_img.set_title("Image Overlay View"); self.ax_img.axis('off'); self.fig_img.tight_layout()

        # Right panel: Just the path, plotted like a real math graph. No distractions.
        if self.is_math_plot_rotated:
            self.ax_math.plot(self.path_points[:, 1], self.path_points[:, 0], color='red', linewidth=2)
            self.ax_math.set_xlabel("Primary Axis (Y pixels)"); self.ax_math.set_ylabel("Secondary Axis (X pixels)")
        else:
            self.ax_math.plot(self.path_points[:, 0], self.path_points[:, 1], color='red', linewidth=2)
            self.ax_math.set_xlabel("Primary Axis (X pixels)"); self.ax_math.set_ylabel("Secondary Axis (Y pixels)")
        self.point_on_math, = self.ax_math.plot([], [], 'o', color='magenta', markersize=10, markeredgecolor='black', zorder=10)
        self.tangent_on_math, = self.ax_math.plot([], [], color='blue', linestyle='--', linewidth=2, zorder=9)
        self.ax_math.set_aspect('equal', adjustable='box'); self.ax_math.invert_yaxis(); self.ax_math.grid(True, linestyle='--', alpha=0.6)
        self.ax_math.set_title("Mathematical View"); self.fig_math.tight_layout()
        self.canvas_img.draw(); self.canvas_math.draw()

    def _update_plots(self, val):
        # The slider moved! Time to do some math and update everything.
        index = int(val); u_val = self.u_params[index]
        # Ask our spline for the point and its derivative (the direction it's heading).
        current_point = splev(u_val, self.tck, der=0)
        dx_dt, dy_dt = splev(u_val, self.tck, der=1)
        cx, cy = current_point[0], current_point[1]
        
        # ~~~ START: NEW FIX FOR ANGLE DIRECTION (The Great Derivative Alignment) ~~~
        # Sometimes the derivative vector points "backwards" along the path, which messes up our angle.
        # We had a stern talk with it and now we force it to always point "forwards"
        # relative to the main direction of the path. This makes the angles behave.
        if self.is_math_plot_rotated: # If path is mostly vertical...
            if dy_dt < 0: dx_dt, dy_dt = -dx_dt, -dy_dt #... "forward" means Y is increasing.
        else: # If path is mostly horizontal...
            if dx_dt < 0: dx_dt, dy_dt = -dx_dt, -dy_dt #... "forward" means X is increasing.
        # ~~~ END: NEW FIX FOR ANGLE DIRECTION (Peace in our time) ~~~

        # Figure out which derivative is which, depending on if we rotated the plot.
        if self.is_math_plot_rotated:
            primary_deriv, secondary_deriv = dy_dt, dx_dt
            slope_label_text = "Slope (dX/dY)"
        else:
            primary_deriv, secondary_deriv = dx_dt, dy_dt
            slope_label_text = "Slope (dY/dX)"
        
        # Calculate the angle using some good old-fashioned trigonometry (atan2 is our friend).
        angle_deg = np.degrees(np.arctan2(secondary_deriv, primary_deriv))

        # Calculate slope, but watch out for division by zero! That's how you get black holes.
        if abs(primary_deriv) < 1e-6:
            self.slope_var.set(f"{slope_label_text}: Infinity (It's vertical!)")
            self.int_slope_var.set("Integer Slope: N/A")
        else:
            slope_val = secondary_deriv / primary_deriv
            self.slope_var.set(f"{slope_label_text}: {slope_val:.2f}")
            self.int_slope_var.set(f"Integer Slope: {int(round(slope_val))}")
        
        self.angle_var.set(f"Angle: {angle_deg:.1f}°")
        
        # Now, calculate the coordinates for our little tangent line visualization.
        magnitude = np.hypot(dx_dt, dy_dt)
        line_length = 50 # How long should our tangent stick be? 50 pixels sounds good.
        if magnitude > 1e-6:
            ux, uy = dx_dt / magnitude, dy_dt / magnitude
            x_coords = [cx - ux * line_length, cx + ux * line_length]; y_coords = [cy - uy * line_length, cy + uy * line_length]
        else: x_coords, y_coords = [], [] # If magnitude is zero, don't draw anything.
            
        # Update all the little moving parts on our plots.
        self.point_on_img.set_data([cx], [cy]); self.tangent_on_img.set_data(x_coords, y_coords)
        if self.is_math_plot_rotated:
            self.point_on_math.set_data([cy], [cx]); self.tangent_on_math.set_data(y_coords, x_coords)
        else:
            self.point_on_math.set_data([cx], [cy]); self.tangent_on_math.set_data(x_coords, y_coords)
        # Tell the canvases, "Hey, wake up and redraw yourselves!"
        self.canvas_img.draw_idle(); self.canvas_math.draw_idle()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: THE MAIN ATTRACTION (THE APP ITSELF)
# This is the ringmaster, the conductor, the big boss. It brings everything
# together into one glorious, chaotic application.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PathAnalyzerApp:
    def __init__(self, root):
        self.root = root; self.root.title("Path Analyzer 9000"); self.root.geometry("1200x800")
        # Here we prepare all our variables. It's like laying out your tools before a big project.
        # Or like a squirrel hoarding nuts for the winter. Lots and lots of `None` nuts.
        self.image_path = None; self.original_cv_img = None; self.final_mask = None
        self.raw_skeleton_points = None; self.all_ordered_points = None; self.active_ordered_points = None
        self.displayed_nodes = None; self.smooth_path_points = None; self.start_node = None
        self.tck = None; self.u_params = None # These two hold the magical spline formula.
        self.hsv_lower = np.array([0, 70, 50]); self.hsv_upper = np.array([179, 255, 255]) # Default color range.
        self.selection_mode = "none"; self.manual_start_point = None; self.manual_end_point = None
        self.click_handler_id = None; self.motion_handler_id = None # To keep track of our event spies.
        self.hovered_node_marker = None; self.currently_hovered_node_index = None
        self._build_gui() # Let's build this thing!

    def _build_gui(self):
        """ Assembles the user interface. It's like LEGOs for programmers. """
        # Set up the main window layout with a draggable divider. Fancy!
        self.main_paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)
        self.left_pane = tk.Frame(self.main_paned_window); self.main_paned_window.add(self.left_pane, weight=3) # The big image panel
        self.right_paned_window = PanedWindow(self.main_paned_window, orient=tk.VERTICAL); self.main_paned_window.add(self.right_paned_window, weight=1) # The controls panel
        
        # The main canvas where we show the pretty pictures.
        self.fig = Figure(); self.ax = self.fig.add_subplot(111); self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # The top part of the right panel: all our buttons and sliders.
        controls_frame = tk.Frame(self.right_paned_window)
        # The bottom part: the little math plot.
        math_plot_frame = tk.Frame(self.right_paned_window)
        self.right_paned_window.add(controls_frame, weight=0)
        self.right_paned_window.add(math_plot_frame, weight=1)

        # The main workflow buttons. Numbered for your convenience!
        workflow_frame = tk.LabelFrame(controls_frame, text="Workflow Steps", padx=10, pady=10)
        workflow_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        self.btn_load = tk.Button(workflow_frame, text="1. Load Image", command=self.load_image); self.btn_load.pack(fill=tk.X, pady=2)
        self.btn_tuner = tk.Button(workflow_frame, text="2. Tune Color Range...", command=self.open_color_tuner, state=tk.DISABLED); self.btn_tuner.pack(fill=tk.X, pady=2)
        self.btn_analyze = tk.Button(workflow_frame, text="3. Analyze & Select Endpoints", command=self.analyze_path, state=tk.DISABLED); self.btn_analyze.pack(fill=tk.X, pady=(2, 0))

        # Sliders for fiddling with the path after it's found.
        tuning_frame = tk.LabelFrame(controls_frame, text="Path Visualization Tuning", padx=10, pady=5)
        tuning_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(tuning_frame, text="Path Endpoint Control (Fine-tuning):").pack()
        self.path_length_var = tk.IntVar(value=100)
        self.path_length_slider = tk.Scale(tuning_frame, from_=2, to=100, orient=tk.HORIZONTAL, variable=self.path_length_var, command=self._update_path_length, state=tk.DISABLED, showvalue=0); self.path_length_slider.pack(fill=tk.X)
        tk.Label(tuning_frame, text="Number of Nodes to Display:").pack()
        self.nodes_var = tk.IntVar(value=150)
        self.nodes_slider = tk.Scale(tuning_frame, from_=2, to=1000, orient=tk.HORIZONTAL, variable=self.nodes_var, command=self.update_node_display_during_selection, showvalue=0); self.nodes_slider.pack(fill=tk.X)
        
        # The grand finale: exporting the data.
        export_frame = tk.LabelFrame(controls_frame, text="Export for Vehicle Control", padx=10, pady=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(export_frame, text="Export Path Resolution (Number of Steps):").pack()
        self.export_nodes_var = tk.IntVar(value=100)
        self.export_nodes_slider = tk.Scale(export_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.export_nodes_var, state=tk.DISABLED); self.export_nodes_slider.pack(fill=tk.X, pady=(0, 5))
        self.btn_export = tk.Button(export_frame, text="4. Export Path to CSV", command=self.export_path_data, state=tk.DISABLED, bg="#ccffcc"); self.btn_export.pack(fill=tk.X)
        
        # Extra tools for the curious user.
        tools_frame = tk.LabelFrame(controls_frame, text="Diagnostic Tools", padx=10, pady=5)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_show_mask = tk.Button(tools_frame, text="Show Full Mask", command=self.show_full_mask, state=tk.DISABLED); self.btn_show_mask.pack(fill=tk.X, pady=2)
        self.btn_slope = tk.Button(tools_frame, text="Interactive Slope Analysis", command=self.open_slope_visualizer, state=tk.DISABLED); self.btn_slope.pack(fill=tk.X, pady=2)
        
        # A little status bar to tell the user what's happening (or what they should do next).
        self.status_label = tk.Label(controls_frame, text="Please load an image to begin.", wraplength=350, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)
        
        # The mini math plot for the control panel.
        tk.Label(math_plot_frame, text="Mathematical 2D Plot").pack()
        self.math_fig = Figure(); self.math_ax = self.math_fig.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_fig, master=math_plot_frame); self.math_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

    def open_slope_visualizer(self):
        """ Opens the Slope-O-Matic 5000 window. """
        if self.smooth_path_points is None or self.tck is None: messagebox.showinfo("Info", "No path analyzed yet. Can't slope what doesn't exist!"); return
        InteractiveSlopeWindow(self.root, self.original_cv_img, self.smooth_path_points, self.tck, self.u_params)

    def _disconnect_event_handlers(self):
        """ Kicks our event spies off the canvas so they stop listening to clicks and motions. """
        if self.click_handler_id: self.canvas.mpl_disconnect(self.click_handler_id); self.click_handler_id = None
        if self.motion_handler_id: self.canvas.mpl_disconnect(self.motion_handler_id); self.motion_handler_id = None
        self._remove_hover_marker()

    def load_image(self):
        """ Step 1: Find an image. Any image. """
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path: return # User changed their mind. It happens.
        self.image_path = path; self.original_cv_img = cv2.imread(self.image_path)
        if self.original_cv_img is None: messagebox.showerror("Error", "Failed to load image. Maybe it's shy?"); return
        # Reset everything from the previous run. A clean slate!
        self.raw_skeleton_points = None; self.all_ordered_points = None; self.active_ordered_points = None
        self.display_image(self.original_cv_img, "Original Image Loaded")
        self.update_math_plot()
        # Enable the next step button.
        self.btn_tuner.config(state=tk.NORMAL); self.btn_analyze.config(state=tk.DISABLED)
        self.btn_show_mask.config(state=tk.DISABLED); self.btn_slope.config(state=tk.DISABLED)
        self.path_length_slider.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
        self._disconnect_event_handlers()
        self.status_label.config(text="Image loaded. Now, let's play with some colors in the tuner.")

    def analyze_path(self):
        """ The magic button! This unleashes the full power of our image processing pipeline. """
        self.status_label.config(text="Unleashing the algorithms... please stand back."); self.root.update_idletasks()
        self.btn_analyze.config(state=tk.DISABLED) # Don't let them click it again while it's working.
        
        self.original_cv_img, raw_points, self.final_mask, scale_factor = preprocess_and_extract_path(self.image_path, self.hsv_lower, self.hsv_upper)
        
        if raw_points is None or len(raw_points) < 20: # We need a reasonable number of points.
            messagebox.showerror("Error", "Could not find a decent path. Try tuning the colors again?"); self.status_label.config(text="Analysis failed. The image remains mysterious.")
            self.btn_analyze.config(state=tk.NORMAL); return
        
        # Remember to scale the points back up if we used the shrink ray earlier!
        self.raw_skeleton_points = (raw_points / scale_factor).astype(int)
        self.smooth_path_points = None
        self.update_node_display_during_selection()
        
        # Now, we enter manual selection mode.
        self.selection_mode = "start"; self.manual_start_point, self.manual_end_point = None, None
        # Deploy the event spies!
        self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.motion_handler_id = self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.status_label.config(text="Path detected! Now, please click on the node where the path should START.")

    def _on_canvas_motion(self, event):
        """ Our motion spy reports back whenever the mouse moves over the canvas. """
        if event.inaxes != self.ax or self.displayed_nodes is None or len(self.displayed_nodes) == 0: self._remove_hover_marker(); return
        mouse_point = np.array([event.xdata, event.ydata])
        # Find the node closest to the mouse pointer.
        distances = np.linalg.norm(self.displayed_nodes - mouse_point, axis=1)
        closest_node_index = np.argmin(distances); min_dist = distances[closest_node_index]
        
        hover_threshold = 15 # How close does the mouse have to be to "snap" to a node?
        if min_dist < hover_threshold:
            # If we're hovering over a *new* node, update the yellow highlight circle.
            if self.currently_hovered_node_index != closest_node_index:
                self._remove_hover_marker()
                node_coords = self.displayed_nodes[closest_node_index]
                self.hovered_node_marker, = self.ax.plot(node_coords[0], node_coords[1], 'o', markersize=18, color='yellow', alpha=0.7, zorder=10)
                self.currently_hovered_node_index = closest_node_index; self.canvas.draw_idle()
        else:
            # Mouse is out in the middle of nowhere. Remove the highlight.
            self._remove_hover_marker()
            
    def _remove_hover_marker(self):
        """ Makes the big yellow hover circle disappear. Poof! """
        if self.hovered_node_marker:
            self.hovered_node_marker.remove(); self.hovered_node_marker = None
            self.currently_hovered_node_index = None; self.canvas.draw_idle()
            
    def _on_canvas_click(self, event):
        """ A click! A palpable click! The user has chosen a node. """
        if self.selection_mode == "none" or self.currently_hovered_node_index is None: return
        selected_node = self.displayed_nodes[self.currently_hovered_node_index]
        
        if self.selection_mode == "start":
            self.manual_start_point = selected_node; self.selection_mode = "end"
            self.status_label.config(text="Start node locked in! Now, click on the END node.")
        elif self.selection_mode == "end":
            self.manual_end_point = selected_node; self.selection_mode = "none"
            self._disconnect_event_handlers() # Recall the spies, their mission is complete.
            self.status_label.config(text="Endpoints selected. Crunching the numbers...")
            # We use `after` to give the GUI a moment to breathe before the next big calculation.
            self.root.after(50, self._process_manual_endpoints)
            
    def _process_manual_endpoints(self):
        """ Now that we have a start and end, let's put all the points in order. """
        self.status_label.config(text="Playing connect-the-dots from start to end..."); self.root.update_idletasks()
        self.all_ordered_points = self.order_points_by_proximity(self.raw_skeleton_points, start_point=self.manual_start_point, end_point=self.manual_end_point)
        self.btn_show_mask.config(state=tk.NORMAL)
        # Configure the path length slider based on how many points we found.
        num_total_points = len(self.all_ordered_points)
        self.path_length_slider.config(to=num_total_points, state=tk.NORMAL); self.path_length_var.set(num_total_points)
        self._update_path_length()
        self.status_label.config(text="Path analysis complete! Feel free to tweak the sliders or export."); self.btn_analyze.config(state=tk.NORMAL)

    def order_points_by_proximity(self, points, start_point, end_point):
        """ A greedy algorithm for ordering points. It's simple but surprisingly effective. """
        # It's like a very focused game of connect-the-dots.
        # Start at the start point, find the nearest neighbor, jump to it, repeat until you hit the end.
        if len(points) < 2: return points
        points_list = points.tolist() # Working with lists is easier for popping.
        ordered_points = []
        
        # Find the actual point in our list that's closest to the user's click.
        dists = np.linalg.norm(np.array(points_list) - np.array(start_point), axis=1)
        start_index = np.argmin(dists)
        current_point = points_list.pop(start_index)
        ordered_points.append(current_point)
        
        # Just in case the start and end are the same point (unlikely, but hey).
        if np.array_equal(current_point, end_point): return np.array(ordered_points)
        
        # Let the point-hopping begin!
        while points_list:
            distances_sq = np.sum((np.array(points_list) - current_point)**2, axis=1)
            closest_index = np.argmin(distances_sq)
            current_point = points_list.pop(closest_index)
            ordered_points.append(current_point)
            # If we've arrived at our destination, we can stop.
            if np.array_equal(current_point, end_point): break
        return np.array(ordered_points)
        
    def _update_path_length(self, val=None):
        """ Called when the path length slider is moved. Time to smooth things over. """
        if self.all_ordered_points is None: return
        length = self.path_length_var.get()
        self.active_ordered_points = self.all_ordered_points[:length]

        if len(self.active_ordered_points) < 4: # Not enough points for fancy math.
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
            self.update_node_display_during_selection(); return
        
        # A little pre-smoothing with a moving average. It's like giving the data a nice massage.
        window_size = min(21, len(self.active_ordered_points) -1 if len(self.active_ordered_points) % 2 == 0 else len(self.active_ordered_points) -2)
        if window_size < 3: window_size = 3
        points_for_spline = self.active_ordered_points
        if len(self.active_ordered_points) >= window_size:
            weights = np.repeat(1.0, window_size) / window_size
            x_smooth = np.convolve(self.active_ordered_points[:, 0], weights, 'valid'); y_smooth = np.convolve(self.active_ordered_points[:, 1], weights, 'valid')
            points_for_spline = np.column_stack((x_smooth, y_smooth))
            
        try:
            # Now for the main event: fitting a B-spline to our points.
            # This is what turns our jagged, pixelated path into a beautiful, smooth curve.
            if len(points_for_spline) < 4: raise ValueError("Not enough points for spline fitting.")
            x, y = points_for_spline[:, 0], points_for_spline[:, 1]
            smoothing_factor = len(points_for_spline) * 0.1
            self.tck, u = splprep([x, y], s=smoothing_factor, k=3)
            
            if self.tck:
                # If it worked, generate a bunch of points along our new smooth curve.
                num_smooth_points = len(self.active_ordered_points) * 2; self.u_params = np.linspace(u.min(), u.max(), num_smooth_points)
                x_final, y_final = splev(self.u_params, self.tck); self.smooth_path_points = np.column_stack((x_final, y_final))
                # Enable the cool tools now that we have a smooth path!
                self.btn_slope.config(state=tk.NORMAL)
                self.btn_export.config(state=tk.NORMAL); self.export_nodes_slider.config(state=tk.NORMAL)
            else: raise ValueError("Spline creation failed. The points were uncooperative.")
        except Exception:
            # Sometimes the math just doesn't work out. It's okay. We'll live.
            self.smooth_path_points = None; self.tck = None; self.btn_slope.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED); self.export_nodes_slider.config(state=tk.DISABLED)
        
        self.update_node_display_during_selection()

    def update_node_display_during_selection(self, val=None):
        """ Decides which blue dots to show on the screen based on the slider. """
        # We don't want to show ALL the points, that would be a mess.
        # So we pick a nice, evenly spaced subset of them to display as nodes.
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
            # The start node is always the very first one in our ordered list.
            self.start_node = self.active_ordered_points[0] if self.active_ordered_points is not None else None
        
        # Redraw everything with the new set of nodes.
        self.display_graph(); self.update_math_plot()

    def display_graph(self):
        """ The main drawing routine for the big image panel. """
        self.ax.clear(); img_rgb = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal')
        show_legend = self.smooth_path_points is not None
        # Draw the smooth red line if we have one.
        if self.smooth_path_points is not None: self.ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2.5, label='Fitted Smooth Path', zorder=4)
        # Draw the cyan nodes.
        if self.displayed_nodes is not None: self.ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=35, zorder=5, edgecolors='black', label='Detected Nodes')
        # Draw the big green start node.
        if self.start_node is not None: self.ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=150, zorder=6, edgecolors='black', label='Start Node')
        if show_legend: self.ax.legend()
        self.ax.set_title("Path Node Overlay"); self.ax.axis('off'); self.canvas.draw()
        
    def update_math_plot(self):
        """ The drawing routine for the little math plot on the side. """
        self.math_ax.clear(); self.math_ax.set_title("Mathematical 2D Plot"); self.math_ax.grid(True, linestyle='--', alpha=0.6)
        if self.smooth_path_points is None or len(self.smooth_path_points) == 0: self.math_canvas.draw(); return
        
        # Again, we check if the path is tall or wide to decide on rotation.
        x_range_data = np.ptp(self.smooth_path_points[:, 0]); y_range_data = np.ptp(self.smooth_path_points[:, 1]); is_rotated = y_range_data > x_range_data
        
        if is_rotated:
            self.math_ax.plot(self.smooth_path_points[:, 1], self.smooth_path_points[:, 0], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 1], self.displayed_nodes[:, 0], c='cyan', s=20, ec='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[1], self.start_node[0], c='lime', s=100, label='Start Node', ec='black')
            self.math_ax.set_xlabel("Primary Axis (pixels)"); self.math_ax.set_ylabel("Secondary Axis (pixels)")
        else: # The normal, un-rotated case.
            self.math_ax.plot(self.smooth_path_points[:, 0], self.smooth_path_points[:, 1], color='red', linewidth=2, label='Fitted Smooth Path')
            if self.displayed_nodes is not None: self.math_ax.scatter(self.displayed_nodes[:, 0], self.displayed_nodes[:, 1], c='cyan', s=20, ec='black', label='Detected Nodes')
            if self.start_node is not None: self.math_ax.scatter(self.start_node[0], self.start_node[1], c='lime', s=100, label='Start Node', ec='black')
            self.math_ax.set_xlabel("Primary Axis (pixels)"); self.math_ax.set_ylabel("Secondary Axis (pixels)")
            
        self.math_ax.set_aspect('equal', adjustable='box'); self.math_ax.invert_yaxis() # Y usually points down in images.
        self.math_ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small') # Put legend outside the plot.
        self.math_fig.subplots_adjust(right=0.75)
        self.math_canvas.draw()

    def on_tuner_apply(self, new_lower, new_upper):
        """ The color tuner window is calling home to give us the new color values. """
        self.hsv_lower, self.hsv_upper = new_lower, new_upper
        self.status_label.config(text="New color range applied. You may now Analyze the path.")
        self.btn_analyze.config(state=tk.NORMAL)

    def open_color_tuner(self):
        """ Opens the Color DJ Booth. """
        if self.image_path: ColorTunerWindow(self.root, self.image_path, self.hsv_lower, self.hsv_upper, self.on_tuner_apply)

    def display_image(self, cv_img, title=""):
        """ A simple helper to show an image on the main canvas. """
        self.ax.clear(); img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); self.ax.imshow(img_rgb, aspect='equal'); self.ax.set_title(title); self.ax.axis('off'); self.canvas.draw()
    
    def show_full_mask(self):
        """ Let's see that beautiful mask we made. """
        if self.final_mask is not None: MaskViewer(self.root, self.final_mask)
        else: messagebox.showinfo("Info", "No mask has been generated yet. Patience, young grasshopper.")

    def export_path_data(self):
        """ The grand finale! Let's save our hard work to a CSV file. """
        if self.tck is None or self.u_params is None:
            messagebox.showerror("Error", "No valid smooth path to export. Did you skip a step?"); return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")], title="Save Path Data")
        if not file_path: return # User chickened out.
        
        # Get the desired number of points from the export slider.
        num_points = self.export_nodes_var.get()
        u_vals_export = np.linspace(self.u_params.min(), self.u_params.max(), num_points)
        points = np.column_stack(splev(u_vals_export, self.tck))
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'y', 'distance_segment', 'turning_angle_deg'])
                prev_heading_rad = 0.0
                # Go through each point and calculate how far it is from the last one,
                # and how sharp of a turn it represents. Very useful for robots!
                for i in range(len(points)):
                    x, y = points[i]
                    if i == 0:
                        dist_segment, turn_angle_deg = 0.0, 0.0
                        if len(points) > 1: # Figure out the initial direction.
                            dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
                            prev_heading_rad = np.arctan2(dy, dx)
                    else:
                        prev_x, prev_y = points[i-1]; dx, dy = x - prev_x, y - prev_y
                        dist_segment = np.hypot(dx, dy)
                        current_heading_rad = np.arctan2(dy, dx) if (dx**2 + dy**2) > 1e-12 else prev_heading_rad
                        # This bit of math wizardry calculates the angle change correctly, even when crossing 180 degrees.
                        turn_angle_rad = current_heading_rad - prev_heading_rad
                        turn_angle_rad = (turn_angle_rad + np.pi) % (2 * np.pi) - np.pi
                        turn_angle_deg = np.degrees(turn_angle_rad)
                        prev_heading_rad = current_heading_rad
                    writer.writerow([f"{x:.2f}", f"{y:.2f}", f"{dist_segment:.3f}", f"{turn_angle_deg:.3f}"])
            messagebox.showinfo("Success", f"Path data with {num_points} steps successfully exported to:\n{file_path}")
        except Exception as e: messagebox.showerror("Error", f"Failed to write to file. The computer says no: {e}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: THE IGNITION SWITCH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # The sacred incantation that brings our application to life.
    # Let there be windows!
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()
