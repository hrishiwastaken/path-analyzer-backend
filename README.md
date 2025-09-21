Path Analyzer 🚀
An interactive desktop GUI application built with Python and Tkinter for extracting, smoothing, and analyzing colored paths from images.
This tool allows users to load an image, isolate a specific color trail using an interactive HSV color tuner, and then apply a series of image processing and mathematical techniques to extract a smooth, analyzable path. The final output includes a clean spline curve, selectable nodes, and an interactive slope analysis tool.
✨ Features
Image Loading: Supports common image formats (.jpg, .png, .bmp, etc.).
Interactive Color Tuning: A real-time HSV color thresholding window to perfectly isolate the desired path color.
Path Extraction: Uses morphological operations and a skeletonization algorithm to find the centerline of the colored path.
Spline Curve Fitting: Smooths the jagged, pixel-based path into a continuous, differentiable B-spline curve for mathematical analysis.
Node Visualization: Overlays a configurable number of nodes along the detected path on the original image.
Interactive Slope Analysis: A dedicated window to scrub through the smoothed path, visualizing the point, its tangent line, and the precise slope and angle at that location.
Dual-View Analysis: Displays the path overlaid on the original image and as a clean, rotatable 2D mathematical plot.
📸 Screenshot
(Action Required: Replace this with a screenshot or, even better, a GIF of your application in action! A great tool for making GIFs is ScreenToGif)
Fig 1: The main application window showing an analyzed path with the control panel and mathematical plot.
🔧 Tech Stack
Language: Python 3
GUI: Tkinter (via Python's standard library)
Image Processing: OpenCV (opencv-python)
Numerical & Scientific Computing: NumPy, SciPy
Plotting: Matplotlib
Image I/O & Tkinter Integration: Pillow (PIL Fork)
⚙️ Installation & Setup
To run this application on your local machine, follow these steps.
Prerequisites:
Python 3.7+
pip (Python package installer)
Steps:
Clone the repository:
code
Bash
git clone https://github.com/your-username/path-analyzer.git
cd path-analyzer
Create and activate a virtual environment (recommended):
code
Bash
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
Install the required dependencies:
A requirements.txt file is provided for convenience.
code
Bash
pip install -r requirements.txt
(If you don't have a requirements.txt file yet, you can create one with pip freeze > requirements.txt after installing the modules below)
code
Bash
pip install opencv-python numpy scipy matplotlib Pillow
Run the application:
code
Bash
python your_script_name.py
(Replace your_script_name.py with the actual name of your Python file)
📖 How to Use
Load Image: Click the "Load Image" button to select an image file containing a colored path you want to analyze.
Tune Color Range: Click "Tune Color Range..." to open the tuner window. Adjust the HSV sliders until the preview window shows the path isolated in white and everything else in black. Click "Apply" to save the settings.
Analyze Path: Click "Analyze Path". The application will:
Create a binary mask using your color settings.
Clean up the mask and skeletonize it to a 1-pixel-wide line.
Order the pixels to form a path.
Fit a smooth B-spline curve to the path data.
Explore the Results:
Use the "Number of Nodes" slider to change how many points are visualized on the path.
Click "Show Full Mask" to see the raw binary image used for skeletonization.
Click "Interactive Slope Analysis" to open the powerful slope visualizer and inspect the curve's properties at any point.
📦 Building an Executable
You can package this application into a single .exe file for easy distribution on Windows, with no Python installation required on the target machine.
Install PyInstaller:
code
Bash
pip install pyinstaller
Run the build command:
Navigate to the project directory in your terminal and run the following command:
code
Bash
pyinstaller --name PathAnalyzer --windowed --onefile your_script_name.py
--name PathAnalyzer: Sets the name of your final executable.
--windowed: Prevents a console window from appearing behind your GUI.
--onefile: Bundles everything into a single .exe file.
Find your application:
The final PathAnalyzer.exe will be located in the dist folder.
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
