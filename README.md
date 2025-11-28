# CV Document Scanner

This project is a full-stack web application that provides a powerful computer vision pipeline to automatically scan, clean, and extract text from images of documents. It features a React-based frontend for user interaction and a Python Flask backend that performs the heavy lifting with OpenCV and AI-based models.

## Features

- **Automatic Skew Correction**: Straightens documents that are slightly tilted in the photo.
- **Perspective Warping**: Transforms the detected document into a flat, top-down "scanned" view.
- **AI-Powered Super-Resolution**: Uses an FSRCNN model to enhance the image resolution, improving text clarity.
- **Advanced Binarization**: Employs a multi-step process including contrast enhancement (CLAHE) and adaptive thresholding to create a clean, high-contrast image for OCR.
- **Optical Character Recognition (OCR)**: Extracts text from the processed image using Tesseract.
- **Moiré Noise Removal**: An optional feature that uses a Fast Fourier Transform (FFT) based low-pass filter to remove periodic noise patterns (moiré) often seen in photos of digital screens.
- **Detailed Metrics**: The frontend displays quality metrics for both the image preprocessing steps and the OCR confidence, giving insight into the pipeline's performance.
- **Interactive Web Interface**: A user-friendly UI to upload images, toggle features, and view the step-by-step output of the vision pipeline.

## Technology Stack

| Component | Technologies Used                                       |
| :-------- | :------------------------------------------------------ |
| **Backend**  | Python, Flask, OpenCV, NumPy, Pytesseract             |
| **Frontend** | React, Vite, Tailwind CSS, Axios, lucide-react          |
| **AI Model** | FSRCNN (Fast Super-Resolution Convolutional Neural Network) for 4x upscaling. |

## Project Structure

The project is organized into two main directories:

-   `frontend/`: Contains the React application, providing the user interface.
-   `backend/`: Contains the Flask server, which exposes the image processing API.

```
.
├── backend/
│   ├── FSRCNN_x4.pb        # Super-resolution model
│   ├── requirements.txt    # Python dependencies
│   └── server.py           # Flask server with the CV pipeline
└── frontend/
    ├── src/
    │   └── App.jsx         # Main React component
    ├── package.json        # Node.js dependencies
    └── ...                 # Other React project files
```

## Backend CV Pipeline Explained

The core of this project is the multi-stage computer vision pipeline in `backend/server.py`. When an image is uploaded, it goes through the following stages:

1.  **Skew Correction**:
    -   **What it does**: Detects the overall orientation of the document in the image and rotates it to be perfectly horizontal.
    -   **How it works**: It uses Principal Component Analysis (PCA) on the distribution of edge pixels found by a Canny edge detector. The principal axis gives the document's tilt angle, which is then corrected.

2.  **Moiré Removal (Optional)**:
    -   **What it does**: Removes periodic noise patterns, which often appear when taking a picture of a digital screen. These patterns can severely degrade OCR quality.
    -   **How it works**: The image is converted to the frequency domain using a 2D Fast Fourier Transform (FFT). A Gaussian low-pass filter is applied to attenuate the high-frequency noise. The result is then transformed back to the spatial domain and blended with the original to preserve text sharpness. This feature is controlled by a checkbox in the UI.

3.  **Document Detection**:
    -   **What it does**: Finds the four corners of the document in the image.
    -   **How it works**: The image is preprocessed (grayscale, bilateral filter) and Canny edge detection is applied. It then finds all contours, sorts them by size, and identifies the largest four-sided contour, assuming it is the document. A fallback to the full image is used if no suitable contour is found.

4.  **Perspective Transform**:
    -   **What it does**: "Scans" the document by warping the four detected corner points into a flat, rectangular image.
    -   **How it works**: Using the four corners from the previous step, it computes a perspective transform matrix and applies it to the image, resulting in a top-down view.

5.  **AI Enhancement (Super-Resolution)**:
    -   **What it does**: Increases the resolution of the warped document image by 4x. This is crucial for improving the readability of small text before OCR.
    -   **How it works**: It uses a pre-trained FSRCNN model loaded via OpenCV's `dnn_superres` module. The system attempts to use a CUDA-enabled GPU if available in the OpenCV build but defaults to CPU-based processing.

6.  **Binarization for OCR**:
    -   **What it does**: Converts the enhanced image into a pure black-and-white format, which is optimal for OCR.
    -   **How it works**: This is a careful, multi-step process:
        1.  A bilateral filter smooths the image while preserving edges.
        2.  Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances local contrast.
        3.  An unsharp mask sharpens the text.
        4.  Finally, adaptive Gaussian thresholding is used to create the final binary image, handling variations in lighting.

7.  **OCR (Optical Character Recognition)**:
    -   **What it does**: Extracts the text from the final binary image.
    -   **How it works**: It uses the Tesseract OCR engine (`pytesseract` wrapper) with Page Segmentation Mode 6 (PSM 6), which assumes a single uniform block of text, ideal for scanned documents.

## Setup and Installation

### Prerequisites

1.  **Python 3.8+**: Make sure Python and pip are installed.
2.  **Node.js and npm**: Required for the frontend.
3.  **Tesseract OCR Engine**: This must be installed on your system.
    -   **Windows**: Download and run the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). During installation, make sure to note the installation path.
    -   **macOS**: `brew install tesseract`
    -   **Linux (Ubuntu)**: `sudo apt-get install tesseract-ocr`

### Backend Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```

2.  **Create a virtual environment and activate it** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Tesseract Path (Windows Only)**:
    The backend needs to know where Tesseract is installed. By default, it looks in `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you installed it elsewhere, set the `TESSERACT_CMD` environment variable.
    ```powershell
    # Example:
    $env:TESSERACT_CMD = "C:\path\to\your\tesseract.exe"
    ```

5.  **Run the backend server**:
    ```bash
    python server.py
    ```
    The server will start on `http://127.0.0.1:5000`.

### Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```

3.  **Run the frontend development server**:
    ```bash
    npm run dev
    ```
    The React app will be available at `http://localhost:5173` (or another port if 5173 is busy).

## How to Use

1.  Ensure both the backend and frontend servers are running.
2.  Open your browser and go to the frontend URL (e.g., `http://localhost:5173`).
3.  Click the upload area to select an image of a document.
4.  A preview of the image will appear.
5.  (Optional) If the image has moiré patterns, check the **"Fix Moiré Noise (FFT)"** box. You can adjust the `Cutoff` slider to control the filter's strength (lower is stronger).
6.  Click the **"Run Pipeline"** button.
7.  Wait for the processing to complete. The UI will then display the output of each stage of the pipeline, the final extracted text, and performance metrics.
