# AutoFarm Track Detector

A full-stack computer vision application designed to automatically detect and segment farming equipment tracks and planter skips in high-resolution aerial agricultural imagery. 

---

## Part 1: High-Level Overview

### Objective
Modern agriculture relies heavily on aerial imagery to monitor crop health. However, anomalous patterns like tractor tracks, planter skips, and waterways can distort yield predictions. This project provides an automated, machine-learning-driven pipeline to identify and isolate these features.

### Features
*   **Instant Visualizer:** A beautiful, responsive React frontend that allows researchers to seamlessly stream, explore, and analyze high-resolution aerial datasets.
*   **Real-Time Inference:** Images are automatically predicted using a deeply trained PyTorch U-Net architecture as soon as they are loaded.
*   **Precision Metrics:** Ground truth masks are instantly compared against AI predictions, calculating live Intersection-over-Union (IoU) and F1 precision scores.
*   **Optimized Data Pipeline:** The backend maps directly into local `.arrow` binary caches, enabling it to scan over 1,000,000 satellite images and filter down to exact track anomalies in mere seconds.

### Tech Stack
*   **Frontend:** React 18, Vite, Lucide Icons, Vanilla CSS (Glassmorphism design).
*   **Backend:** Python, FastAPI, Uvicorn.
*   **AI / ML:** PyTorch, OpenCV, Segmentation Models (U-Net).
*   **Infrastructure:** Docker, Docker Compose, HuggingFace Datasets.

### Quick Start
To launch the entire stack:
```bash
docker compose up --build -d
```
Then open **[http://localhost](http://localhost)** in your browser.

---

## Part 2: Low-Level Technical Details

### 1. The Dataset & Storage Layer (`huggingface/datasets`)
The application utilizes the `shi-labs/Agriculture-Vision` dataset. To bypass significant network delays, offline hash-check crashes, and massive RAM overhead from the standard HuggingFace `load_dataset()` pipeline, the backend is engineered to parse the raw `.arrow` columnar files directly.
*   **Arrow Concatenation:** `app/server.py` uses python's `glob` and `Dataset.from_file()` to fuse 45 offline shards.
*   **Background Indexing:** A daemon thread asynchronously traverses the 1M+ rows on startup, ensuring the FastAPI endpoints remain unblocked. It purposefully targets indices `431,062` (RGB images) and `680,000` (Planter Skip Masks).

### 2. The Backend API (`app/server.py`)
Built on FastAPI, the backend acts as the orchestrator between the cached data and the PyTorch models.
*   **Empty Mask Filtering:** Because ~94.5% of the `planter_skip` ground-truth masks in the dataset are entirely black (containing zero anomalies), the background indexer uses `np.any()` to heavily filter the batch mapping. This guarantees that the UI only receives the ~67 specific images that contain visible tracks.
*   **`/batch` Endpoint:** Serves paginated data (10 images per request). Instead of requiring secondary requests for predictions, this endpoint internally `await` calls the PyTorch inference engine and packages the base64 thumbnails, masks, and metrics in a single unified JSON response.

### 3. The Inference Engine (`src/train.py` & UNet)
The core segmentation model is a custom `UNetFarmTrack` defined in PyTorch. 
*   **Architecture:** Standard Contracting/Expanding symmetric blocks with MaxPooling down-sampling and ConvTranspose2d up-sampling.
*   **Transforms:** Inputs are tensorized, resized to `512x512`, and normalized using `(0.5, 0.5, 0.5)` mean/std arrays.
*   **Metrics:** The backend calculates IoU via boolean intersection-over-union matrix operations: `sum(pred * gt) / [sum(pred) + sum(gt) - intersection]`.

### 4. The Frontend Pipeline (`frontend/src/App.jsx`)
The UI is a decoupled React single-page application heavily reliant on CSS Grid/Flexbox and dynamic styling.
*   **Status Polling:** A standard `useEffect` interval polls the `/status` endpoint to render a dynamic progress bar during the background Arrow indexing phase.
*   **CSS Rendering Architecture:** High-resolution RGB images are aggressively layered with `filter: blur()` base64 placeholders to prevent layout shifts. Ground Truth and Prediction masks are overlaid utilizing absolutely positioned `img` tags combined with `mix-blend-mode: screen`, enabling the neural network predictions to appear perfectly composited over the raw geography. Specificity overrides ensure CSS `opacity: 0` states explicitly clear via `onLoad` DOM events to prevent visual darkening bugs.
*   **Metrics Injection:** IoU scores are injected dynamically via JavaScript templating strings straight into the sidebar components for quick analysis.