import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Traco, Image as ImageIcon, Loader2, Play, LayoutDashboard } from 'lucide-react';
import './App.css';

const API_BASE = "http://localhost:8000";

function App() {
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isInferencing, setIsInferencing] = useState(false);
  const [maskData, setMaskData] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // Fetch available test set images
    axios.get(`${API_BASE}/images`)
      .then(res => setImages(res.data.images))
      .catch(err => console.error("Could not fetch images:", err));
  }, []);

  const handleSelect = (img) => {
    setSelectedImage(img);
    setMaskData(null);
    setMetrics(null);
  };

  const handleInfer = async () => {
    if (!selectedImage) return;
    setIsInferencing(true);
    
    try {
      const resp = await axios.post(`${API_BASE}/infer/${selectedImage}`);
      setMaskData(`data:image/png;base64,${resp.data.mask_base64}`);
      setMetrics(resp.data.metrics);
    } catch (err) {
      console.error("Inference failed", err);
      alert("Failed to run inference.");
    } finally {
      setIsInferencing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header glass-panel">
        <h1><LayoutDashboard size={24} color="#58a6ff" /> FarmTrack Analytics</h1>
        <div style={{color: "var(--text-muted)", fontSize: "0.9rem"}}>
          Segmentation Engine: <strong>PyTorch U-Net (Mock)</strong>
        </div>
      </header>

      <main className="main-content">
        <aside className="gallery-sidebar glass-panel">
          <h2>Dataset Explorer</h2>
          <div style={{color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: "1rem"}}>
            Raw Field Imagery (data/raw/)
          </div>
          <div className="image-list">
            {images.map(img => (
              <div 
                key={img} 
                className={`image-item ${selectedImage === img ? 'active' : ''}`}
                onClick={() => handleSelect(img)}
              >
                <ImageIcon size={18} />
                <span style={{fontSize: "0.9rem", overflow: "hidden", textOverflow: "ellipsis"}}>{img}</span>
              </div>
            ))}
            {images.length === 0 && (
              <div style={{color: '#8b949e', fontStyle: 'italic', fontSize: '0.9rem'}}>No images loaded.</div>
            )}
          </div>
        </aside>

        <section className="visualizer-area">
          <div className="toolbar glass-panel">
            <div>
              {selectedImage ? (
                <span>Currently Viewing: <strong style={{color:"#fff"}}>{selectedImage}</strong></span>
              ) : (
                <span className="text-muted">Select an image to begin</span>
              )}
            </div>
            <button 
              className="btn-primary" 
              onClick={handleInfer}
              disabled={!selectedImage || isInferencing || maskData}
            >
              {isInferencing ? (
                <><Loader2 size={18} className="spin" /> Processing...</>
              ) : (
                <><Play size={18} /> Detect Tracks</>
              )}
            </button>
          </div>

          <div className="image-display glass-panel">
            {selectedImage ? (
              <div className="image-container">
                {/* Underlay Image */}
                <img src={`${API_BASE}/image/${selectedImage}`} alt="Raw Field" />
                {/* Overlay Mask */}
                {maskData && (
                  <img 
                    src={maskData} 
                    alt="Segmentation Mask Overlay" 
                    className={`mask-overlay visible`} 
                  />
                )}
              </div>
            ) : (
              <div className="empty-state">
                <ImageIcon size={48} opacity={0.3} />
                <p>Select an image from the dataset explorer to visualize</p>
              </div>
            )}
          </div>

          <div className="metrics-panel">
            <div className="metric-card glass-panel">
              <span className="metric-label">Mean IoU</span>
              <span className="metric-value">
                {metrics ? `${(metrics.mIoU * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
            <div className="metric-card glass-panel">
              <span className="metric-label">F1 / Dice Score</span>
              <span className="metric-value">
                {metrics ? `${(metrics.f1Score * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
