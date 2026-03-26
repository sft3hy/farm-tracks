import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Tractor, Loader2, Play, LayoutDashboard, ChevronRight } from 'lucide-react';
import './App.css';

const API_BASE = "http://localhost:8001";

function App() {
  const [batch, setBatch] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isInferencing, setIsInferencing] = useState(false);
  const [isBatchLoading, setIsBatchLoading] = useState(false);
  const [maskData, setMaskData] = useState(null);
  const [gtMaskData, setGtMaskData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [totalFields, setTotalFields] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState(null);
  const pollRef = useRef(null);

  // Poll /status until dataset is ready
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const resp = await axios.get(`${API_BASE}/status`);
        setLoadingStatus(resp.data);
        if (resp.data.state === 'ready') {
          clearInterval(pollRef.current);
          fetchBatch();
        }
      } catch (err) {
        // Server not up yet, keep polling
      }
    };

    pollStatus();
    pollRef.current = setInterval(pollStatus, 2000);
    return () => clearInterval(pollRef.current);
  }, []);

  const fetchBatch = async () => {
    setIsBatchLoading(true);
    setBatch([]);
    setSelectedImage(null);
    setMaskData(null);
    setGtMaskData(null);
    setMetrics(null);
    try {
      const resp = await axios.get(`${API_BASE}/batch`);
      const data = resp.data;

      if (data.loading) {
        setLoadingStatus(data.loading);
        return;
      }

      setBatch(data.batch);
      setTotalFields(data.total_fields);
      setCurrentPage(data.page);

      if (data.batch.length > 0) {
        handleSelect(data.batch[0]);
      }
    } catch (err) {
      console.error("Failed to fetch batch:", err);
    } finally {
      setIsBatchLoading(false);
    }
  };

  const handleSelect = (img) => {
    setSelectedImage(img);
    if (img.inference) {
      setMaskData(`data:image/png;base64,${img.inference.mask_base64}`);
      if (img.inference.gt_mask_base64) {
        setGtMaskData(`data:image/png;base64,${img.inference.gt_mask_base64}`);
      } else {
        setGtMaskData(null);
      }
      setMetrics(img.inference.metrics);
    } else {
      setMaskData(null);
      setGtMaskData(null);
      setMetrics(null);
    }
  };

  // Show loading screen while dataset initializes
  const isReady = loadingStatus?.state === 'ready';

  return (
    <div className="app-container">
      {!isReady && (
        <div className="global-loader">
          <div className="loader-content">
            <Loader2 size={48} className="spin" color="#58a6ff" />
            <p>Initializing Agriculture-Vision Dataset</p>
            {loadingStatus?.state === 'loading_dataset' && (
              <span className="loader-subtext">Downloading &amp; indexing dataset (first time may take a few minutes)...</span>
            )}
            {loadingStatus?.state === 'building_index' && (
              <>
                <span className="loader-subtext">
                  Building RGB/mask pairs: {loadingStatus.progress?.toLocaleString()} / {loadingStatus.total?.toLocaleString()} scanned
                </span>
                <span className="loader-subtext" style={{ color: '#3fb950' }}>
                  Found {loadingStatus.found} paired fields so far
                </span>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{ width: loadingStatus.total > 0 ? `${(loadingStatus.progress / loadingStatus.total) * 100}%` : '0%' }}
                  />
                </div>
              </>
            )}
            {loadingStatus?.state === 'error' && (
              <span className="loader-subtext" style={{ color: '#f85149' }}>Error loading dataset. Check backend logs.</span>
            )}
            {!loadingStatus && (
              <span className="loader-subtext">Connecting to backend...</span>
            )}
          </div>
        </div>
      )}

      <header className="header glass-panel">
        <div className="header-title">
          <LayoutDashboard size={24} color="#58a6ff" />
          <h1>FarmTrack Analytics</h1>
        </div>
        <div className="header-info">
          <span>Source: <strong>Agriculture-Vision</strong></span>
          <span className="separator">|</span>
          <span>Engine: <strong style={{ color: "#3fb950" }}>PyTorch U-Net</strong></span>
          {totalFields > 0 && (
            <>
              <span className="separator">|</span>
              <span>Fields: <strong>{totalFields.toLocaleString()}</strong></span>
            </>
          )}
        </div>
      </header>

      <main className="main-content">
        <aside className="gallery-sidebar glass-panel">
          <div className="sidebar-header">
            <h2>Dataset Explorer</h2>
            <button
              className="btn-secondary btn-small"
              onClick={fetchBatch}
              disabled={isBatchLoading || !isReady}
            >
              {isBatchLoading ? (
                <><Loader2 size={14} className="spin" style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '4px' }} /> Auto-Predicting...</>
              ) : (
                "Load & Predict Next 10"
              )}
            </button>
          </div>
          <div className="sidebar-subtext">
            Page {currentPage} &middot; Paired fields with planter_skip masks
          </div>
          <div className="image-list">
            {batch.map(img => (
              <div
                key={img.file_id}
                className={`image-item ${selectedImage?.file_id === img.file_id ? 'active' : ''}`}
                onClick={() => handleSelect(img)}
              >
                <div className="thumb-preview">
                  <img src={`data:image/jpeg;base64,${img.thumbnail}`} alt="thumb" />
                </div>
                <div className="image-meta">
                  <span className="filename">{img.file_id}</span>
                  <span className="index" style={{ color: img.inference?.metrics?.mIoU > 0 ? '#7ee787' : '#8b949e'}}>
                    IoU: {img.inference ? (img.inference.metrics.mIoU * 100).toFixed(1) + "%" : "..."}
                  </span>
                </div>
                {selectedImage?.file_id === img.file_id && <ChevronRight size={16} />}
              </div>
            ))}
            {batch.length === 0 && isReady && !isBatchLoading && (
              <div className="empty-msg">No images found.</div>
            )}
          </div>
        </aside>

        <section className="visualizer-area">
          <div className="toolbar glass-panel">
            <div className="active-selection">
              {selectedImage ? (
                <>
                  <span className="label">Viewing Analysis:</span>
                  <span className="value">{selectedImage.file_id}</span>
                </>
              ) : (
                <span className="text-muted">Select a field to view analysis</span>
              )}
            </div>
          </div>

          <div className="image-display glass-panel">
            {selectedImage ? (
              <>
                <div className="dual-view">
                  <div className="view-panel">
                    <div className="view-label">Input + Prediction Overlay</div>
                  <div className="image-container">
                    <img
                      className="low-res-blur"
                      src={`data:image/jpeg;base64,${selectedImage.thumbnail}`}
                      alt="placeholder"
                    />
                    <img
                      src={`${API_BASE}/image/${selectedImage.file_id}`}
                      alt="Raw Aerial Field"
                      key={`raw-${selectedImage.file_id}`}
                      onLoad={(e) => e.target.classList.add('loaded')}
                    />
                    {maskData && (
                      <img
                        src={maskData}
                        alt="Prediction Overlay"
                        className="mask-overlay visible"
                        key={`mask-${selectedImage.file_id}`}
                        onLoad={(e) => e.target.classList.add('loaded')}
                      />
                    )}
                    {!maskData && (
                      <div className="hint-overlay">
                        No prediction masks found
                      </div>
                    )}
                  </div>
                  <p className="explain-text">The raw aerial RGB image overlaid with the model's AI-generated track predictions in hot pink.</p>
                </div>

                <div className="view-panel">
                  <div className="view-label">Ground Truth (planter_skip)</div>
                  <div className="image-container gt-container">
                    {gtMaskData ? (
                      <img 
                        src={gtMaskData} 
                        alt="Ground Truth Mask" 
                        className="gt-mask" 
                        onLoad={(e) => e.target.classList.add('loaded')}
                      />
                    ) : (
                      <div className="hint-overlay">
                        No ground truth mask found
                      </div>
                    )}
                  </div>
                  <p className="explain-text">The human-annotated actual locations of planter skips (areas where seeds failed to drop, leaving bare tracks).</p>
                </div>
              </div>

              <div className="dual-view" style={{ marginTop: '1.5rem' }}>
                <div className="view-panel">
                  <div className="view-label">Raw Aerial Input</div>
                  <div className="image-container">
                    <img
                      className="low-res-blur"
                      src={`data:image/jpeg;base64,${selectedImage.thumbnail}`}
                      alt="placeholder"
                    />
                    <img
                      src={`${API_BASE}/image/${selectedImage.file_id}`}
                      alt="Raw Aerial Field Isolated"
                      key={`raw-iso-${selectedImage.file_id}`}
                      onLoad={(e) => e.target.classList.add('loaded')}
                    />
                  </div>
                  <p className="explain-text">The original high-resolution satellite/drone image without any AI overlays.</p>
                </div>

                <div className="view-panel">
                  <div className="view-label">AI Detection Mask (Isolated)</div>
                  <div className="image-container gt-container">
                    {maskData ? (
                      <img 
                        src={maskData} 
                        alt="Isolated Prediction Mask" 
                        className="gt-mask" 
                        onLoad={(e) => e.target.classList.add('loaded')}
                      />
                    ) : (
                      <div className="hint-overlay">
                        No prediction masks found
                      </div>
                    )}
                  </div>
                  <p className="explain-text">The AI model's raw semantic segmentation output on a black background, clearly isolating the detected tracks.</p>
                </div>
              </div>
              </>
            ) : (
              <div className="empty-state">
                <Tractor size={64} opacity={0.2} />
                <p>Select a field from the database to start track detection</p>
              </div>
            )}
          </div>

          <div className="metrics-panel">
            <div className="metric-card glass-panel">
              <span className="metric-label">IoU Score</span>
              <span className="metric-value">
                {metrics ? `${(metrics.mIoU * 100).toFixed(1)}%` : '--'}
              </span>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: metrics ? `${metrics.mIoU * 100}%` : '0%' }}></div>
              </div>
              <p className="explain-text metric-explain">Intersection over Union. Measures spatial overlap between the model's prediction and the ground truth mask. 100% means a perfect pixel-for-pixel match.</p>
            </div>
            <div className="metric-card glass-panel">
              <span className="metric-label">F1 / Dice Score</span>
              <span className="metric-value">
                {metrics ? `${(metrics.f1Score * 100).toFixed(1)}%` : '--'}
              </span>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: metrics ? `${metrics.f1Score * 100}%` : '0%', backgroundColor: '#58a6ff' }}></div>
              </div>
              <p className="explain-text metric-explain">Harmonic mean of precision and recall. Punishes false positives and false negatives equally. Often called the Dice Coefficient in segmentation.</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
