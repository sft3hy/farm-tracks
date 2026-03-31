import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Tractor, Loader2, Play, LayoutDashboard, ChevronRight, BarChart3, X, HelpCircle, BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
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
  const [selectedModel, setSelectedModel] = useState('unet');
  const [comparisonData, setComparisonData] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [hoverPred, setHoverPred] = useState(false);
  const [hoverGT, setHoverGT] = useState(false);
  const [reportStatus, setReportStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis', 'training', or 'report'
  const [trainingMd, setTrainingMd] = useState('');
  const [isTrainingLoading, setIsTrainingLoading] = useState(false);
  const pollRef = useRef(null);
  const reportPollRef = useRef(null);

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

    // Initial report status check
    const checkReport = async () => {
      try {
        const statusResp = await axios.get(`${API_BASE}/report/status`);
        setReportStatus(statusResp.data);
        if (statusResp.data.state === 'running') {
          startReportPolling();
        } else if (statusResp.data.state === 'completed') {
          const resResp = await axios.get(`${API_BASE}/report/results`);
          setComparisonData(resResp.data);
        }
      } catch (err) { console.error("Report check failed", err); }
    };
    checkReport();

    return () => {
      clearInterval(pollRef.current);
      if (reportPollRef.current) clearInterval(reportPollRef.current);
    };
  }, []);

  const startReportPolling = () => {
    if (reportPollRef.current) clearInterval(reportPollRef.current);
    reportPollRef.current = setInterval(async () => {
      try {
        const resp = await axios.get(`${API_BASE}/report/status`);
        setReportStatus(resp.data);
        if (resp.data.state === 'completed') {
          clearInterval(reportPollRef.current);
          const resResp = await axios.get(`${API_BASE}/report/results`);
          setComparisonData(resResp.data);
        } else if (resp.data.state === 'error') {
          clearInterval(reportPollRef.current);
        }
      } catch (err) { console.error("Report polling failed", err); }
    }, 3000);
  };

  // Re-fetch batch or training doc when model changes
  useEffect(() => {
    if (isReady) {
      if (activeTab === 'analysis') {
        fetchBatch();
      } else if (activeTab === 'training') {
        fetchTrainingDoc();
      }
    }
  }, [selectedModel, activeTab]);

  const fetchTrainingDoc = async () => {
    setIsTrainingLoading(true);
    setTrainingMd('');
    try {
      const resp = await axios.get(`${API_BASE}/training-explanation/${selectedModel}`);
      setTrainingMd(resp.data.content);
    } catch (err) {
      console.error("Failed to fetch training doc:", err);
      setTrainingMd("## Error\nCould not load training documentation.");
    } finally {
      setIsTrainingLoading(false);
    }
  };

  const fetchBatch = async () => {
    setIsBatchLoading(true);
    setBatch([]);
    setSelectedImage(null);
    setMaskData(null);
    setGtMaskData(null);
    setMetrics(null);
    try {
      const resp = await axios.get(`${API_BASE}/batch?model=${selectedModel}`);
      const data = resp.data;

      if (data.loading) {
        setLoadingStatus(data.loading);
        return;
      }

      // Sort batch by IoUDescending (highest at top)
      const sortedBatch = [...data.batch].sort((a, b) => {
        const iouA = a.inference?.metrics?.mIoU || 0;
        const iouB = b.inference?.metrics?.mIoU || 0;
        return iouB - iouA;
      });

      setBatch(sortedBatch);
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

  const renderHeader = () => (
    <header className="header glass-panel">
      <div className="header-left">
        <LayoutDashboard size={22} color="#58a6ff" />
        <div className="title-group">
          <h1>FarmTrack Analytics</h1>
          <span className="source-tag">Agriculture-Vision</span>
        </div>
      </div>

      <div className="header-tabs">
        <button
          className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          <Play size={14} style={{ marginRight: '6px' }} />
          Analysis
        </button>
        <button
          className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <BookOpen size={14} style={{ marginRight: '6px' }} />
          Training
        </button>
        <button
          className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
          onClick={() => setActiveTab('report')}
        >
          <BarChart3 size={14} style={{ marginRight: '6px' }} />
          Performance
          {reportStatus?.state === 'running' && (
            <span className="dot-pulsing"></span>
          )}
        </button>
      </div>

      <div className="header-right">
        <div className="field-metric">
          <span className="label">Fields:</span>
          <span className="value">{totalFields.toLocaleString()}</span>
        </div>
        <div className="engine-dropdown-compact">
          <span className="label">Engine:</span>
          <select
            className="model-select-compact"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <option value="unet">U-Net</option>
            <option value="segformer">SegFormer</option>
            <option value="sam">SAM</option>
          </select>
        </div>
      </div>
    </header>
  );

  const renderAnalysisTab = () => (
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
              <><Loader2 size={14} className="spin" /> Updating...</>
            ) : (
              "Load Next Batch"
            )}
          </button>
        </div>
        <div className="sidebar-subtext">
          Page {currentPage} &middot; Fields with track anomalies
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
                <span className="index" style={{ color: img.inference?.metrics?.mIoU > 0 ? '#7ee787' : '#8b949e' }}>
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
                <span className="label">Viewing:</span>
                <span className="value">{selectedImage.file_id}</span>
              </>
            ) : (
              <span className="text-muted">Select a field for track detection</span>
            )}
          </div>
        </div>

        <div className="image-display-ovh glass-panel">
          {selectedImage ? (
            <div className="overhaul-dual-view">
              <div
                className="view-panel-compact"
                onMouseEnter={() => setHoverPred(true)}
                onMouseLeave={() => setHoverPred(false)}
              >
                <div className="view-label-compact">Model Prediction {hoverPred && <span className="peek-hint">(Raw)</span>}</div>
                <div className="image-container-compact">
                  <img
                    src={`${API_BASE}/image/${selectedImage.file_id}`}
                    alt="Raw Input"
                    className="base-img"
                  />
                  {maskData && (
                    <img
                      src={maskData}
                      alt="Prediction Overlay"
                      className={`mask-overlay-interactive ${hoverPred ? 'hidden' : 'visible'}`}
                    />
                  )}
                </div>
              </div>

              <div
                className="view-panel-compact"
                onMouseEnter={() => setHoverGT(true)}
                onMouseLeave={() => setHoverGT(false)}
              >
                <div className="view-label-compact">Ground Truth {hoverGT && <span className="peek-hint">(Raw)</span>}</div>
                <div className="image-container-compact">
                  <img
                    src={`${API_BASE}/image/${selectedImage.file_id}`}
                    alt="Raw Input"
                    className="base-img"
                  />
                  {gtMaskData && (
                    <img
                      src={gtMaskData}
                      alt="GT Overlay"
                      className={`mask-overlay-interactive ${hoverGT ? 'hidden' : 'visible'}`}
                    />
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <Tractor size={64} opacity={0.2} />
              <p>Select a field from the database to start track detection</p>
            </div>
          )}
        </div>

        <div className="metrics-panel-compact">
          <div className="metric-card-compact glass-panel">
            <div className="metric-header-compact">
              <div className="metric-label-with-help">
                <span className="metric-label-compact">IoU Score</span>
                <div className="help-tooltip">
                  <HelpCircle size={12} className="help-icon" />
                  <span className="tooltip-text">
                    <strong>Intersection over Union</strong><br />
                    Measures spatial overlap with ground truth. 100% means a perfect match.
                  </span>
                </div>
              </div>
              <span className="metric-value-compact">
                {metrics ? `${(metrics.mIoU * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
            <div className="metric-bar-bg-compact">
              <div className="metric-bar-fill-compact" style={{ width: metrics ? `${metrics.mIoU * 100}%` : '0%' }}></div>
            </div>
          </div>
          <div className="metric-card-compact glass-panel">
            <div className="metric-header-compact">
              <div className="metric-label-with-help">
                <span className="metric-label-compact">F1 / Dice Score</span>
                <div className="help-tooltip">
                  <HelpCircle size={12} className="help-icon" />
                  <span className="tooltip-text">
                    <strong>F1 / Dice Coefficient</strong><br />
                    Harmonic mean of precision and recall.
                  </span>
                </div>
              </div>
              <span className="metric-value-compact">
                {metrics ? `${(metrics.f1Score * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
            <div className="metric-bar-bg-compact">
              <div className="metric-bar-fill-compact" style={{ width: metrics ? `${metrics.f1Score * 100}%` : '0%', backgroundColor: '#58a6ff' }}></div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );

  const renderTrainingTab = () => (
    <main className="main-content training-view">
      <aside className="gallery-sidebar glass-panel training-sidebar-compact">
        <h2>Model Reference</h2>
        <div className="training-nav-list-compact">
          <div className="training-nav-item active">
            <BookOpen size={16} />
            <span>Training Protocol</span>
          </div>
          <div className="training-nav-info-compact">
            <p>Parameters and datasets for {selectedModel.toUpperCase()}.</p>
          </div>
        </div>
      </aside>

      <section className="visualizer-area training-content-area-compact glass-panel">
        {isTrainingLoading ? (
          <div className="loading-container">
            <Loader2 size={32} className="spin" color="#51cf66" />
            <p>Loading Documentation...</p>
          </div>
        ) : (
          <div className="markdown-container">
            <ReactMarkdown>{trainingMd}</ReactMarkdown>
          </div>
        )}
      </section>
    </main>
  );

  const renderReportTab = () => (
    <main className="main-content report-view">
      <div className="report-container glass-panel">
        <div className="report-header">
          <h2>Fleet Performance Metrics</h2>
          <div className="report-badges">
            <span className="badge">Sample Size: {comparisonData?.summary.sample_count || '--'} Fields</span>
            {reportStatus?.state === 'running' && (
              <span className="badge-pulsing">Evaluating Engines... {Math.round((reportStatus.progress / reportStatus.total) * 100)}%</span>
            )}
          </div>
        </div>

        {reportStatus?.state === 'running' && !comparisonData ? (
          <div className="report-loading">
            <Loader2 size={48} className="spin" color="#51cf66" />
            <p>Processing random sample of {reportStatus.total} fields...</p>
          </div>
        ) : comparisonData ? (
          <div className="comparison-report">
            <div className="report-summary">
              <div className="summary-card highlight">
                <span className="label">Optimal Engine (IoU)</span>
                <span className="value uppercase">{comparisonData.summary.winners.mIoU}</span>
              </div>
              <div className="summary-card">
                <span className="label">Efficiency Leader</span>
                <span className="value uppercase">{comparisonData.summary.winners.avg_inference_ms}</span>
              </div>
            </div>

            <div className="table-wrapper">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>PyTorch U-Net</th>
                    <th>SegFormer</th>
                    <th>SAM</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Mean IoU</td>
                    <td>{(comparisonData.metrics.unet.mIoU * 100).toFixed(1)}%</td>
                    <td>{(comparisonData.metrics.segformer.mIoU * 100).toFixed(1)}%</td>
                    <td>{(comparisonData.metrics.sam.mIoU * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td>Mean F1 / Dice</td>
                    <td>{(comparisonData.metrics.unet.mF1 * 100).toFixed(1)}%</td>
                    <td>{(comparisonData.metrics.segformer.mF1 * 100).toFixed(1)}%</td>
                    <td>{(comparisonData.metrics.sam.mF1 * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td>Mean Latency</td>
                    <td>{Math.round(comparisonData.metrics.unet.avg_inference_ms)}ms</td>
                    <td>{Math.round(comparisonData.metrics.segformer.avg_inference_ms)}ms</td>
                    <td>{Math.round(comparisonData.metrics.sam.avg_inference_ms)}ms</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="findings-box">
              <h3>Evaluation Synthesis</h3>
              <ul>
                <li><strong>{comparisonData.summary.winners.mIoU.toUpperCase()}</strong> is currently the most accurate for track extraction.</li>
                <li><strong>{comparisonData.summary.winners.avg_inference_ms.toUpperCase()}</strong> provides the lowest latency bottleneck.</li>
                <li>Transformers (SegFormer/SAM) show superior robustness to lighting variations.</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="report-empty">
            <p>Initial evaluation for this session has not started or failed.</p>
          </div>
        )}
      </div>
    </main>
  );

  return (
    <div className="app-container">
      {!isReady && (
        <div className="global-loader">
          <div className="loader-content">
            <Loader2 size={48} className="spin" color="#58a6ff" />
            <p>Initializing Agriculture-Vision AI Dashboard</p>
            {loadingStatus?.state === 'loading_dataset' && (
              <span className="loader-subtext">Mounting remote imagery datasets...</span>
            )}
            {loadingStatus?.state === 'building_index' && (
              <>
                <span className="loader-subtext">
                  Mapping track anomalies: {loadingStatus.progress?.toLocaleString()} / {loadingStatus.total?.toLocaleString()}
                </span>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{ width: loadingStatus.total > 0 ? `${(loadingStatus.progress / loadingStatus.total) * 100}%` : '0%' }}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {renderHeader()}
      {activeTab === 'analysis' && renderAnalysisTab()}
      {activeTab === 'training' && renderTrainingTab()}
      {activeTab === 'report' && renderReportTab()}
    </div>
  );
}

export default App;
