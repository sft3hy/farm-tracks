import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Tractor, Loader2, Play, LayoutDashboard, ChevronRight, BarChart3, X, HelpCircle, BookOpen, Trophy, Clock, Zap, Target, Crosshair, Award } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = "http://localhost:8001";

const MODEL_DISPLAY = {
  unet: { name: 'U-Net', color: '#58a6ff', gradient: 'linear-gradient(135deg, #1e3a5f, #2d5a87)' },
  segformer: { name: 'SegFormer B0', color: '#a78bfa', gradient: 'linear-gradient(135deg, #3b2070, #5b3a9a)' },
  segformer_b4: { name: 'SegFormer B4', color: '#f59e0b', gradient: 'linear-gradient(135deg, #78350f, #b45309)' },
  sam: { name: 'SAM', color: '#34d399', gradient: 'linear-gradient(135deg, #134e3a, #1a7a5a)' },
};

const METRIC_LABELS = {
  iou: 'Mean IoU',
  f1: 'Mean F1 / Dice',
  precision: 'Mean Precision',
  recall: 'Mean Recall',
  latency_ms: 'Avg Latency',
};

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
  const [hoverPred, setHoverPred] = useState(false);
  const [hoverGT, setHoverGT] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [benchmarkStatus, setBenchmarkStatus] = useState(null);
  const [histogramData, setHistogramData] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis', 'training', or 'report'
  const [trainingMd, setTrainingMd] = useState('');
  const [isTrainingLoading, setIsTrainingLoading] = useState(false);
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

    // Fetch benchmark report (one-time, permanent data)
    const fetchBenchmark = async () => {
      try {
        const statusResp = await axios.get(`${API_BASE}/report/status`);
        setBenchmarkStatus(statusResp.data);
        if (statusResp.data.state === 'completed') {
          const [summaryResp, histResp] = await Promise.all([
            axios.get(`${API_BASE}/report/summary`),
            axios.get(`${API_BASE}/report/histograms`),
          ]);
          setBenchmarkData(summaryResp.data);
          setHistogramData(histResp.data);
        }
      } catch (err) { console.error("Benchmark fetch failed", err); }
    };
    fetchBenchmark();

    return () => clearInterval(pollRef.current);
  }, []);


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
          {benchmarkStatus?.state === 'completed' && (
            <span className="report-ready-dot"></span>
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
            <option value="segformer">SegFormer B0</option>
            <option value="segformer_b4">SegFormer B4 (A100)</option>
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
            <p>Parameters and datasets for {MODEL_DISPLAY[selectedModel]?.name || selectedModel}.</p>
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

  // Helper: generate findings text from data
  const generateFindings = (data) => {
    if (!data) return [];
    const { winners, models } = data;
    const findings = [];

    const iouWinner = winners.iou;
    const f1Winner = winners.f1;
    const latencyWinner = winners.latency_ms;

    findings.push({
      icon: <Trophy size={16} />,
      text: <><strong>{MODEL_DISPLAY[iouWinner]?.name}</strong> achieves the highest spatial accuracy (IoU: {(models[iouWinner].iou.mean * 100).toFixed(1)}%), making it the best model for track extraction.</>,
      type: 'success'
    });

    if (f1Winner !== iouWinner) {
      findings.push({
        icon: <Target size={16} />,
        text: <><strong>{MODEL_DISPLAY[f1Winner]?.name}</strong> leads in F1/Dice score ({(models[f1Winner].f1.mean * 100).toFixed(1)}%), indicating superior precision-recall balance.</>,
        type: 'info'
      });
    }

    findings.push({
      icon: <Zap size={16} />,
      text: <><strong>{MODEL_DISPLAY[latencyWinner]?.name}</strong> is the fastest engine at {models[latencyWinner].latency_ms.mean.toFixed(1)}ms average inference, ideal for real-time deployment.</>,
      type: 'speed'
    });

    // Find weakest model by IoU
    const modelNames = Object.keys(models);
    const weakest = modelNames.reduce((a, b) => models[a].iou.mean < models[b].iou.mean ? a : b);
    if (weakest !== iouWinner) {
      const gap = ((models[iouWinner].iou.mean - models[weakest].iou.mean) * 100).toFixed(1);
      findings.push({
        icon: <Crosshair size={16} />,
        text: <>{MODEL_DISPLAY[iouWinner]?.name} outperforms {MODEL_DISPLAY[weakest]?.name} by <strong>{gap} percentage points</strong> on IoU, the largest gap across the fleet.</>,
        type: 'gap'
      });
    }

    // Consistency insight (lowest std)
    const mostConsistent = modelNames.reduce((a, b) => models[a].iou.std < models[b].iou.std ? a : b);
    findings.push({
      icon: <Award size={16} />,
      text: <><strong>{MODEL_DISPLAY[mostConsistent]?.name}</strong> shows the most consistent predictions (IoU σ = {(models[mostConsistent].iou.std * 100).toFixed(1)}%).</>,
      type: 'info'
    });

    return findings;
  };

  // Helper: render a mini histogram
  const renderHistogram = (histData, color, label) => {
    if (!histData) return null;
    const maxCount = Math.max(...histData.counts);
    return (
      <div className="bm-histogram">
        <div className="bm-histogram-label">{label}</div>
        <div className="bm-histogram-bars">
          {histData.counts.map((count, i) => (
            <div key={i} className="bm-histogram-col">
              <div
                className="bm-histogram-bar"
                style={{
                  height: maxCount > 0 ? `${(count / maxCount) * 100}%` : '0%',
                  background: color,
                }}
                title={`${(histData.bin_edges[i] * 100).toFixed(0)}–${(histData.bin_edges[i + 1] * 100).toFixed(0)}%: ${count} images`}
              />
            </div>
          ))}
        </div>
        <div className="bm-histogram-axis">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    );
  };

  const renderReportTab = () => {
    const data = benchmarkData;
    const hasData = data && data.models;
    const modelNames = hasData ? Object.keys(data.models) : [];

    return (
    <main className="main-content report-view">
      <div className="bm-report-wrapper">

        {/* ── Report Header ── */}
        <div className="bm-report-header glass-panel">
          <div className="bm-report-title-row">
            <div className="bm-report-title">
              <BarChart3 size={24} color="#58a6ff" />
              <div>
                <h2>Model Performance Benchmark</h2>
                <span className="bm-subtitle">Agriculture-Vision · Planter Skip Segmentation</span>
              </div>
            </div>
            <div className="bm-report-badges">
              {hasData && (
                <>
                  <span className="bm-badge bm-badge-finalized">
                    <span className="bm-badge-dot" />
                    Report Finalized
                  </span>
                  <span className="bm-badge">
                    <Target size={12} />
                    {data.meta.total_samples.toLocaleString()} Samples
                  </span>
                  <span className="bm-badge">
                    <Zap size={12} />
                    {data.meta.device.toUpperCase()}
                  </span>
                  <span className="bm-badge">
                    <Clock size={12} />
                    {new Date(data.meta.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {!hasData ? (
          <div className="bm-empty-state glass-panel">
            <BarChart3 size={56} opacity={0.2} />
            <h3>No Benchmark Report Available</h3>
            <p>Run the benchmark to generate permanent performance data:</p>
            <code>python src/run_benchmark.py</code>
          </div>
        ) : (
          <>

        {/* ── Model Cards ── */}
        <div className="bm-model-cards">
          {modelNames.map(mName => {
            const m = data.models[mName];
            const display = MODEL_DISPLAY[mName] || { name: mName, color: '#fff', gradient: 'linear-gradient(135deg, #222, #333)' };
            const isIoUWinner = data.winners.iou === mName;
            const isLatencyWinner = data.winners.latency_ms === mName;

            return (
              <div key={mName} className={`bm-model-card glass-panel ${isIoUWinner ? 'bm-winner' : ''}`}>
                {isIoUWinner && (
                  <div className="bm-crown-badge">
                    <Trophy size={14} />
                    Best Accuracy
                  </div>
                )}
                {!isIoUWinner && isLatencyWinner && (
                  <div className="bm-crown-badge bm-speed-badge">
                    <Zap size={14} />
                    Fastest
                  </div>
                )}
                <div className="bm-card-header" style={{ background: display.gradient }}>
                  <h3>{display.name}</h3>
                </div>
                <div className="bm-card-body">
                  <div className="bm-card-metric-primary">
                    <span className="bm-card-metric-label">IoU</span>
                    <span className="bm-card-metric-value" style={{ color: display.color }}>
                      {(m.iou.mean * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bm-card-metrics-grid">
                    <div className="bm-card-metric-small">
                      <span>F1</span>
                      <span>{(m.f1.mean * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bm-card-metric-small">
                      <span>Precision</span>
                      <span>{(m.precision.mean * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bm-card-metric-small">
                      <span>Recall</span>
                      <span>{(m.recall.mean * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bm-card-metric-small">
                      <span>Latency</span>
                      <span>{m.latency_ms.mean.toFixed(1)}ms</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Metrics Comparison Bars ── */}
        <div className="bm-bars-section glass-panel">
          <h3 className="bm-section-title">Metric Comparison</h3>
          <div className="bm-bar-groups">
            {['iou', 'f1', 'precision', 'recall'].map(metric => {
              const maxVal = Math.max(...modelNames.map(m => data.models[m][metric].mean));
              return (
                <div key={metric} className="bm-bar-group">
                  <div className="bm-bar-group-label">{METRIC_LABELS[metric]}</div>
                  {modelNames.map(mName => {
                    const val = data.models[mName][metric].mean;
                    const display = MODEL_DISPLAY[mName];
                    const isWinner = data.winners[metric] === mName;
                    return (
                      <div key={mName} className="bm-bar-row">
                        <span className="bm-bar-model-label" style={{ color: display.color }}>{display.name}</span>
                        <div className="bm-bar-track">
                          <div
                            className="bm-bar-fill"
                            style={{ width: `${val * 100}%`, background: display.color, opacity: isWinner ? 1 : 0.6 }}
                          />
                        </div>
                        <span className={`bm-bar-value ${isWinner ? 'bm-bar-value-winner' : ''}`}>
                          {(val * 100).toFixed(1)}%
                          {isWinner && <Trophy size={10} />}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Detailed Stats Table + Latency ── */}
        <div className="bm-detail-row">
          <div className="bm-detailed-table glass-panel">
            <h3 className="bm-section-title">Statistical Breakdown</h3>
            <div className="bm-table-scroll">
              <table className="bm-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Model</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std σ</th>
                    <th>Min</th>
                    <th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {['iou', 'f1', 'precision', 'recall'].map(metric =>
                    modelNames.map((mName, idx) => {
                      const s = data.models[mName][metric];
                      const display = MODEL_DISPLAY[mName];
                      const isWinner = data.winners[metric] === mName;
                      return (
                        <tr key={`${metric}-${mName}`} className={isWinner ? 'bm-row-winner' : ''}>
                          {idx === 0 && <td rowSpan={modelNames.length} className="bm-metric-cell">{METRIC_LABELS[metric]}</td>}
                          <td style={{ color: display.color }}>{display.name}</td>
                          <td className="bm-val-cell">{(s.mean * 100).toFixed(2)}%</td>
                          <td>{(s.median * 100).toFixed(2)}%</td>
                          <td>{(s.std * 100).toFixed(2)}%</td>
                          <td>{(s.min * 100).toFixed(2)}%</td>
                          <td>{(s.max * 100).toFixed(2)}%</td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bm-latency-panel glass-panel">
            <h3 className="bm-section-title">Inference Latency</h3>
            <div className="bm-latency-bars">
              {modelNames.map(mName => {
                const lat = data.models[mName].latency_ms;
                const display = MODEL_DISPLAY[mName];
                const maxLat = Math.max(...modelNames.map(m => data.models[m].latency_ms.mean));
                const isWinner = data.winners.latency_ms === mName;
                return (
                  <div key={mName} className="bm-latency-item">
                    <div className="bm-latency-label">
                      <span style={{ color: display.color }}>{display.name}</span>
                      <span className="bm-latency-value">
                        {lat.mean.toFixed(1)}ms
                        {isWinner && <Zap size={12} color="#fbbf24" />}
                      </span>
                    </div>
                    <div className="bm-latency-track">
                      <div
                        className="bm-latency-fill"
                        style={{ width: `${(lat.mean / maxLat) * 100}%`, background: display.color }}
                      />
                    </div>
                    <div className="bm-latency-stats">
                      <span>med: {lat.median.toFixed(1)}</span>
                      <span>σ: {lat.std.toFixed(1)}</span>
                      <span>max: {lat.max.toFixed(0)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── IoU Distribution Histograms ── */}
        {histogramData && (
          <div className="bm-histograms-section glass-panel">
            <h3 className="bm-section-title">IoU Score Distribution</h3>
            <p className="bm-section-subtitle">How each model's predictions spread across IoU bins (0–100%)</p>
            <div className="bm-histograms-grid">
              {modelNames.map(mName => {
                const display = MODEL_DISPLAY[mName];
                const hist = histogramData[mName];
                return (
                  <div key={mName} className="bm-histogram-card">
                    {renderHistogram(hist?.iou, display.color, display.name)}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Key Findings ── */}
        <div className="bm-findings glass-panel">
          <h3 className="bm-section-title">Key Findings</h3>
          <div className="bm-findings-list">
            {generateFindings(data).map((finding, i) => (
              <div key={i} className={`bm-finding-item bm-finding-${finding.type}`}>
                <div className="bm-finding-icon">{finding.icon}</div>
                <p>{finding.text}</p>
              </div>
            ))}
          </div>
        </div>

          </>
        )}
      </div>
    </main>
  );
  };

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

      {/* Footer for report tab */}
      {activeTab === 'report' && benchmarkData && (
        <footer className="bm-footer">
          <span>Report generated via <code>python src/run_benchmark.py</code> · Results are permanent and not recomputed on each visit</span>
        </footer>
      )}
    </div>
  );
}

export default App;
