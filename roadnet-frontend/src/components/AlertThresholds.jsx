import { useState } from "react";

function AlertThresholds() {
    const [latency, setLatency] = useState(15);
    const [level, setLevel] = useState("Level 5 (Emergency)");

    const pct = ((latency - 5) / (60 - 5)) * 100;

    return (
        <div className="s-panel alert-panel">
            <div className="alert-panel-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                Alert Thresholds
            </div>

            <div className="slider-section">
                <div className="slider-label">Response Latency (min)</div>
                <div className="slider-track">
                    <input
                        type="range"
                        min={5}
                        max={60}
                        value={latency}
                        onChange={(e) => setLatency(Number(e.target.value))}
                        style={{ "--pct": `${pct}%` }}
                    />
                </div>
                <div className="slider-marks">
                    <span>5m</span>
                    <span>{latency}m (Current)</span>
                    <span>60m</span>
                </div>
            </div>

            <div>
                <div className="threshold-dropdown-label">Critical Priority Level</div>
                <select
                    className="threshold-select"
                    value={level}
                    onChange={(e) => setLevel(e.target.value)}
                >
                    <option>Level 1</option>
                    <option>Level 2</option>
                    <option>Level 3</option>
                    <option>Level 4</option>
                    <option>Level 5 (Emergency)</option>
                </select>
            </div>
        </div>
    );
}

export default AlertThresholds;
