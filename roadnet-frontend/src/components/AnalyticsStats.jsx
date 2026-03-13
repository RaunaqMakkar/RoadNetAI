function AnalyticsStats({ stats, tickets }) {
    const total = stats?.total_tickets || 0;
    const avgRps = stats?.avg_rps_score || 0;
    const maxRps = stats?.max_rps_score || 0;
    const statusBreakdown = stats?.breakdown_by_status || {};
    const priorityBreakdown = stats?.breakdown_by_priority || {};

    const closed = statusBreakdown["Closed"] || 0;
    const open = statusBreakdown["Open"] || 0;
    const inProgress = statusBreakdown["In Progress"] || 0;
    const critical = priorityBreakdown["Critical"] || 0;
    const high = priorityBreakdown["High"] || 0;

    const closureRate = total > 0 ? ((closed / total) * 100).toFixed(1) : "0.0";
    const rpsIndex = avgRps > 0 ? (avgRps / 10).toFixed(1) : "0.0";

    // Compute avg response time from RPS scores in ticket data
    const rpsScores = (tickets || []).map((t) => t.rps_score).filter((v) => v != null);
    const avgResponseHrs = rpsScores.length > 0
        ? (rpsScores.reduce((a, b) => a + b, 0) / rpsScores.length / 10).toFixed(1)
        : "—";

    // Critical ratio for trend
    const criticalPct = total > 0 ? ((critical + high) / total * 100).toFixed(1) : "0";
    const openPct = total > 0 ? ((open + inProgress) / total * 100).toFixed(1) : "0";

    return (
        <div className="analytics-kpis">
            <div className="analytics-kpi">
                <div className="analytics-kpi-content">
                    <div className="analytics-kpi-title">Total Issues Detected</div>
                    <div className="analytics-kpi-value">{total.toLocaleString()}</div>
                    <div className={`analytics-kpi-trend ${open > closed ? "up" : "down"}`}>
                        {open > closed ? "↗" : "↘"} {openPct}% still open
                    </div>
                </div>
                <div className="analytics-kpi-icon blue">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /></svg>
                </div>
            </div>

            <div className="analytics-kpi">
                <div className="analytics-kpi-content">
                    <div className="analytics-kpi-title">Closure Rate</div>
                    <div className="analytics-kpi-value">{closureRate}%</div>
                    <div className={`analytics-kpi-trend ${Number(closureRate) > 50 ? "up" : "down"}`}>
                        {Number(closureRate) > 50 ? "↗" : "↘"} {closed} of {total} closed
                    </div>
                </div>
                <div className="analytics-kpi-icon green">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
                </div>
            </div>

            <div className="analytics-kpi">
                <div className="analytics-kpi-content">
                    <div className="analytics-kpi-title">Avg. RPS Score</div>
                    <div className="analytics-kpi-value">{avgRps}</div>
                    <div className={`analytics-kpi-trend ${avgRps > 50 ? "up" : "down"}`}>
                        {avgRps > 50 ? "↗" : "↘"} Max: {maxRps}
                    </div>
                </div>
                <div className="analytics-kpi-icon orange">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
                </div>
            </div>

            <div className="analytics-kpi">
                <div className="analytics-kpi-content">
                    <div className="analytics-kpi-title">RPS Index</div>
                    <div className="analytics-kpi-value">{rpsIndex} / 10</div>
                    <div className={`analytics-kpi-trend ${criticalPct > 20 ? "up" : "neutral"}`}>
                        {criticalPct}% critical/high priority
                    </div>
                </div>
                <div className="analytics-kpi-icon purple">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                </div>
            </div>
        </div>
    );
}

export default AnalyticsStats;
