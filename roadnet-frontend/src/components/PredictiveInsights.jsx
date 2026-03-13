function PredictiveInsights({ stats, tickets }) {
    const total = stats?.total_tickets || 0;
    const avgRps = stats?.avg_rps_score || 0;
    const typeBreakdown = stats?.breakdown_by_type || {};
    const priorityBreakdown = stats?.breakdown_by_priority || {};

    // Find the most common issue type
    const topType = Object.entries(typeBreakdown).sort((a, b) => b[1] - a[1])[0];
    const topTypeName = topType ? topType[0].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Unknown";
    const topTypeCount = topType ? topType[1] : 0;
    const topTypePct = total > 0 ? Math.round((topTypeCount / total) * 100) : 0;

    // Find high-RPS tickets (potential priority areas)
    const highRpsTickets = (tickets || []).filter((t) => (t.rps_score || 0) > 75);
    const highRpsZones = {};
    highRpsTickets.forEach((t) => {
        const zone = t.zone || "Unknown";
        highRpsZones[zone] = (highRpsZones[zone] || 0) + 1;
    });
    const topRpsZone = Object.entries(highRpsZones).sort((a, b) => b[1] - a[1])[0];

    // Critical ratio
    const critical = (priorityBreakdown["Critical"] || 0) + (priorityBreakdown["High"] || 0);
    const critPct = total > 0 ? Math.round((critical / total) * 100) : 0;

    const insights = [
        {
            title: `${topTypeName} Hotspot Alert`,
            desc: `${topTypePct}% of all detected issues (${topTypeCount} of ${total}) are ${topTypeName.toLowerCase()} defects. Recommend prioritized inspection schedule.`,
            badge: "High Priority",
            badgeClass: "high",
        },
        {
            title: "Resource Optimization",
            desc: highRpsTickets.length > 0
                ? `${highRpsTickets.length} tickets have RPS > 75. ${topRpsZone ? `${topRpsZone[0]} has ${topRpsZone[1]} high-severity issues` : "Review pending"} — consider consolidating repair crews.`
                : `Average RPS of ${avgRps} across ${total} tickets. All within normal operational thresholds.`,
            badge: "Operational Insight",
            badgeClass: "saving",
        },
        {
            title: "Risk Assessment",
            desc: critPct > 15
                ? `${critPct}% of tickets are Critical or High priority. This exceeds the 15% threshold — escalation recommended.`
                : `${critPct}% critical/high priority rate is within acceptable range. System health status: Normal.`,
            badge: critPct > 15 ? "Risk Alert" : "Status Normal",
            badgeClass: critPct > 15 ? "alert" : "saving",
        },
    ];

    return (
        <div className="predictive-section">
            <div className="predictive-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                </svg>
                Predictive AI Insights
            </div>
            <div className="predictive-cards">
                {insights.map((ins, i) => (
                    <div className="predictive-card" key={i}>
                        <h4>{ins.title}</h4>
                        <p>{ins.desc}</p>
                        <span className={`predictive-badge ${ins.badgeClass}`}>{ins.badge}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default PredictiveInsights;
