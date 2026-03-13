function DepartmentCards({ tickets, stats }) {
    // Build department data from real ticket data
    const deptData = {};

    (tickets || []).forEach((t) => {
        const dept = t.assigned_department || "Unassigned";
        if (!deptData[dept]) {
            deptData[dept] = { open: 0, closed: 0, inProgress: 0, total: 0, totalRps: 0 };
        }
        deptData[dept].total++;
        deptData[dept].totalRps += (t.rps_score || 0);
        const s = (t.status || "").toLowerCase();
        if (s === "closed") deptData[dept].closed++;
        else if (s === "in progress") deptData[dept].inProgress++;
        else deptData[dept].open++;
    });

    const ICONS = ["🏗️", "🔧", "🛡️", "🚧", "📡", "⚙️"];
    const ICON_CLASSES = ["blue", "orange", "green", "purple", "blue", "orange"];

    const departments = Object.entries(deptData)
        .sort((a, b) => b[1].total - a[1].total)
        .map(([name, data], i) => ({
            name,
            ...data,
            icon: ICONS[i % ICONS.length],
            iconClass: ICON_CLASSES[i % ICON_CLASSES.length],
            avgRps: data.total > 0 ? (data.totalRps / data.total).toFixed(1) : 0,
        }));

    if (departments.length === 0) {
        return <div className="dept-cards-grid"><div style={{ color: "#94a3b8", padding: 30 }}>No department data</div></div>;
    }

    return (
        <div className="dept-cards-grid">
            {departments.map((d) => {
                const closureRate = d.total > 0 ? Math.round((d.closed / d.total) * 100) : 0;
                const openPct = d.total > 0 ? Math.round((d.open / d.total) * 100) : 0;

                return (
                    <div className="dept-card" key={d.name}>
                        <div className="dept-card-top">
                            <div className={`dept-card-icon ${d.iconClass}`}>{d.icon}</div>
                            <span className={`dept-sla-badge ${closureRate >= 50 ? "green" : "yellow"}`}>
                                {closureRate}% Closed
                            </span>
                        </div>
                        <h3>{d.name}</h3>
                        <div className="dept-card-desc">
                            {d.total} tickets · Avg RPS: {d.avgRps}
                        </div>

                        <div className="dept-card-open">
                            <span>Open Tickets</span>
                            <span>{d.open}</span>
                        </div>
                        <div className="dept-progress-bar">
                            <div className={`dept-progress-fill ${d.iconClass}`} style={{ width: `${Math.min(openPct, 100)}%` }} />
                        </div>
                        <div className="dept-card-counts">
                            <span>Closed: {d.closed}</span>
                            <span>In Progress: {d.inProgress}</span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

export default DepartmentCards;
