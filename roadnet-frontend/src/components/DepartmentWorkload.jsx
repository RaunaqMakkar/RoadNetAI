function DepartmentWorkload({ deptBreakdown, total }) {
    const depts = Object.entries(deptBreakdown || {})
        .map(([name, count]) => {
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;
            let status, color;
            if (pct >= 70) { status = "High Demand"; color = "red"; }
            else if (pct >= 40) { status = "Near Capacity"; color = "orange"; }
            else if (pct >= 15) { status = "Optimized"; color = "green"; }
            else { status = "Idle Capacity"; color = "blue"; }
            return { name, count, pct, status, color };
        })
        .sort((a, b) => b.count - a.count);

    if (depts.length === 0) {
        return (
            <div className="a-panel">
                <div className="a-panel-header"><h3>Department Workload</h3></div>
                <div style={{ textAlign: "center", padding: 40, color: "#94a3b8" }}>No department data</div>
            </div>
        );
    }

    return (
        <div className="a-panel">
            <div className="a-panel-header"><h3>Department Workload</h3></div>
            <div className="dept-grid">
                {depts.map((d) => (
                    <div className="dept-card" key={d.name}>
                        <div className="dept-card-name">{d.name}</div>
                        <div className="dept-card-pct">{d.pct}%</div>
                        <div className={`dept-card-status ${d.color}`}>{d.status}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default DepartmentWorkload;
