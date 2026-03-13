function ResourceAllocation({ tickets, stats }) {
    const total = (tickets || []).length;
    const openCount = (tickets || []).filter((t) => (t.status || "").toLowerCase() !== "closed").length;
    const avgRps = stats?.avg_rps_score || 0;
    const maxRps = stats?.max_rps_score || 0;

    // Compute tickets per department
    const deptCounts = {};
    (tickets || []).forEach((t) => {
        const d = t.assigned_department || "Unassigned";
        deptCounts[d] = (deptCounts[d] || 0) + 1;
    });
    const deptCount = Object.keys(deptCounts).length;

    // Utilization = open tickets / total
    const utilization = total > 0 ? Math.round((openCount / total) * 100) : 0;

    return (
        <>
            <div className="resource-stat">
                <div className="resource-stat-icon blue">📋</div>
                <div className="resource-stat-content">
                    <div className="resource-stat-label">Active Tickets</div>
                    <div className="resource-stat-value">{openCount} of {total}</div>
                </div>
                <span className={`resource-stat-extra ${utilization > 80 ? "red" : "green"}`}>{utilization}% load</span>
            </div>
            <div className="resource-stat">
                <div className="resource-stat-icon orange">📊</div>
                <div className="resource-stat-content">
                    <div className="resource-stat-label">Avg RPS Score</div>
                    <div className="resource-stat-value">{avgRps}</div>
                </div>
                <span className="resource-stat-extra muted">Max: {maxRps}</span>
            </div>
            <div className="resource-stat">
                <div className="resource-stat-icon green">🏢</div>
                <div className="resource-stat-content">
                    <div className="resource-stat-label">Departments</div>
                    <div className="resource-stat-value">{deptCount} Active</div>
                </div>
                <span className="resource-stat-extra muted">{total > 0 ? Math.round(total / deptCount) : 0} avg/dept</span>
            </div>
        </>
    );
}

export default ResourceAllocation;
