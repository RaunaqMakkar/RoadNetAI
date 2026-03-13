import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function TicketTrendChart({ tickets }) {
    // Group tickets by type to show distribution instead of fake day data
    const typeGroups = {};
    (tickets || []).forEach((t) => {
        const type = (t.type || "unknown").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        if (!typeGroups[type]) typeGroups[type] = { count: 0, totalRps: 0 };
        typeGroups[type].count++;
        typeGroups[type].totalRps += (t.rps_score || 0);
    });

    const data = Object.entries(typeGroups)
        .sort((a, b) => b[1].count - a[1].count)
        .map(([type, info]) => ({
            type: type.length > 14 ? type.slice(0, 12) + "…" : type,
            tickets: info.count,
            avgRps: +(info.totalRps / info.count).toFixed(1),
        }));

    return (
        <div className="dept-trend-panel">
            <div className="dept-trend-header">
                <h3>Issue Type Distribution</h3>
                <div className="dept-trend-legend">
                    <div className="dept-trend-legend-item">
                        <span className="chart-legend-dot" style={{ width: 8, height: 8, borderRadius: "50%", display: "inline-block", background: "#2563eb" }} />
                        Ticket Count
                    </div>
                    <div className="dept-trend-legend-item">
                        <span className="chart-legend-dot" style={{ width: 8, height: 8, borderRadius: "50%", display: "inline-block", background: "#22c55e" }} />
                        Avg RPS
                    </div>
                </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis dataKey="type" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                    <Line type="monotone" dataKey="avgRps" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="tickets" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 4 }} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export default TicketTrendChart;
