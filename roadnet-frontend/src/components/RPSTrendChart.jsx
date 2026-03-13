import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

function RPSTrendChart({ tickets, avgRps }) {
    const [view, setView] = useState("type");

    // Group RPS by issue type
    const typeRps = {};
    const typeCounts = {};
    (tickets || []).forEach((t) => {
        const type = (t.type || "unknown").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        typeRps[type] = (typeRps[type] || 0) + (t.rps_score || 0);
        typeCounts[type] = (typeCounts[type] || 0) + 1;
    });

    const typeData = Object.entries(typeRps).map(([name, total]) => ({
        name: name.length > 12 ? name.slice(0, 12) + "…" : name,
        value: +(total / (typeCounts[name] || 1)).toFixed(1),
    })).sort((a, b) => b.value - a.value);

    // Group RPS by priority
    const prioRps = {};
    const prioCounts = {};
    (tickets || []).forEach((t) => {
        const prio = t.priority || "Unknown";
        prioRps[prio] = (prioRps[prio] || 0) + (t.rps_score || 0);
        prioCounts[prio] = (prioCounts[prio] || 0) + 1;
    });

    const prioData = Object.entries(prioRps).map(([name, total]) => ({
        name,
        value: +(total / (prioCounts[name] || 1)).toFixed(1),
    })).sort((a, b) => b.value - a.value);

    const data = view === "type" ? typeData : prioData;
    const maxVal = data.length > 0 ? Math.max(...data.map((d) => d.value)) : 1;

    return (
        <div className="a-panel">
            <div className="a-panel-header">
                <h3>RPS Trend Analysis</h3>
                <div className="rps-toggle">
                    <button className={view === "type" ? "active" : ""} onClick={() => setView("type")}>By Type</button>
                    <button className={view === "priority" ? "active" : ""} onClick={() => setView("priority")}>By Priority</button>
                </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={32}>
                        {data.map((entry, i) => (
                            <Cell key={i} fill={entry.value === maxVal ? "#1d4ed8" : "#93c5fd"} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}

export default RPSTrendChart;
