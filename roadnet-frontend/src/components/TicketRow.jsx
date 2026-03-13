function TicketRow({ ticket, onView, onAssign, onClose }) {
    const t = ticket;
    const priorityClass = (t.priority || "").toLowerCase();
    const statusRaw = (t.status || "").toLowerCase();
    const statusClass = statusRaw === "in progress" ? "in-progress" : statusRaw;
    const typeLabel = (t.type || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    return (
        <tr>
            <td><a className="ticket-id-link" href="#" onClick={(e) => { e.preventDefault(); onView(t); }}>#{t.ticket_id}</a></td>
            <td className="ticket-type-cell">{typeLabel}</td>
            <td><span className={`priority-badge ${priorityClass}`}>{t.priority}</span></td>
            <td className="rps-cell">{t.rps_score ?? "—"}</td>
            <td>{t.zone || "—"}</td>
            <td>{t.assigned_department || "—"}</td>
            <td><span className={`status-badge ${statusClass}`}>{t.status}</span></td>
            <td className="timestamp-cell">{t.video_timestamp_formatted || "—"}</td>
            <td className="source-cell">
                <span className="source-badge">
                    {t.source ? t.source.replace('_', ' ') : "MANUAL"}
                </span>
            </td>
            <td>
                <div className="ticket-actions">
                    <button className="action-btn" title="View" onClick={() => onView(t)}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
                    </button>
                    <button className="action-btn" title="Assign" onClick={() => onAssign(t)}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" /><circle cx="8.5" cy="7" r="4" /><line x1="20" y1="8" x2="20" y2="14" /><line x1="23" y1="11" x2="17" y2="11" /></svg>
                    </button>
                    <button className="action-btn" title="Close" onClick={() => onClose(t)}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
                    </button>
                </div>
            </td>
        </tr>
    );
}

export default TicketRow;
