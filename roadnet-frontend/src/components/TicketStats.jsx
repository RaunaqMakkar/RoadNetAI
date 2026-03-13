function TicketStats({ stats }) {
    const total = stats?.total_tickets || 0;
    const priorityBreakdown = stats?.breakdown_by_priority || {};
    const statusBreakdown = stats?.breakdown_by_status || {};
    const avgRps = stats?.avg_rps_score || 0;

    const critical = priorityBreakdown["Critical"] || 0;
    const high = priorityBreakdown["High"] || 0;
    const openCount = statusBreakdown["Open"] || 0;
    const inProgress = statusBreakdown["In Progress"] || 0;
    const closed = statusBreakdown["Closed"] || 0;
    const activeCount = openCount + inProgress;

    return (
        <div className="tickets-stats">
            <div className="ticket-stat-card">
                <div className="ticket-stat-title">TOTAL TICKETS</div>
                <div className="ticket-stat-value">{total.toLocaleString()}</div>
            </div>
            <div className="ticket-stat-card">
                <div className="ticket-stat-title">CRITICAL ISSUES</div>
                <div className="ticket-stat-value red">{(critical + high).toLocaleString()}</div>
            </div>
            <div className="ticket-stat-card">
                <div className="ticket-stat-title">OPEN / ACTIVE</div>
                <div className="ticket-stat-value">{activeCount.toLocaleString()}</div>
            </div>
            <div className="ticket-stat-card">
                <div className="ticket-stat-title">AVG RPS SCORE</div>
                <div className="ticket-stat-value green">{avgRps}</div>
            </div>
        </div>
    );
}

export default TicketStats;
