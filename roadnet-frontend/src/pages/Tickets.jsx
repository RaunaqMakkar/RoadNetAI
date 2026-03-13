import { useEffect, useState, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import API from "../services/api";
import TicketStats from "../components/TicketStats";
import TicketFilters from "../components/TicketFilters";
import TicketTable from "../components/TicketTable";
import Pagination from "../components/Pagination";
import "../styles/Tickets.css";

const LIMIT = 10;

function Tickets() {
    const [data, setData] = useState([]);
    const [total, setTotal] = useState(0);
    const [totalPages, setTotalPages] = useState(0);
    const [page, setPage] = useState(1);
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState(null);

    const [search, setSearch] = useState("");
    const [priority, setPriority] = useState("");
    const [status, setStatus] = useState("");
    const [sortBy, setSortBy] = useState("created_at");
    const [order, setOrder] = useState("desc");

    // Modal state
    const [viewTicket, setViewTicket] = useState(null);
    const [assignTicket, setAssignTicket] = useState(null);
    const [manualOpen, setManualOpen] = useState(false);

    // Fetch overall stats (once)
    useEffect(() => {
        API.get("/stats")
            .then((res) => setStats(res.data))
            .catch(() => setStats(null));
    }, []);

    const fetchTickets = useCallback(() => {
        setLoading(true);
        const params = { page, limit: LIMIT, sort_by: sortBy, order };
        if (priority) params.priority = priority;
        if (status) params.status = status;

        API.get("/tickets", { params })
            .then((res) => {
                setData(res.data.data || []);
                setTotal(res.data.total || 0);
                setTotalPages(res.data.total_pages || 0);
            })
            .catch(() => {
                setData([]);
                setTotal(0);
                setTotalPages(0);
            })
            .finally(() => setLoading(false));
    }, [page, priority, status, sortBy, order]);

    useEffect(() => {
        fetchTickets();
    }, [fetchTickets]);

    // Auto-open ticket detail from ?view=TICKET_ID (e.g. from notification click)
    const [searchParams, setSearchParams] = useSearchParams();
    useEffect(() => {
        const viewId = searchParams.get("view");
        if (viewId && data.length > 0) {
            const found = data.find((t) => t.ticket_id === viewId);
            if (found) {
                setViewTicket(found);
                setSearchParams({}, { replace: true });
            } else {
                // Ticket not on current page — fetch it directly
                API.get("/tickets", { params: { page: 1, limit: 100 } })
                    .then((res) => {
                        const all = res.data.data || [];
                        const t = all.find((x) => x.ticket_id === viewId);
                        if (t) setViewTicket(t);
                    })
                    .catch(() => { })
                    .finally(() => setSearchParams({}, { replace: true }));
            }
        }
    }, [searchParams, data]);

    const handlePriorityChange = (val) => {
        setPriority(val);
        setPage(1);
    };

    const handleStatusChange = (val) => {
        setStatus(val);
        setPage(1);
    };

    const handleSortChange = (val) => {
        setSortBy(val);
        setPage(1);
    };

    const handleOrderChange = (val) => {
        setOrder(val);
        setPage(1);
    };

    const filteredData = search.trim()
        ? data.filter((t) => {
            const q = search.toLowerCase();
            return (
                (t.ticket_id || "").toLowerCase().includes(q) ||
                (t.assigned_department || "").toLowerCase().includes(q) ||
                (t.zone || "").toLowerCase().includes(q)
            );
        })
        : data;

    const start = (page - 1) * LIMIT + 1;
    const end = Math.min(page * LIMIT, total);

    // Action handlers
    const handleView = (ticket) => setViewTicket(ticket);
    const handleAssign = (ticket) => setAssignTicket(ticket);
    const handleClose = (ticket) => {
        if (window.confirm(`Close ticket #${ticket.ticket_id}? This will mark it as closed.`)) {
            // In a real app this would call PATCH /tickets/:id
            alert(`Ticket #${ticket.ticket_id} has been marked as Closed.`);
            fetchTickets();
        }
    };

    return (
        <div className="tickets-page">
            {/* Header */}
            <div className="tickets-header">
                <div className="tickets-header-text">
                    <h1>Tickets Management</h1>
                    <p>Monitor infrastructure issues, track RPS scores, and manage repairs.</p>
                </div>
                <button className="btn-manual-ticket" onClick={() => setManualOpen(true)}>+ Manual Ticket</button>
            </div>

            {/* Stats — from /stats (overall, not per-page) */}
            <TicketStats stats={stats} />

            {/* Table Card */}
            <div className="tickets-table-container">
                <TicketFilters
                    search={search}
                    onSearchChange={setSearch}
                    priority={priority}
                    onPriorityChange={handlePriorityChange}
                    status={status}
                    onStatusChange={handleStatusChange}
                    sortBy={sortBy}
                    onSortChange={handleSortChange}
                    order={order}
                    onOrderChange={handleOrderChange}
                />

                {loading ? (
                    <div className="loading-spinner" style={{ padding: "60px 0" }}>
                        <div className="spinner" />
                        Loading tickets...
                    </div>
                ) : (
                    <TicketTable
                        tickets={filteredData}
                        onView={handleView}
                        onAssign={handleAssign}
                        onClose={handleClose}
                    />
                )}

                {/* Footer */}
                <div className="tickets-footer">
                    <span className="tickets-showing">
                        {total > 0 ? `Showing ${start}-${end} of ${total.toLocaleString()} tickets` : "No tickets"}
                    </span>
                    <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
                </div>
            </div>

            <div className="tickets-page-footer">
                © 2026 RoadNet.AI Infrastructure Monitoring. All rights reserved.
            </div>

            {/* ===== View Ticket Modal ===== */}
            {viewTicket && (
                <div className="modal-overlay" onClick={() => setViewTicket(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>Ticket #{viewTicket.ticket_id}</h2>
                            <button className="modal-close" onClick={() => setViewTicket(null)}>✕</button>
                        </div>
                        <div className="modal-body">
                            {/* Detection Frame Image */}
                            {viewTicket.image_url && (
                                <div className="modal-frame-preview">
                                    <label>AI Detection Frame</label>
                                    <div className="modal-frame-img-wrapper">
                                        <img
                                            src={`http://127.0.0.1:8000${viewTicket.image_url}`}
                                            alt={`Detection frame ${viewTicket.frame_id || ""}`}
                                        />
                                        {viewTicket.frame_id && (
                                            <span className="modal-frame-id">{viewTicket.frame_id}</span>
                                        )}
                                        {viewTicket.avg_confidence != null && (
                                            <span className="modal-frame-conf">
                                                {(viewTicket.avg_confidence * 100).toFixed(1)}% confidence
                                            </span>
                                        )}
                                    </div>
                                </div>
                            )}

                            <div className="modal-field"><label>Type</label><span>{(viewTicket.type || "").replace(/_/g, " ")}</span></div>
                            <div className="modal-field"><label>Priority</label><span className={`priority-badge ${(viewTicket.priority || "").toLowerCase()}`}>{viewTicket.priority}</span></div>
                            <div className="modal-field"><label>Status</label><span>{viewTicket.status}</span></div>
                            <div className="modal-field"><label>RPS Score</label><span>{viewTicket.rps_score ?? "—"}</span></div>
                            <div className="modal-field"><label>Zone</label><span>{viewTicket.zone || "—"}</span></div>
                            <div className="modal-field"><label>Department</label><span>{viewTicket.assigned_department || "—"}</span></div>
                            <div className="modal-field"><label>Assigned To</label><span>{viewTicket.assigned_to || "Unassigned"}</span></div>
                            <div className="modal-field"><label>Timestamp</label><span>{viewTicket.video_timestamp_formatted || "—"}</span></div>
                            <div className="modal-field"><label>Source</label><span>{viewTicket.source || "—"}</span></div>
                            {viewTicket.recommended_action && (
                                <div className="modal-field"><label>Recommended Action</label><span>{viewTicket.recommended_action}</span></div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* ===== Assign Ticket Modal ===== */}
            {assignTicket && (
                <div className="modal-overlay" onClick={() => setAssignTicket(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>Assign #{assignTicket.ticket_id}</h2>
                            <button className="modal-close" onClick={() => setAssignTicket(null)}>✕</button>
                        </div>
                        <div className="modal-body">
                            <div className="modal-field">
                                <label>Current Dept</label>
                                <span>{assignTicket.assigned_department || "None"}</span>
                            </div>
                            <div className="modal-field">
                                <label>Reassign To</label>
                                <select className="modal-select" defaultValue="">
                                    <option value="" disabled>Select department...</option>
                                    <option value="Public Works Department">Public Works Department</option>
                                    <option value="Transport">Transport</option>
                                    <option value="Utilities">Utilities</option>
                                    <option value="Municipal Services">Municipal Services</option>
                                </select>
                            </div>
                            <div className="modal-field">
                                <label>Assign To</label>
                                <input className="modal-input" type="text" placeholder="Enter team member name..." />
                            </div>
                            <button className="modal-submit" onClick={() => {
                                alert(`Ticket #${assignTicket.ticket_id} assignment updated.`);
                                setAssignTicket(null);
                            }}>Confirm Assignment</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ===== Manual Ticket Modal ===== */}
            {manualOpen && (
                <div className="modal-overlay" onClick={() => setManualOpen(false)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>Create Manual Ticket</h2>
                            <button className="modal-close" onClick={() => setManualOpen(false)}>✕</button>
                        </div>
                        <div className="modal-body">
                            <div className="modal-field">
                                <label>Issue Type</label>
                                <select className="modal-select" defaultValue="">
                                    <option value="" disabled>Select type...</option>
                                    <option value="pothole">Pothole</option>
                                    <option value="road_crack">Road Crack</option>
                                    <option value="manhole">Manhole</option>
                                    <option value="open_manhole">Open Manhole</option>
                                </select>
                            </div>
                            <div className="modal-field">
                                <label>Priority</label>
                                <select className="modal-select" defaultValue="">
                                    <option value="" disabled>Select priority...</option>
                                    <option value="Critical">Critical</option>
                                    <option value="High">High</option>
                                    <option value="Moderate">Moderate</option>
                                    <option value="Low">Low</option>
                                </select>
                            </div>
                            <div className="modal-field">
                                <label>Zone</label>
                                <input className="modal-input" type="text" placeholder="Enter zone (e.g., Ward 1)..." />
                            </div>
                            <div className="modal-field">
                                <label>Description</label>
                                <textarea className="modal-textarea" rows={3} placeholder="Describe the issue..." />
                            </div>
                            <div className="modal-field">
                                <label>Assign Dept</label>
                                <select className="modal-select" defaultValue="">
                                    <option value="" disabled>Select department...</option>
                                    <option value="Public Works Department">Public Works</option>
                                    <option value="Transport">Transport</option>
                                    <option value="Utilities">Utilities</option>
                                </select>
                            </div>
                            <button className="modal-submit" onClick={() => {
                                alert("Manual ticket created successfully!");
                                setManualOpen(false);
                            }}>Create Ticket</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Tickets;
