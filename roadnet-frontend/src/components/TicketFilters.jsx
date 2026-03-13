import { useState, useRef, useEffect } from "react";

function TicketFilters({
    search, onSearchChange,
    priority, onPriorityChange,
    status, onStatusChange,
    sortBy, onSortChange,
    order, onOrderChange,
}) {
    const [showPriority, setShowPriority] = useState(false);
    const [showStatus, setShowStatus] = useState(false);
    const [showSort, setShowSort] = useState(false);
    const prioRef = useRef(null);
    const statRef = useRef(null);
    const sortRef = useRef(null);

    useEffect(() => {
        function handleClick(e) {
            if (prioRef.current && !prioRef.current.contains(e.target)) setShowPriority(false);
            if (statRef.current && !statRef.current.contains(e.target)) setShowStatus(false);
            if (sortRef.current && !sortRef.current.contains(e.target)) setShowSort(false);
        }
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, []);

    const priorities = ["Critical", "High", "Moderate", "Low"];
    const statuses = ["Open", "In Progress", "Closed"];
    const sortOptions = [
        { value: "created_at", label: "Date Created" },
        { value: "rps_score", label: "RPS Score" },
        { value: "priority", label: "Priority" },
        { value: "type", label: "Issue Type" },
        { value: "status", label: "Status" },
        { value: "updated_at", label: "Last Updated" },
    ];

    const closeAll = () => { setShowPriority(false); setShowStatus(false); setShowSort(false); };
    const currentSortLabel = sortOptions.find((s) => s.value === sortBy)?.label || "Date Created";

    return (
        <div className="tickets-toolbar">
            <div className="tickets-search-wrapper">
                <svg className="tickets-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <input
                    className="tickets-search"
                    type="text"
                    placeholder="Search Ticket ID, Department, or Zone..."
                    value={search}
                    onChange={(e) => onSearchChange(e.target.value)}
                />
            </div>

            {/* Priority dropdown */}
            <div className="filter-dropdown-wrapper" ref={prioRef}>
                <button
                    className={`filter-dropdown-btn${priority ? " active" : ""}`}
                    onClick={() => { setShowPriority(!showPriority); setShowStatus(false); setShowSort(false); }}
                >
                    Priority
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>
                </button>
                {showPriority && (
                    <div className="filter-dropdown-menu">
                        <button className={`filter-dropdown-item${!priority ? " selected" : ""}`} onClick={() => { onPriorityChange(""); setShowPriority(false); }}>All</button>
                        {priorities.map((p) => (
                            <button key={p} className={`filter-dropdown-item${priority === p ? " selected" : ""}`} onClick={() => { onPriorityChange(p); setShowPriority(false); }}>{p}</button>
                        ))}
                    </div>
                )}
            </div>

            {/* Status dropdown */}
            <div className="filter-dropdown-wrapper" ref={statRef}>
                <button
                    className={`filter-dropdown-btn${status ? " active" : ""}`}
                    onClick={() => { setShowStatus(!showStatus); setShowPriority(false); setShowSort(false); }}
                >
                    Status
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>
                </button>
                {showStatus && (
                    <div className="filter-dropdown-menu">
                        <button className={`filter-dropdown-item${!status ? " selected" : ""}`} onClick={() => { onStatusChange(""); setShowStatus(false); }}>All</button>
                        {statuses.map((s) => (
                            <button key={s} className={`filter-dropdown-item${status === s ? " selected" : ""}`} onClick={() => { onStatusChange(s); setShowStatus(false); }}>{s}</button>
                        ))}
                    </div>
                )}
            </div>

            {/* Sort By dropdown */}
            <div className="filter-dropdown-wrapper" ref={sortRef}>
                <button
                    className={`filter-dropdown-btn${sortBy !== "created_at" ? " active" : ""}`}
                    onClick={() => { setShowSort(!showSort); setShowPriority(false); setShowStatus(false); }}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 14, height: 14 }}>
                        <line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="12" x2="16" y2="12" /><line x1="4" y1="18" x2="12" y2="18" />
                    </svg>
                    Sort: {currentSortLabel}
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9" /></svg>
                </button>
                {showSort && (
                    <div className="filter-dropdown-menu sort-menu">
                        <div className="sort-menu-section-label">Sort By</div>
                        {sortOptions.map((opt) => (
                            <button
                                key={opt.value}
                                className={`filter-dropdown-item${sortBy === opt.value ? " selected" : ""}`}
                                onClick={() => { onSortChange(opt.value); setShowSort(false); }}
                            >
                                {opt.label}
                            </button>
                        ))}
                        <div className="sort-menu-divider" />
                        <div className="sort-menu-section-label">Order</div>
                        <button
                            className={`filter-dropdown-item${order === "desc" ? " selected" : ""}`}
                            onClick={() => { onOrderChange("desc"); setShowSort(false); }}
                        >
                            ↓ Newest / Highest First
                        </button>
                        <button
                            className={`filter-dropdown-item${order === "asc" ? " selected" : ""}`}
                            onClick={() => { onOrderChange("asc"); setShowSort(false); }}
                        >
                            ↑ Oldest / Lowest First
                        </button>
                    </div>
                )}
            </div>

            {/* Export */}
            <button className="filter-dropdown-btn">
                Export
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
            </button>
        </div>
    );
}

export default TicketFilters;
