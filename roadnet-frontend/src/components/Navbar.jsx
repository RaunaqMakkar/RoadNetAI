import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import API from "../services/api";
import UserManagement from "./UserManagement";
import NotificationSettings from "./NotificationSettings";
import AlertThresholds from "./AlertThresholds";
import RoleDefinitions from "./RoleDefinitions";
import "../styles/Settings.css";

const PROFILE_TABS = ["User Management", "Notifications", "Alert Thresholds", "Role-Based Access"];

function Navbar() {
    const location = useLocation();
    const navigate = useNavigate();
    const isDashboard = location.pathname === "/";
    const searchPlaceholder = {
        "/map": "Search locations...",
        "/tickets": "Search tickets...",
        "/analytics": "Search metrics...",
        "/departments": "Search departments...",
        "/inspection": "Search detections...",
    }[location.pathname] || "Search...";
    const [searchQuery, setSearchQuery] = useState("");
    const [bellOpen, setBellOpen] = useState(false);
    const [profileOpen, setProfileOpen] = useState(false);
    const [profileTab, setProfileTab] = useState(null); // null = summary, string = settings tab
    const [users, setUsers] = useState([]);
    const [tickets, setTickets] = useState([]);
    const [stats, setStats] = useState(null);
    const [readNotifs, setReadNotifs] = useState(new Set());

    const bellRef = useRef(null);
    const profileRef = useRef(null);

    // Fetch data on mount
    useEffect(() => {
        API.get("/users")
            .then((res) => setUsers(Array.isArray(res.data) ? res.data : []))
            .catch(() => setUsers([]));
        API.get("/tickets", { params: { page: 1, limit: 10, order: "desc" } })
            .then((res) => setTickets(res.data.data || []))
            .catch(() => setTickets([]));
        API.get("/stats")
            .then((res) => setStats(res.data))
            .catch(() => setStats(null));
    }, []);

    // Close dropdowns on outside click
    useEffect(() => {
        const handler = (e) => {
            if (bellRef.current && !bellRef.current.contains(e.target)) setBellOpen(false);
            if (profileRef.current && !profileRef.current.contains(e.target)) {
                setProfileOpen(false);
                setProfileTab(null);
            }
        };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    // Generate notifications from recent tickets
    const notifications = tickets.slice(0, 6).map((t) => {
        const type = (t.type || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
        const isUrgent = t.priority === "Critical" || t.priority === "High";
        return {
            id: t.ticket_id,
            title: `${type} Detected`,
            desc: `${t.zone || "Unknown zone"} · ${t.assigned_department || "Unassigned"}`,
            priority: t.priority,
            isUrgent,
            time: t.video_timestamp_formatted || "—",
        };
    });

    const urgentCount = notifications.filter((n) => n.isUrgent && !readNotifs.has(n.id)).length;

    const handleBellClick = () => {
        setBellOpen(!bellOpen);
        setProfileOpen(false);
        setProfileTab(null);
    };

    const handleProfileClick = () => {
        setProfileOpen(!profileOpen);
        setProfileTab(null);
        setBellOpen(false);
    };

    const openSettingsTab = (tab) => {
        setProfileTab(tab);
    };

    return (
        <nav className="navbar">
            <a href="/" className="navbar-brand">
                <svg viewBox="0 0 32 32" fill="none">
                    <rect width="32" height="32" rx="8" fill="#2563eb" />
                    <path d="M8 22L14 10L18 18L24 8" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M8 16L12 14L16 20L24 12" stroke="#93c5fd" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                RoadNet.AI
            </a>

            <ul className="navbar-links">
                <li><NavLink to="/" className={({ isActive }) => isActive ? "active" : ""}>Dashboard</NavLink></li>
                <li><NavLink to="/inspection">AI Inspection</NavLink></li>
                <li><NavLink to="/map">Map View</NavLink></li>
                <li><NavLink to="/tickets">Tickets</NavLink></li>
                <li><NavLink to="/analytics">Analytics</NavLink></li>
                <li><NavLink to="/departments">Departments</NavLink></li>
            </ul>

            <div className="navbar-right">
                {/* ===== Search Bar (hidden on Dashboard) ===== */}
                {!isDashboard && (
                    <div className="navbar-search">
                        <svg className="navbar-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="11" cy="11" r="8" />
                            <line x1="21" y1="21" x2="16.65" y2="16.65" />
                        </svg>
                        <input
                            type="text"
                            className="navbar-search-input"
                            placeholder={searchPlaceholder}
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                )}
                {/* ===== Bell / Notifications ===== */}
                <div className="navbar-bell-wrapper" ref={bellRef}>
                    <button className="navbar-bell" onClick={handleBellClick}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                        </svg>
                        {urgentCount > 0 && <span className="bell-badge">{urgentCount}</span>}
                    </button>

                    {bellOpen && (
                        <div className="notif-dropdown">
                            <div className="notif-dropdown-header">
                                <h4>Notifications</h4>
                                <span className="notif-dropdown-count">{notifications.length} recent</span>
                            </div>
                            <div className="notif-dropdown-list">
                                {notifications.length === 0 ? (
                                    <div className="notif-empty">No recent notifications</div>
                                ) : (
                                    notifications.map((n) => {
                                        const isUnreadUrgent = n.isUrgent && !readNotifs.has(n.id);
                                        return (
                                            <div
                                                className={`notif-item ${isUnreadUrgent ? "urgent" : ""}`}
                                                key={n.id}
                                                onClick={() => {
                                                    setReadNotifs((prev) => new Set(prev).add(n.id));
                                                    setBellOpen(false);
                                                    navigate(`/tickets?view=${encodeURIComponent(n.id)}`);
                                                }}
                                            >
                                                <div className={`notif-item-dot ${(n.priority || "").toLowerCase()}`} />
                                                <div className="notif-item-content">
                                                    <div className="notif-item-title">{n.title}</div>
                                                    <div className="notif-item-desc">{n.desc}</div>
                                                </div>
                                                <span className="notif-item-time">{n.time}</span>
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                            <NavLink to="/tickets" className="notif-dropdown-footer" onClick={() => setBellOpen(false)}>
                                View All Tickets →
                            </NavLink>
                        </div>
                    )}
                </div>

                {/* ===== Profile / Settings ===== */}
                <div className="navbar-profile-wrapper" ref={profileRef}>
                    <button className="navbar-avatar-btn" onClick={handleProfileClick} title="Admin User">
                        <div className="navbar-avatar">A</div>
                    </button>

                    {profileOpen && (
                        <div className={`profile-dropdown ${profileTab ? "expanded" : ""}`}>
                            {!profileTab ? (
                                <>
                                    {/* Profile Summary */}
                                    <div className="profile-summary">
                                        <div className="profile-avatar-large">A</div>
                                        <div className="profile-info">
                                            <div className="profile-name">Admin User</div>
                                            <div className="profile-email">admin@roadnet.ai</div>
                                            <div className="profile-role-badge">Administrator</div>
                                        </div>
                                    </div>

                                    <div className="profile-stats-row">
                                        <div className="profile-stat">
                                            <span className="profile-stat-val">{stats?.total_tickets || 0}</span>
                                            <span className="profile-stat-label">Tickets</span>
                                        </div>
                                        <div className="profile-stat">
                                            <span className="profile-stat-val">{stats?.avg_rps_score || 0}</span>
                                            <span className="profile-stat-label">Avg RPS</span>
                                        </div>
                                        <div className="profile-stat">
                                            <span className="profile-stat-val">{users.length}</span>
                                            <span className="profile-stat-label">Users</span>
                                        </div>
                                    </div>

                                    <div className="profile-menu-divider" />

                                    {/* Settings Tabs as menu items */}
                                    <div className="profile-menu">
                                        {PROFILE_TABS.map((tab) => (
                                            <button className="profile-menu-item" key={tab} onClick={() => openSettingsTab(tab)}>
                                                <span className="profile-menu-icon">
                                                    {tab === "User Management" && "👥"}
                                                    {tab === "Notifications" && "🔔"}
                                                    {tab === "Alert Thresholds" && "⚠️"}
                                                    {tab === "Role-Based Access" && "🔒"}
                                                </span>
                                                {tab}
                                                <span className="profile-menu-arrow">›</span>
                                            </button>
                                        ))}
                                    </div>

                                    <div className="profile-menu-divider" />

                                    <button className="profile-menu-item logout" onClick={() => alert("Logged out (demo)")}>
                                        <span className="profile-menu-icon">🚪</span>
                                        Sign Out
                                    </button>
                                </>
                            ) : (
                                <>
                                    {/* Settings Sub-Panel */}
                                    <div className="profile-panel-header">
                                        <button className="profile-back-btn" onClick={() => setProfileTab(null)}>
                                            ← Back
                                        </button>
                                        <h4>{profileTab}</h4>
                                    </div>
                                    <div className="profile-panel-body">
                                        {profileTab === "User Management" && <UserManagement users={users} />}
                                        {profileTab === "Notifications" && <NotificationSettings />}
                                        {profileTab === "Alert Thresholds" && <AlertThresholds />}
                                        {profileTab === "Role-Based Access" && <RoleDefinitions />}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
}

export default Navbar;
