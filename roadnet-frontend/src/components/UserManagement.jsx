const DEFAULT_USERS = [
    { name: "John Doe", email: "john@roadnet.ai", role: "Admin", status: "Active" },
    { name: "Sarah Smith", email: "sarah@roadnet.ai", role: "Dispatcher", status: "Active" },
    { name: "Mike Jones", email: "mike@roadnet.ai", role: "Viewer", status: "Pending" },
];

function UserManagement({ users }) {
    const list = users && users.length > 0 ? users : DEFAULT_USERS;

    return (
        <div>
            <div className="users-header">
                <h2>Active Users</h2>
                <button className="btn-invite">+ Invite User</button>
            </div>

            <div className="s-panel">
                <table className="users-table">
                    <thead>
                        <tr>
                            <th>NAME & EMAIL</th>
                            <th>ROLE</th>
                            <th>STATUS</th>
                            <th>ACTIONS</th>
                        </tr>
                    </thead>
                    <tbody>
                        {list.map((u, i) => {
                            const roleClass = (u.role || "").toLowerCase();
                            const statusLower = (u.status || "active").toLowerCase();
                            return (
                                <tr key={i}>
                                    <td>
                                        <div className="user-name">{u.name}</div>
                                        <div className="user-email">{u.email}</div>
                                    </td>
                                    <td><span className={`role-badge ${roleClass}`}>{u.role}</span></td>
                                    <td>
                                        <div className={`user-status ${statusLower}`}>
                                            <span className={`user-status-dot ${statusLower}`} />
                                            {u.status}
                                        </div>
                                    </td>
                                    <td>
                                        <button className="user-edit-btn" title="Edit">
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default UserManagement;
