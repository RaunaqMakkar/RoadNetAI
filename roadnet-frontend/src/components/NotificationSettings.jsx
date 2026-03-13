import { useState } from "react";

function NotificationSettings() {
    const [email, setEmail] = useState(true);
    const [sms, setSms] = useState(false);
    const [inApp, setInApp] = useState(true);

    return (
        <div className="s-panel notif-panel">
            <div className="notif-panel-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 0 1-3.46 0" /></svg>
                System Notifications
            </div>

            <div className="notif-row">
                <span className="notif-label">Email Alerts</span>
                <label className="s-toggle">
                    <input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} />
                    <span className="s-toggle-slider" />
                </label>
            </div>

            <div className="notif-row">
                <span className="notif-label">SMS Critical Pings</span>
                <label className="s-toggle">
                    <input type="checkbox" checked={sms} onChange={(e) => setSms(e.target.checked)} />
                    <span className="s-toggle-slider" />
                </label>
            </div>

            <div className="notif-row">
                <span className="notif-label">In-App Banner</span>
                <label className="s-toggle">
                    <input type="checkbox" checked={inApp} onChange={(e) => setInApp(e.target.checked)} />
                    <span className="s-toggle-slider" />
                </label>
            </div>
        </div>
    );
}

export default NotificationSettings;
