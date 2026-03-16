import { NavLink, useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'

const NAV_ITEMS = [
    { label: 'Dashboard', to: '/', icon: '⬡', desc: 'Overview' },
    { label: 'Predict', to: '/predict', icon: '⚕', desc: '4 diseases' },
    { label: 'Analytics', to: '/analytics', icon: '◈', desc: 'Insights' },
    { label: 'Patients', to: '/patients', icon: '⊕', desc: 'Registry' },
]

export default function Navbar() {
    const navigate = useNavigate()

    const handleLogout = async () => {
        try { await authAPI.logout() } catch { }
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        navigate('/login')
    }

    return (
        <nav style={styles.nav}>
            {/* Brand */}
            <div style={styles.brand}>
                <div style={styles.brandIcon}>✚</div>
                <div>
                    <div style={styles.brandText}>MedPredict</div>
                    <div style={{ fontSize: '0.65rem', color: '#4a5568', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Clinical AI</div>
                </div>
            </div>

            {/* Divider */}
            <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '0 0 16px 0' }} />

            {/* Nav links */}
            <div style={styles.links}>
                <div style={{ fontSize: '0.65rem', color: '#4a5568', letterSpacing: '0.1em', textTransform: 'uppercase', padding: '0 12px', marginBottom: 6 }}>
                    Navigation
                </div>
                {NAV_ITEMS.map(({ label, to, icon, desc }) => (
                    <NavLink
                        key={to}
                        to={to}
                        end={to === '/'}
                        style={({ isActive }) => ({
                            ...styles.link,
                            ...(isActive ? styles.activeLink : {}),
                        })}
                    >
                        <div style={{
                            width: 32, height: 32,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            borderRadius: 8, fontSize: '1rem',
                            background: 'rgba(255,255,255,0.04)',
                            flexShrink: 0,
                        }}>{icon}</div>
                        <div>
                            <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>{label}</div>
                            <div style={{ fontSize: '0.65rem', color: 'inherit', opacity: 0.6 }}>{desc}</div>
                        </div>
                    </NavLink>
                ))}
            </div>

            {/* System status */}
            <div style={{
                margin: '16px 0',
                padding: '10px 12px',
                background: 'rgba(34,197,94,0.06)',
                border: '1px solid rgba(34,197,94,0.15)',
                borderRadius: 10,
                fontSize: '0.7rem',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#22c55e', marginBottom: 4, fontWeight: 600 }}>
                    <span style={{
                        width: 6, height: 6, borderRadius: '50%', background: '#22c55e',
                        boxShadow: '0 0 6px #22c55e', display: 'inline-block',
                        animation: 'pulse 2s infinite',
                    }} />
                    System Status
                </div>
                <div style={{ color: '#8b9ab5', lineHeight: 1.6 }}>
                    <div>API ✓  Models ✓</div>
                    <div>4 models loaded</div>
                </div>
            </div>

            {/* User and logout */}
            <div style={{
                padding: '10px 12px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 10,
                marginBottom: 8,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
            }}>
                <div style={{
                    width: 30, height: 30, borderRadius: '50%',
                    background: 'linear-gradient(135deg, #00d4c8, #6366f1)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.75rem', fontWeight: 700, color: '#000', flexShrink: 0,
                }}>A</div>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#f0f4f8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Admin</div>
                    <div style={{ fontSize: '0.65rem', color: '#4a5568' }}>Clinician</div>
                </div>
            </div>

            <button onClick={handleLogout} style={styles.logout}>
                ↪ Sign out
            </button>
        </nav>
    )
}

const styles: Record<string, React.CSSProperties> = {
    nav: {
        position: 'fixed',
        top: 0, left: 0, bottom: 0,
        width: 240,
        background: 'rgba(9, 12, 18, 0.97)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(16px)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px 20px',
        zIndex: 100,
    },
    brand: {
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 20,
        padding: '0 4px',
    },
    brandIcon: {
        width: 38, height: 38,
        background: 'linear-gradient(135deg, #00d4c8, #0066ff)',
        borderRadius: 11,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '1.1rem', fontWeight: 700, color: '#000',
        boxShadow: '0 0 16px rgba(0,212,200,0.25)',
        flexShrink: 0,
    },
    brandText: {
        fontSize: '1rem',
        fontWeight: 700,
        color: '#f0f4f8',
        letterSpacing: '-0.02em',
    },
    links: {
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        flex: 1,
    },
    link: {
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 10px',
        borderRadius: 10,
        color: '#8b9ab5',
        fontSize: '0.875rem',
        textDecoration: 'none',
        transition: 'all 180ms ease',
        marginBottom: 2,
    },
    activeLink: {
        background: 'rgba(0, 212, 200, 0.10)',
        color: '#00d4c8',
        borderLeft: '2px solid #00d4c8',
        paddingLeft: 8,
    },
    logout: {
        background: 'transparent',
        border: '1px solid rgba(255,255,255,0.06)',
        color: '#8b9ab5',
        borderRadius: 8,
        padding: '9px 12px',
        fontSize: '0.8125rem',
        cursor: 'pointer',
        fontFamily: 'inherit',
        textAlign: 'left',
        transition: 'all 180ms ease',
        width: '100%',
    },
}
