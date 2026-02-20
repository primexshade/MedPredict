import { NavLink, useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'

const NAV_ITEMS = [
    { label: 'Dashboard', to: '/', icon: '⬡' },
    { label: 'Predict', to: '/predict', icon: '⚕' },
    { label: 'Analytics', to: '/analytics', icon: '◈' },
    { label: 'Patients', to: '/patients', icon: '⊕' },
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
                <span style={styles.brandText}>MedPredict</span>
            </div>

            {/* Nav links */}
            <div style={styles.links}>
                {NAV_ITEMS.map(({ label, to, icon }) => (
                    <NavLink
                        key={to}
                        to={to}
                        end={to === '/'}
                        style={({ isActive }) => ({
                            ...styles.link,
                            ...(isActive ? styles.activeLink : {}),
                        })}
                    >
                        <span style={{ fontSize: '1rem' }}>{icon}</span>
                        {label}
                    </NavLink>
                ))}
            </div>

            {/* Logout */}
            <button onClick={handleLogout} style={styles.logout}>
                ↪ Logout
            </button>
        </nav>
    )
}

const styles: Record<string, React.CSSProperties> = {
    nav: {
        position: 'fixed',
        top: 0, left: 0, bottom: 0,
        width: 240,
        background: 'rgba(13, 17, 23, 0.95)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(12px)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px',
        zIndex: 100,
    },
    brand: {
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 40,
        padding: '0 8px',
    },
    brandIcon: {
        width: 36, height: 36,
        background: 'linear-gradient(135deg, #00d4c8, #0066ff)',
        borderRadius: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '1.1rem', fontWeight: 700, color: '#000',
    },
    brandText: {
        fontSize: '1.05rem',
        fontWeight: 700,
        color: '#f0f4f8',
        letterSpacing: '-0.02em',
    },
    links: {
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        flex: 1,
    },
    link: {
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 12px',
        borderRadius: 8,
        color: '#8b9ab5',
        fontSize: '0.875rem',
        fontWeight: 500,
        textDecoration: 'none',
        transition: 'all 180ms ease',
    },
    activeLink: {
        background: 'rgba(0, 212, 200, 0.12)',
        color: '#00d4c8',
    },
    logout: {
        background: 'transparent',
        border: '1px solid rgba(255,255,255,0.06)',
        color: '#8b9ab5',
        borderRadius: 8,
        padding: '10px 12px',
        fontSize: '0.8125rem',
        cursor: 'pointer',
        fontFamily: 'inherit',
        textAlign: 'left',
        transition: 'all 180ms ease',
    },
}
