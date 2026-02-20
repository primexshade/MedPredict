import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'

export default function Login() {
    const navigate = useNavigate()
    const [email, setEmail] = useState('admin@example.com')
    const [password, setPassword] = useState('admin')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            const { data } = await authAPI.login(email, password)
            localStorage.setItem('access_token', data.access_token)
            localStorage.setItem('refresh_token', data.refresh_token)
            navigate('/')
        } catch {
            setError('Invalid credentials. Try admin@example.com / admin')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={styles.page}>
            <div style={styles.card}>
                {/* Brand */}
                <div style={styles.brand}>
                    <div style={styles.brandIcon}>✚</div>
                    <span style={styles.brandName}>MedPredict</span>
                </div>
                <h2 style={styles.title}>Sign in to your account</h2>
                <p style={styles.sub}>AI-powered disease risk analysis platform</p>

                <form onSubmit={handleSubmit} style={styles.form}>
                    <div className="form-group">
                        <label className="form-label">Email</label>
                        <input
                            id="email"
                            type="email"
                            className="form-input"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            autoComplete="email"
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Password</label>
                        <input
                            id="password"
                            type="password"
                            className="form-input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            autoComplete="current-password"
                        />
                    </div>

                    {error && <p style={styles.errorMsg}>{error}</p>}

                    <button
                        id="login-btn"
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading}
                        style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
                    >
                        {loading ? <span className="spinner" /> : 'Sign in →'}
                    </button>
                </form>

                <p style={styles.hint}>Demo: admin@example.com / admin</p>
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    page: {
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: `radial-gradient(ellipse at 50% 0%, rgba(0,212,200,0.08) 0%, transparent 60%), #06080f`,
        padding: 20,
    },
    card: {
        background: 'rgba(13,17,23,0.92)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 20,
        padding: '40px 36px',
        width: '100%',
        maxWidth: 400,
        backdropFilter: 'blur(20px)',
        boxShadow: '0 24px 80px rgba(0,0,0,0.5)',
    },
    brand: {
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 24,
    },
    brandIcon: {
        width: 40, height: 40,
        background: 'linear-gradient(135deg, #00d4c8, #0066ff)',
        borderRadius: 12,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '1.25rem', color: '#000', fontWeight: 700,
    },
    brandName: {
        fontSize: '1.1rem',
        fontWeight: 700,
        color: '#f0f4f8',
    },
    title: {
        fontSize: '1.4rem',
        fontWeight: 700,
        color: '#f0f4f8',
        marginBottom: 6,
    },
    sub: { color: '#8b9ab5', fontSize: '0.875rem', marginBottom: 28 },
    form: { display: 'flex', flexDirection: 'column', gap: 16 },
    errorMsg: {
        color: '#ef4444',
        fontSize: '0.8125rem',
        background: 'rgba(239,68,68,0.1)',
        border: '1px solid rgba(239,68,68,0.2)',
        borderRadius: 8,
        padding: '8px 12px',
    },
    hint: {
        color: '#4a5568',
        fontSize: '0.75rem',
        textAlign: 'center',
        marginTop: 20,
    },
}
