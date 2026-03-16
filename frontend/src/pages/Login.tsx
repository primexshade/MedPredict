import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'

// Floating particle background
function Particles() {
    return (
        <div style={{ position: 'fixed', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
            {Array.from({ length: 18 }).map((_, i) => (
                <div
                    key={i}
                    style={{
                        position: 'absolute',
                        width: i % 3 === 0 ? 3 : 2,
                        height: i % 3 === 0 ? 3 : 2,
                        borderRadius: '50%',
                        background: i % 4 === 0 ? '#00d4c8' : i % 4 === 1 ? '#6366f1' : 'rgba(255,255,255,0.3)',
                        left: `${(i * 17 + 5) % 100}%`,
                        top: `${(i * 23 + 10) % 100}%`,
                        animation: `float-particle ${8 + (i % 5) * 2}s ease-in-out infinite`,
                        animationDelay: `${i * 0.5}s`,
                        opacity: 0.6,
                    }}
                />
            ))}
        </div>
    )
}

// Animated stat pill for the left panel
function StatPill({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div style={{
            background: 'rgba(255,255,255,0.04)',
            border: `1px solid ${color}30`,
            borderRadius: 10,
            padding: '10px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
        }}>
            <span style={{ fontSize: '1.125rem', fontWeight: 700, color }}>{value}</span>
            <span style={{ fontSize: '0.7rem', color: '#8b9ab5', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
        </div>
    )
}

export default function Login() {
    const navigate = useNavigate()
    const [email, setEmail] = useState('admin@example.com')
    const [password, setPassword] = useState('admin')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [mounted, setMounted] = useState(false)

    useEffect(() => { setMounted(true) }, [])

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
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            background: `radial-gradient(ellipse at 30% 20%, rgba(0,212,200,0.08) 0%, transparent 55%),
                         radial-gradient(ellipse at 80% 80%, rgba(99,102,241,0.07) 0%, transparent 55%),
                         #06080f`,
        }}>
            <Particles />

            {/* Left panel - branding */}
            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                padding: '60px 80px',
                position: 'relative',
                zIndex: 1,
                opacity: mounted ? 1 : 0,
                transform: mounted ? 'translateX(0)' : 'translateX(-20px)',
                transition: 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 48 }}>
                    <div style={{
                        width: 48, height: 48,
                        background: 'linear-gradient(135deg, #00d4c8, #0066ff)',
                        borderRadius: 14,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '1.5rem', color: '#000', fontWeight: 700,
                        boxShadow: '0 0 24px rgba(0,212,200,0.35)',
                    }}>✚</div>
                    <span style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f0f4f8', letterSpacing: '-0.03em' }}>MedPredict</span>
                </div>

                <h1 style={{
                    fontSize: '2.75rem',
                    fontWeight: 800,
                    color: '#f0f4f8',
                    lineHeight: 1.15,
                    letterSpacing: '-0.03em',
                    marginBottom: 20,
                }}>
                    Clinical AI<br />
                    <span style={{
                        background: 'linear-gradient(135deg, #00d4c8, #6366f1)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>Disease Prediction</span>
                </h1>

                <p style={{ color: '#8b9ab5', fontSize: '1rem', marginBottom: 40, maxWidth: 400, lineHeight: 1.7 }}>
                    Harness the power of ML-driven risk stratification across Heart Disease, Diabetes, Cancer, and Kidney Disease with SHAP explainability.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, maxWidth: 360 }}>
                    <StatPill label="Diseases Covered" value="4" color="#00d4c8" />
                    <StatPill label="Model AUC-PR" value="99.4%" color="#6366f1" />
                    <StatPill label="Training Samples" value="3,820" color="#f97316" />
                    <StatPill label="Avg Response" value="<200ms" color="#22c55e" />
                </div>

                <div style={{ marginTop: 48, display: 'flex', gap: 16 }}>
                    {['Heart', 'Diabetes', 'Cancer', 'Kidney'].map((d, i) => (
                        <div key={d} style={{
                            fontSize: '0.7rem',
                            color: ['#ef4444', '#eab308', '#a855f7', '#3b82f6'][i],
                            background: `${['rgba(239,68,68', 'rgba(234,179,8', 'rgba(168,85,247', 'rgba(59,130,246'][i]},0.1)`,
                            border: `1px solid ${['rgba(239,68,68', 'rgba(234,179,8', 'rgba(168,85,247', 'rgba(59,130,246'][i]},0.25)`,
                            borderRadius: 6,
                            padding: '4px 10px',
                            fontWeight: 600,
                        }}>{d}</div>
                    ))}
                </div>
            </div>

            {/* Right panel - form */}
            <div style={{
                width: 480,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 32,
                position: 'relative',
                zIndex: 1,
            }}>
                <div style={{
                    width: '100%',
                    maxWidth: 420,
                    background: 'rgba(13,17,23,0.92)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 24,
                    padding: '44px 40px',
                    backdropFilter: 'blur(24px)',
                    boxShadow: '0 32px 80px rgba(0,0,0,0.6)',
                    opacity: mounted ? 1 : 0,
                    transform: mounted ? 'translateY(0)' : 'translateY(20px)',
                    transition: 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.1s',
                }}>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f0f4f8', marginBottom: 6 }}>
                        Welcome back
                    </h2>
                    <p style={{ color: '#8b9ab5', fontSize: '0.875rem', marginBottom: 32 }}>
                        Sign in to access the clinical dashboard
                    </p>

                    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
                        <div className="form-group">
                            <label className="form-label">Email address</label>
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

                        {error && (
                            <p style={{
                                color: '#ef4444', fontSize: '0.8125rem',
                                background: 'rgba(239,68,68,0.08)',
                                border: '1px solid rgba(239,68,68,0.2)',
                                borderRadius: 10, padding: '10px 14px',
                            }}>⚠ {error}</p>
                        )}

                        <button
                            id="login-btn"
                            type="submit"
                            className="btn btn-primary"
                            disabled={loading}
                            style={{ width: '100%', justifyContent: 'center', padding: '13px 20px', borderRadius: 12, fontSize: '0.9375rem', marginTop: 4 }}
                        >
                            {loading ? <span className="spinner" /> : 'Sign in →'}
                        </button>
                    </form>

                    <div style={{
                        marginTop: 24,
                        padding: '12px 14px',
                        background: 'rgba(0,212,200,0.05)',
                        border: '1px solid rgba(0,212,200,0.15)',
                        borderRadius: 10,
                        fontSize: '0.75rem',
                        color: '#8b9ab5',
                        textAlign: 'center',
                    }}>
                        <span style={{ color: '#00d4c8' }}>Demo credentials: </span>
                        admin@example.com / admin
                    </div>
                </div>
            </div>

            <style>{`
                @keyframes float-particle {
                    0%, 100% { transform: translateY(0px) scale(1); opacity: 0.4; }
                    50% { transform: translateY(-20px) scale(1.2); opacity: 0.9; }
                }
            `}</style>
        </div>
    )
}
