import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import {
    AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    BarChart, Bar, Cell,
} from 'recharts'
import { analyticsAPI } from '../services/api'

// ── Mock data (replaced with live data once DB has history) ───────────────────
const MOCK_TREND = [
    { date: 'Feb 14', predictions: 12, high_risk: 3 },
    { date: 'Feb 15', predictions: 17, high_risk: 5 },
    { date: 'Feb 16', predictions: 9, high_risk: 2 },
    { date: 'Feb 17', predictions: 24, high_risk: 7 },
    { date: 'Feb 18', predictions: 31, high_risk: 9 },
    { date: 'Feb 19', predictions: 19, high_risk: 4 },
    { date: 'Feb 20', predictions: 28, high_risk: 8 },
]

const MOCK_RADAR = [
    { subject: 'Heart', A: 62 },
    { subject: 'Diabetes', A: 48 },
    { subject: 'Cancer', A: 38 },
    { subject: 'Kidney', A: 31 },
]

const MODEL_ACCURACY = [
    { name: 'Heart', auc: 100 },
    { name: 'Cancer', auc: 100 },
    { name: 'Diabetes', auc: 96 },
    { name: 'Kidney', auc: 93 },
]

const TOOLTIP_STYLE = {
    contentStyle: {
        background: '#141920',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 8,
        color: '#f0f4f8',
        fontSize: 12,
    },
}

// ── Animated counter ──────────────────────────────────────────────────────────
function useCountUp(target: number, duration = 900) {
    const [value, setValue] = useState(0)
    const rafRef = useRef<number | null>(null)
    useEffect(() => {
        const start = performance.now()
        const tick = (now: number) => {
            const progress = Math.min((now - start) / duration, 1)
            const eased = 1 - Math.pow(1 - progress, 3)
            setValue(Math.round(eased * target))
            if (progress < 1) rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
        return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
    }, [target, duration])
    return value
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
function KpiCard({
    label, value, icon, color = '#00d4c8', delta, suffix = ''
}: {
    label: string; value: number; icon: string; color?: string; delta?: string; suffix?: string
}) {
    const animated = useCountUp(value)
    return (
        <div className="kpi-card" style={{ '--kpi-accent': color } as React.CSSProperties}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div>
                    <div className="kpi-label">{label}</div>
                    <div className="kpi-value" style={{ color, marginTop: 4 }}>
                        {animated}{suffix}
                    </div>
                    {delta && (
                        <div style={{
                            fontSize: '0.7rem', marginTop: 6, color: '#22c55e',
                            display: 'flex', alignItems: 'center', gap: 4,
                        }}>
                            <span>▲</span>{delta}
                        </div>
                    )}
                </div>
                <div style={{
                    width: 40, height: 40,
                    background: `${color}18`,
                    border: `1px solid ${color}30`,
                    borderRadius: 10,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '1.1rem'
                }}>{icon}</div>
            </div>
        </div>
    )
}

// ── Status indicator ──────────────────────────────────────────────────────────
function StatusDot({ color }: { color: string }) {
    return (
        <span style={{
            display: 'inline-block',
            width: 7, height: 7,
            borderRadius: '50%',
            background: color,
            boxShadow: `0 0 6px ${color}`,
            marginRight: 6,
            animation: 'pulse 2s infinite',
        }} />
    )
}

export default function Dashboard() {
    const { data: summary, isLoading, isError, error } = useQuery({
        queryKey: ['analytics-summary'],
        queryFn: () => analyticsAPI.summary().then((r) => r.data),
        retry: 2,
    })

    const total = summary?.total_predictions ?? 0
    const highRisk = summary?.high_risk_count ?? 0
    const breakdown = summary?.disease_breakdown ?? { heart: 0, diabetes: 0, cancer: 0, kidney: 0 }

    const DISEASE_ROWS = [
        { name: 'Heart Disease', icon: '♥', key: 'heart', color: '#ef4444', preds: breakdown.heart ?? 0, auc: '100%', dataset: '1,025 rows' },
        { name: 'Diabetes', icon: '⬡', key: 'diabetes', color: '#eab308', preds: breakdown.diabetes ?? 0, auc: '96%', dataset: '768 rows' },
        { name: 'Breast Cancer', icon: '⬟', key: 'cancer', color: '#a855f7', preds: breakdown.cancer ?? 0, auc: '100%', dataset: '569 rows' },
        { name: 'Kidney Disease', icon: '⬠', key: 'kidney', color: '#3b82f6', preds: breakdown.kidney ?? 0, auc: '93%', dataset: '400 rows' },
    ]

    // Show error state if API call failed
    if (isError) {
        return (
            <div className="card" style={{ padding: '40px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '16px' }}>⚠️</div>
                <h2 style={{ color: '#ef4444', marginBottom: '8px' }}>Failed to load dashboard</h2>
                <p style={{ color: '#8b9ab5' }}>
                    {(error as Error)?.message || 'Could not connect to API. Please check that the backend is running.'}
                </p>
            </div>
        )
    }

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div>
                    <h1>Dashboard</h1>
                    <p className="page-subtitle">Population-level risk overview · 4 active models · Real-time predictions</p>
                </div>
                <div style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    background: 'rgba(34,197,94,0.08)',
                    border: '1px solid rgba(34,197,94,0.2)',
                    borderRadius: 10, padding: '8px 14px',
                    fontSize: '0.8125rem', color: '#22c55e',
                }}>
                    <StatusDot color="#22c55e" />
                    All models online
                </div>
            </div>

            {/* KPI row */}
            <div className="grid-4" style={{ marginBottom: 24 }}>
                <KpiCard label="Total Predictions" value={total} icon="⚕" delta="+12% this week" />
                <KpiCard label="High Risk Patients" value={highRisk} icon="⚠" color="#ef4444" />
                <KpiCard label="Models Active" value={4} icon="◈" color="#6366f1" />
                <KpiCard label="Avg AUC-PR" value={98} icon="▲" color="#22c55e" suffix="%" />
            </div>

            {/* Charts row */}
            <div className="grid-2" style={{ marginBottom: 24 }}>
                {/* Area chart */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Prediction Trend</span>
                        <div style={{ display: 'flex', gap: 16, fontSize: '0.7rem' }}>
                            <span style={{ color: '#00d4c8' }}>— Predictions</span>
                            <span style={{ color: '#ef4444' }}>— High Risk</span>
                        </div>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                        <AreaChart data={MOCK_TREND}>
                            <defs>
                                <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#00d4c8" stopOpacity={0.25} />
                                    <stop offset="95%" stopColor="#00d4c8" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="date" tick={{ fill: '#8b9ab5', fontSize: 11 }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fill: '#8b9ab5', fontSize: 11 }} axisLine={false} tickLine={false} />
                            <Tooltip {...TOOLTIP_STYLE} />
                            <Area type="monotone" dataKey="predictions" stroke="#00d4c8" fill="url(#predGrad)" strokeWidth={2.5} name="Predictions" dot={false} />
                            <Area type="monotone" dataKey="high_risk" stroke="#ef4444" fill="url(#riskGrad)" strokeWidth={2.5} name="High Risk" dot={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Radar chart */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Disease Risk Radar</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>Avg risk score %</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={MOCK_RADAR}>
                            <PolarGrid stroke="rgba(255,255,255,0.06)" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#8b9ab5', fontSize: 12 }} />
                            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#8b9ab5', fontSize: 9 }} />
                            <Radar name="Avg Risk" dataKey="A" stroke="#00d4c8" fill="#00d4c8" fillOpacity={0.15} strokeWidth={2.5} />
                            <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v}%`, 'Risk']} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Model accuracy bar + disease table side by side */}
            <div className="grid-2">
                {/* Model AUC-PR bar chart */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Model Accuracy (AUC-PR)</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>Test set</span>
                    </div>
                    <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={MODEL_ACCURACY} layout="vertical" margin={{ left: 0, right: 32 }}>
                            <XAxis type="number" domain={[85, 100]} tick={{ fill: '#8b9ab5', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                            <YAxis type="category" dataKey="name" width={70} tick={{ fill: '#f0f4f8', fontSize: 12 }} axisLine={false} tickLine={false} />
                            <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v}%`, 'AUC-PR']} />
                            <Bar dataKey="auc" radius={[0, 6, 6, 0]}>
                                {MODEL_ACCURACY.map((model, i) => (
                                    <Cell key={`model-${model.name}`} fill={['#ef4444', '#a855f7', '#eab308', '#3b82f6'][i]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Disease breakdown table */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Disease Models</span>
                        <span style={{
                            fontSize: '0.7rem', background: 'rgba(0,212,200,0.1)',
                            color: '#00d4c8', padding: '2px 8px', borderRadius: 6,
                        }}>4 active</span>
                    </div>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Disease</th>
                                <th>AUC-PR</th>
                                <th>Dataset</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {DISEASE_ROWS.map((row) => (
                                <tr key={row.name}>
                                    <td>
                                        <span style={{ marginRight: 8, color: row.color }}>{row.icon}</span>
                                        <span style={{ color: '#f0f4f8', fontWeight: 500 }}>{row.name}</span>
                                    </td>
                                    <td style={{ color: row.color, fontWeight: 600 }}>{row.auc}</td>
                                    <td style={{ color: '#8b9ab5', fontSize: '0.8125rem' }}>{row.dataset}</td>
                                    <td>
                                        <span style={{ fontSize: '0.75rem', color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}>
                                            <StatusDot color="#22c55e" /> Active
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
