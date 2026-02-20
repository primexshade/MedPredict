import { useQuery } from '@tanstack/react-query'
import {
    AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { analyticsAPI } from '../services/api'

// ── Mock trend data (replace with real API once DB has data) ──────────────────
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
    { subject: 'Heart', A: 68 },
    { subject: 'Diabetes', A: 54 },
    { subject: 'Cancer', A: 42 },
    { subject: 'Kidney', A: 37 },
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

export default function Dashboard() {
    const { data: summary } = useQuery({
        queryKey: ['analytics-summary'],
        queryFn: () => analyticsAPI.summary().then((r) => r.data),
    })

    const totalPredictions = summary?.total_predictions ?? 0
    const highRisk = summary?.high_risk_count ?? 0
    const breakdown = summary?.disease_breakdown ?? { heart: 0, diabetes: 0, cancer: 0, kidney: 0 }

    return (
        <div>
            <div className="page-header">
                <h1>Dashboard</h1>
                <p className="page-subtitle">Population-level risk overview and prediction trends</p>
            </div>

            {/* KPI Cards */}
            <div className="grid-4" style={{ marginBottom: 24 }}>
                <KpiCard label="Total Predictions" value={totalPredictions.toString()} delta="+12% this week" />
                <KpiCard label="High Risk Patients" value={highRisk.toString()} delta="" color="#ef4444" />
                <KpiCard label="Heart Disease" value={breakdown.heart?.toString() ?? '0'} delta="" />
                <KpiCard label="Diabetes" value={breakdown.diabetes?.toString() ?? '0'} delta="" />
            </div>

            {/* Charts row */}
            <div className="grid-2">
                {/* Area chart — prediction trend */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Prediction Trend</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>Last 7 days</span>
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
                            <XAxis dataKey="date" tick={{ fill: '#8b9ab5', fontSize: 11 }} />
                            <YAxis tick={{ fill: '#8b9ab5', fontSize: 11 }} />
                            <Tooltip {...TOOLTIP_STYLE} />
                            <Area type="monotone" dataKey="predictions" stroke="#00d4c8" fill="url(#predGrad)" strokeWidth={2} name="Predictions" />
                            <Area type="monotone" dataKey="high_risk" stroke="#ef4444" fill="url(#riskGrad)" strokeWidth={2} name="High Risk" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Radar chart — disease comparison */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Disease Risk Radar</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>Avg risk score (%)</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={MOCK_RADAR}>
                            <PolarGrid stroke="rgba(255,255,255,0.06)" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#8b9ab5', fontSize: 12 }} />
                            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#8b9ab5', fontSize: 10 }} />
                            <Radar name="Avg Risk" dataKey="A" stroke="#00d4c8" fill="#00d4c8" fillOpacity={0.15} strokeWidth={2} />
                            <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v}%`, 'Avg Risk']} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Disease stats */}
            <div className="card" style={{ marginTop: 20 }}>
                <div className="card-header">
                    <span className="card-title">Disease Breakdown</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Disease</th>
                            <th>Total Predictions</th>
                            <th>Avg Risk Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[
                            { name: 'Heart Disease', preds: breakdown.heart ?? 0, avg: '62%', status: 'Active' },
                            { name: 'Diabetes', preds: breakdown.diabetes ?? 0, avg: '48%', status: 'Active' },
                            { name: 'Breast Cancer', preds: breakdown.cancer ?? 0, avg: '35%', status: 'Active' },
                            { name: 'Kidney Disease', preds: breakdown.kidney ?? 0, avg: '29%', status: 'Active' },
                        ].map((row) => (
                            <tr key={row.name}>
                                <td style={{ color: '#f0f4f8', fontWeight: 500 }}>{row.name}</td>
                                <td>{row.preds}</td>
                                <td style={{ color: '#00d4c8' }}>{row.avg}</td>
                                <td><span style={{ color: '#22c55e', fontSize: '0.75rem' }}>● {row.status}</span></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

function KpiCard({ label, value, delta, color = '#00d4c8' }: { label: string; value: string; delta: string; color?: string }) {
    return (
        <div className="kpi-card">
            <div className="kpi-label">{label}</div>
            <div className="kpi-value" style={{ color }}>{value}</div>
            {delta && <div className="kpi-delta">{delta}</div>}
        </div>
    )
}
