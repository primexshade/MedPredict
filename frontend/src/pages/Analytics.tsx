import { useQuery } from '@tanstack/react-query'
import {
    ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
    Cell, BarChart, Bar, PieChart, Pie,
} from 'recharts'
import { analyticsAPI } from '../services/api'

const CLUSTER_COLORS = ['#00d4c8', '#6366f1', '#f97316', '#22c55e', '#ec4899', '#eab308']

// Richer mock cluster data with named clusters
const MOCK_CLUSTERS = Array.from({ length: 80 }, (_, i) => ({
    x: Math.sin(i * 0.42) * 80 + (Math.random() - 0.5) * 35,
    y: Math.cos(i * 0.37) * 60 + (Math.random() - 0.5) * 28,
    cluster: Math.floor(i / 20),
}))

const CLUSTER_LABELS = ['Low-risk Healthy', 'Metabolic Syndrome', 'Cardio High-risk', 'Elderly Multi-disease']

const RISK_DIST = [
    { label: 'Low Risk', pct: 38, color: '#22c55e', icon: '▼' },
    { label: 'Borderline', pct: 22, color: '#eab308', icon: '◆' },
    { label: 'Moderate', pct: 25, color: '#f97316', icon: '◆' },
    { label: 'High Risk', pct: 11, color: '#ef4444', icon: '▲' },
    { label: 'Critical', pct: 4, color: '#dc2626', icon: '⚠' },
]

// Feature importance mock (top SHAP features across all diseases)
const TOP_FEATURES = [
    { feature: 'cp (Chest Pain)', importance: 0.312, disease: 'Heart' },
    { feature: 'glucose', importance: 0.287, disease: 'Diabetes' },
    { feature: 'radius_worst', importance: 0.241, disease: 'Cancer' },
    { feature: 'thal', importance: 0.198, disease: 'Heart' },
    { feature: 'hemo (Hemoglobin)', importance: 0.165, disease: 'Kidney' },
    { feature: 'concave_pts_worst', importance: 0.143, disease: 'Cancer' },
]

const TOOLTIP_STYLE = {
    contentStyle: {
        background: '#141920', border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 8, color: '#f0f4f8', fontSize: 12,
    },
}

export default function Analytics() {
    const { data: rulesData } = useQuery({
        queryKey: ['comorbidity-rules'],
        queryFn: () => analyticsAPI.comorbidityRules().then((r) => r.data),
    })

    const rules = rulesData?.rules ?? []

    return (
        <div>
            <div className="page-header">
                <h1>Analytics</h1>
                <p className="page-subtitle">Population phenotyping · comorbidity patterns · global feature importance</p>
            </div>

            {/* Row 1: Clusters + Risk Distribution */}
            <div className="grid-2" style={{ marginBottom: 20 }}>
                {/* Cluster scatter */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Patient Phenotype Clusters</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>GMM · 4 clusters</span>
                    </div>
                    <ResponsiveContainer width="100%" height={240}>
                        <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                            <XAxis type="number" dataKey="x" name="PC1" tick={{ fill: '#8b9ab5', fontSize: 10 }} axisLine={false} tickLine={false} />
                            <YAxis type="number" dataKey="y" name="PC2" tick={{ fill: '#8b9ab5', fontSize: 10 }} axisLine={false} tickLine={false} />
                            <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                            <Scatter data={MOCK_CLUSTERS} name="Patients">
                                {MOCK_CLUSTERS.map((entry, idx) => (
                                    <Cell key={idx} fill={CLUSTER_COLORS[entry.cluster]} fillOpacity={0.8} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 12 }}>
                        {CLUSTER_LABELS.map((label, i) => (
                            <span key={i} style={{
                                fontSize: '0.7rem', color: CLUSTER_COLORS[i],
                                background: `${CLUSTER_COLORS[i]}12`,
                                border: `1px solid ${CLUSTER_COLORS[i]}30`,
                                padding: '2px 8px', borderRadius: 6,
                            }}>
                                ● {label}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Risk distribution */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Population Risk Distribution</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>All predictions</span>
                    </div>
                    {RISK_DIST.map(({ label, pct, color, icon }) => (
                        <div key={label} style={{ marginBottom: 16 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'center' }}>
                                <span style={{ fontSize: '0.8125rem', color: '#f0f4f8', display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <span style={{ color, fontSize: '0.7rem' }}>{icon}</span>
                                    {label}
                                </span>
                                <span style={{ fontSize: '0.8rem', color, fontWeight: 700 }}>{pct}%</span>
                            </div>
                            <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                                <div style={{
                                    height: '100%', width: `${pct}%`,
                                    background: `linear-gradient(90deg, ${color}, ${color}99)`,
                                    borderRadius: 4,
                                    boxShadow: `0 0 8px ${color}50`,
                                    transition: 'width 1s ease',
                                }} />
                            </div>
                        </div>
                    ))}

                    {/* Mini pie summary */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
                        <PieChart width={120} height={80}>
                            <Pie data={RISK_DIST} dataKey="pct" cx={60} cy={40} innerRadius={24} outerRadius={38} paddingAngle={2} startAngle={90} endAngle={-270}>
                                {RISK_DIST.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Pie>
                        </PieChart>
                    </div>
                </div>
            </div>

            {/* Row 2: Global SHAP feature importance */}
            <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                    <span className="card-title">Global Feature Importance (SHAP)</span>
                    <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>Avg |SHAP| across all models</span>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={TOP_FEATURES} layout="vertical" margin={{ left: 0, right: 40, top: 4 }}>
                        <XAxis type="number" tick={{ fill: '#8b9ab5', fontSize: 10 }} tickLine={false} axisLine={false} />
                        <YAxis type="category" dataKey="feature" width={160} tick={{ fill: '#f0f4f8', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(3), 'Importance']} />
                        <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                            {TOP_FEATURES.map((_, i) => (
                                <Cell key={i} fill={['#ef4444', '#eab308', '#a855f7', '#ef4444', '#3b82f6', '#a855f7'][i]} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 8 }}>
                    {['Heart', 'Diabetes', 'Cancer', 'Kidney'].map((d, i) => (
                        <span key={d} style={{
                            fontSize: '0.7rem',
                            color: ['#ef4444', '#eab308', '#a855f7', '#3b82f6'][i],
                            background: `${['rgba(239,68,68', 'rgba(234,179,8', 'rgba(168,85,247', 'rgba(59,130,246'][i]},0.1)`,
                            padding: '2px 8px', borderRadius: 6,
                        }}>■ {d}</span>
                    ))}
                </div>
            </div>

            {/* Comorbidity rules table */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Comorbidity Association Rules (FP-Growth)</span>
                    <span style={{
                        fontSize: '0.7rem', background: rules.length > 0 ? 'rgba(0,212,200,0.1)' : 'rgba(74,85,104,0.2)',
                        color: rules.length > 0 ? '#00d4c8' : '#8b9ab5',
                        padding: '2px 8px', borderRadius: 6,
                    }}>
                        {rules.length} rules discovered
                    </span>
                </div>
                {rules.length > 0 ? (
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Antecedents</th>
                                <th>→ Consequents</th>
                                <th>Support</th>
                                <th>Confidence</th>
                                <th>Lift</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rules.map((r, i) => (
                                <tr key={i}>
                                    <td style={{ color: '#8b9ab5' }}>{r.antecedents.join(', ')}</td>
                                    <td style={{ color: '#00d4c8', fontWeight: 600 }}>{r.consequents.join(', ')}</td>
                                    <td>{(r.support * 100).toFixed(1)}%</td>
                                    <td>{(r.confidence * 100).toFixed(1)}%</td>
                                    <td style={{
                                        color: r.lift > 2 ? '#22c55e' : r.lift > 1.5 ? '#eab308' : '#f0f4f8',
                                        fontWeight: 600,
                                    }}>
                                        {r.lift.toFixed(2)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="empty-state">
                        <div style={{ fontSize: '2.5rem', marginBottom: 12, opacity: 0.25 }}>◈</div>
                        <p style={{ fontWeight: 600, marginBottom: 6 }}>No association rules mined yet</p>
                        <p style={{ fontSize: '0.8125rem' }}>
                            Rules are discovered when enough patients have multi-disease predictions.
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}
