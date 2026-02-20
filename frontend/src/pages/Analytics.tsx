import { useQuery } from '@tanstack/react-query'
import {
    ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
    Cell,
} from 'recharts'
import { analyticsAPI } from '../services/api'

const CLUSTER_COLORS = ['#00d4c8', '#6366f1', '#f97316', '#22c55e', '#ec4899', '#eab308']

// Mock cluster scatter data (until real patient data exists)
const MOCK_CLUSTERS = Array.from({ length: 60 }, (_, i) => ({
    x: Math.sin(i * 0.5) * 80 + (Math.random() - 0.5) * 40,
    y: Math.cos(i * 0.4) * 60 + (Math.random() - 0.5) * 30,
    cluster: Math.floor(i / 15),
}))

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
                <p className="page-subtitle">Population phenotyping, patient clusters, and comorbidity patterns</p>
            </div>

            <div className="grid-2" style={{ marginBottom: 20 }}>
                {/* Cluster scatter */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Patient Phenotype Clusters</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>GMM (PCA projection)</span>
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                        <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                            <XAxis type="number" dataKey="x" name="PC1" tick={{ fill: '#8b9ab5', fontSize: 10 }} />
                            <YAxis type="number" dataKey="y" name="PC2" tick={{ fill: '#8b9ab5', fontSize: 10 }} />
                            <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                            <Scatter data={MOCK_CLUSTERS} name="Patients">
                                {MOCK_CLUSTERS.map((entry, idx) => (
                                    <Cell key={idx} fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]} fillOpacity={0.75} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
                        {CLUSTER_COLORS.slice(0, 4).map((c, i) => (
                            <span key={i} style={{ fontSize: '0.75rem', color: c }}>
                                ● Cluster {i + 1}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Population risk overview */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Population Risk Distribution</span>
                    </div>
                    {[
                        { label: 'Low Risk', pct: 38, color: '#22c55e' },
                        { label: 'Borderline', pct: 22, color: '#eab308' },
                        { label: 'Moderate Risk', pct: 25, color: '#f97316' },
                        { label: 'High Risk', pct: 11, color: '#ef4444' },
                        { label: 'Critical', pct: 4, color: '#dc2626' },
                    ].map(({ label, pct, color }) => (
                        <div key={label} style={{ marginBottom: 14 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                                <span style={{ fontSize: '0.8125rem', color: '#f0f4f8' }}>{label}</span>
                                <span style={{ fontSize: '0.8125rem', color }}>{pct}%</span>
                            </div>
                            <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3 }}>
                                <div style={{
                                    height: '100%', width: `${pct}%`,
                                    background: color, borderRadius: 3,
                                    boxShadow: `0 0 6px ${color}60`,
                                }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Comorbidity rules table */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Comorbidity Association Rules (FP-Growth)</span>
                    <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>
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
                                    <td>{r.antecedents.join(', ')}</td>
                                    <td style={{ color: '#00d4c8' }}>{r.consequents.join(', ')}</td>
                                    <td>{(r.support * 100).toFixed(1)}%</td>
                                    <td>{(r.confidence * 100).toFixed(1)}%</td>
                                    <td style={{ color: r.lift > 2 ? '#22c55e' : '#f0f4f8' }}>
                                        {r.lift.toFixed(2)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="empty-state">
                        <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>◈</div>
                        <p>No association rules yet.</p>
                        <p style={{ fontSize: '0.8125rem', marginTop: 6 }}>
                            Train models and run FP-Growth mining to discover comorbidity patterns.
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}
