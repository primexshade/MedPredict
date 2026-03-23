import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { patientsAPI } from '../services/api'

// Realistic demo patients for when DB is empty
const DEMO_PATIENTS = [
    { id: 'demo-001', mrn: 'MRN-238491', name: 'James R. Wilson', age: 62, disease: 'Heart', risk: 'HIGH', lastPred: '2 hrs ago' },
    { id: 'demo-002', mrn: 'MRN-104827', name: 'Priya S. Sharma', age: 45, disease: 'Diabetes', risk: 'MODERATE', lastPred: '4 hrs ago' },
    { id: 'demo-003', mrn: 'MRN-371920', name: 'Angela T. Davis', age: 58, disease: 'Cancer', risk: 'LOW', lastPred: 'Yesterday' },
    { id: 'demo-004', mrn: 'MRN-029183', name: 'Robert K. Lee', age: 71, disease: 'Kidney', risk: 'CRITICAL', lastPred: '1 hr ago' },
    { id: 'demo-005', mrn: 'MRN-582049', name: 'Maria L. Gonzalez', age: 39, disease: 'Diabetes', risk: 'BORDERLINE', lastPred: '3 hrs ago' },
    { id: 'demo-006', mrn: 'MRN-694710', name: 'David H. Chen', age: 55, disease: 'Heart', risk: 'MODERATE', lastPred: '5 hrs ago' },
]

const RISK_COLORS: Record<string, string> = {
    LOW: '#22c55e',
    BORDERLINE: '#eab308',
    MODERATE: '#f97316',
    HIGH: '#ef4444',
    CRITICAL: '#dc2626',
}

const DISEASE_COLORS: Record<string, string> = {
    Heart: '#ef4444',
    Diabetes: '#eab308',
    Cancer: '#a855f7',
    Kidney: '#3b82f6',
}

function RiskBadgeSmall({ category }: { category: string }) {
    const color = RISK_COLORS[category] ?? '#8b9ab5'
    return (
        <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 5,
            padding: '3px 10px', borderRadius: 100,
            fontSize: '0.7rem', fontWeight: 700,
            background: `${color}18`, color, border: `1px solid ${color}35`,
            textTransform: 'uppercase', letterSpacing: '0.04em',
            animation: category === 'CRITICAL' ? 'pulse 1.4s infinite' : 'none',
        }}>
            ● {category}
        </span>
    )
}

export default function Patients() {
    const { data: apiPatients = [], isLoading, isError, error } = useQuery({
        queryKey: ['patients'],
        queryFn: () => patientsAPI.list().then((r) => r.data),
        retry: 2,
    })
    const [search, setSearch] = useState('')

    // Show error state if API call failed
    if (isError) {
        return (
            <div className="card" style={{ padding: '40px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '16px' }}>⚠️</div>
                <h2 style={{ color: '#ef4444', marginBottom: '8px' }}>Failed to load patients</h2>
                <p style={{ color: '#8b9ab5' }}>
                    {(error as Error)?.message || 'Could not connect to API.'}
                </p>
            </div>
        )
    }

    // Use demo data if no real patients yet
    const isDemo = apiPatients.length === 0 && !isLoading

    const displayPatients = isDemo
        ? DEMO_PATIENTS.filter(p =>
            p.mrn.toLowerCase().includes(search.toLowerCase()) ||
            p.name.toLowerCase().includes(search.toLowerCase())
        )
        : apiPatients.filter(p =>
            p.mrn?.toLowerCase().includes(search.toLowerCase()) ||
            p.id?.toLowerCase().includes(search.toLowerCase())
        )

    const stats = {
        total: DEMO_PATIENTS.length,
        highRisk: DEMO_PATIENTS.filter(p => ['HIGH', 'CRITICAL'].includes(p.risk)).length,
        active: DEMO_PATIENTS.length,
    }

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div>
                    <h1>Patients</h1>
                    <p className="page-subtitle">Patient registry and prediction history</p>
                </div>
                <div style={{ display: 'flex', gap: 10 }}>
                    <button className="btn btn-outline" style={{ fontSize: '0.8125rem' }}>↓ Export CSV</button>
                    <button className="btn btn-primary" id="add-patient-btn">⊕ Add Patient</button>
                </div>
            </div>

            {/* Quick stats */}
            <div className="grid-3" style={{ marginBottom: 24 }}>
                {[
                    { label: 'Total Patients', value: stats.total, color: '#00d4c8', icon: '⊕' },
                    { label: 'High / Critical', value: stats.highRisk, color: '#ef4444', icon: '⚠' },
                    { label: 'Active This Week', value: stats.active, color: '#22c55e', icon: '◈' },
                ].map(({ label, value, color, icon }) => (
                    <div key={label} className="kpi-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div>
                                <div className="kpi-label">{label}</div>
                                <div className="kpi-value" style={{ color }}>{value}</div>
                            </div>
                            <div style={{
                                width: 38, height: 38,
                                background: `${color}18`, border: `1px solid ${color}30`,
                                borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
                            }}>{icon}</div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Search + filter bar */}
            <div style={{ display: 'flex', gap: 12, marginBottom: 20, alignItems: 'center' }}>
                <div style={{ flex: 1, maxWidth: 400, position: 'relative' }}>
                    <span style={{
                        position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)',
                        color: '#8b9ab5', fontSize: '0.875rem', pointerEvents: 'none',
                    }}>🔍</span>
                    <input
                        className="form-input"
                        style={{ paddingLeft: 36 }}
                        placeholder="Search by MRN or name…"
                        id="patient-search"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                </div>
                {isDemo && (
                    <span style={{
                        fontSize: '0.75rem', color: '#eab308',
                        background: 'rgba(234,179,8,0.08)',
                        border: '1px solid rgba(234,179,8,0.2)',
                        borderRadius: 8, padding: '6px 12px',
                    }}>
                        ◆ Showing demo data
                    </span>
                )}
            </div>

            <div className="card">
                {isLoading ? (
                    <div style={{ textAlign: 'center', padding: 60 }}>
                        <div className="spinner" style={{ margin: '0 auto' }} />
                        <p style={{ marginTop: 16, color: '#8b9ab5' }}>Loading patients…</p>
                    </div>
                ) : (
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Patient</th>
                                <th>MRN</th>
                                <th>Age</th>
                                <th>Disease</th>
                                <th>Risk Level</th>
                                <th>Last Prediction</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {displayPatients.map((p) => {
                                const demo = 'name' in p ? p as typeof DEMO_PATIENTS[0] : null
                                const name = demo?.name ?? `Patient ${p.id?.slice(0, 6) ?? '—'}`
                                const age = demo?.age ?? '—'
                                const disease = demo?.disease ?? '—'
                                const risk = demo?.risk ?? '—'
                                const lastPred = demo?.lastPred ?? '—'
                                const mrn = p.mrn ?? '—'

                                return (
                                    <tr key={p.id}>
                                        <td>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <div style={{
                                                    width: 32, height: 32, borderRadius: '50%',
                                                    background: `linear-gradient(135deg, #00d4c8, #6366f1)`,
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    fontSize: '0.7rem', fontWeight: 700, color: '#000', flexShrink: 0,
                                                }}>
                                                    {name.split(' ').map((n: string) => n[0]).slice(0, 2).join('')}
                                                </div>
                                                <span style={{ color: '#f0f4f8', fontWeight: 500 }}>{name}</span>
                                            </div>
                                        </td>
                                        <td style={{ fontFamily: 'monospace', color: '#00d4c8', fontSize: '0.8125rem' }}>{mrn}</td>
                                        <td style={{ color: '#8b9ab5' }}>{age}</td>
                                        <td>
                                            <span style={{
                                                color: DISEASE_COLORS[disease] ?? '#8b9ab5',
                                                fontSize: '0.8125rem', fontWeight: 600,
                                            }}>{disease}</span>
                                        </td>
                                        <td><RiskBadgeSmall category={risk} /></td>
                                        <td style={{ color: '#8b9ab5', fontSize: '0.8125rem' }}>{lastPred}</td>
                                        <td>
                                            <div style={{ display: 'flex', gap: 6 }}>
                                                <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>View</button>
                                                <button className="btn btn-outline" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>Predict</button>
                                            </div>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}
