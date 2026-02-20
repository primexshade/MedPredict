import { useQuery } from '@tanstack/react-query'
import { patientsAPI } from '../services/api'

export default function Patients() {
    const { data: patients = [], isLoading } = useQuery({
        queryKey: ['patients'],
        queryFn: () => patientsAPI.list().then((r) => r.data),
    })

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                    <h1>Patients</h1>
                    <p className="page-subtitle">Patient registry and prediction history</p>
                </div>
                <button className="btn btn-primary" id="add-patient-btn">⊕ Add Patient</button>
            </div>

            {/* Search bar */}
            <div style={{ marginBottom: 20, maxWidth: 400 }}>
                <input
                    className="form-input"
                    placeholder="🔍  Search by MRN or name…"
                    id="patient-search"
                />
            </div>

            <div className="card">
                {isLoading ? (
                    <div style={{ textAlign: 'center', padding: 40 }}>
                        <div className="spinner" style={{ margin: '0 auto' }} />
                    </div>
                ) : patients.length > 0 ? (
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>MRN</th>
                                <th>Patient ID</th>
                                <th>Last Risk Score</th>
                                <th>Risk Category</th>
                                <th>Last Prediction</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {patients.map((p) => (
                                <tr key={p.id}>
                                    <td style={{ fontFamily: 'monospace', color: '#00d4c8' }}>{p.mrn}</td>
                                    <td style={{ color: '#8b9ab5', fontSize: '0.8125rem' }}>{p.id.slice(0, 8)}…</td>
                                    <td>—</td>
                                    <td>—</td>
                                    <td style={{ color: '#8b9ab5', fontSize: '0.8125rem' }}>—</td>
                                    <td>
                                        <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
                                            View
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="empty-state">
                        <div style={{ fontSize: '3rem', marginBottom: 12, opacity: 0.25 }}>⊕</div>
                        <p>No patients registered yet.</p>
                        <p style={{ fontSize: '0.8125rem', marginTop: 6 }}>
                            Patients will appear here once added through the API or admin panel.
                        </p>
                        <button className="btn btn-outline" style={{ marginTop: 20 }} id="empty-add-patient-btn">
                            Add First Patient
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
