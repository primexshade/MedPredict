import { useState } from 'react'
import { predictAPI } from '../services/api'
import type { PredictionResponse } from '../services/api'
import RiskGauge from '../components/RiskGauge'
import ShapChart from '../components/ShapChart'
import RiskBadge from '../components/RiskBadge'

// ── Form field definitions ────────────────────────────────────────────────────
const HEART_FIELDS = [
    { key: 'age', label: 'Age (years)', min: 20, max: 90, step: 1, default: 55 },
    { key: 'sex', label: 'Sex (1=M, 0=F)', min: 0, max: 1, step: 1, default: 1 },
    { key: 'cp', label: 'Chest Pain Type (0–3)', min: 0, max: 3, step: 1, default: 0 },
    { key: 'trestbps', label: 'Resting BP (mmHg)', min: 80, max: 220, step: 1, default: 130 },
    { key: 'chol', label: 'Serum Cholesterol (mg/dl)', min: 100, max: 600, step: 1, default: 240 },
    { key: 'fbs', label: 'Fasting BS > 120 (1/0)', min: 0, max: 1, step: 1, default: 0 },
    { key: 'thalach', label: 'Max Heart Rate', min: 60, max: 220, step: 1, default: 150 },
    { key: 'exang', label: 'Exercise Angina (1/0)', min: 0, max: 1, step: 1, default: 0 },
    { key: 'oldpeak', label: 'ST Depression', min: 0, max: 7, step: 0.1, default: 1.0 },
    { key: 'ca', label: 'Major Vessels (0–3)', min: 0, max: 3, step: 1, default: 0 },
]

const DIABETES_FIELDS = [
    { key: 'pregnancies', label: 'Pregnancies', min: 0, max: 17, step: 1, default: 2 },
    { key: 'glucose', label: 'Glucose (mg/dl)', min: 0, max: 250, step: 1, default: 110 },
    { key: 'bloodpressure', label: 'Blood Pressure (mmHg)', min: 0, max: 130, step: 1, default: 72 },
    { key: 'skinthickness', label: 'Skin Thickness (mm)', min: 0, max: 100, step: 1, default: 23 },
    { key: 'insulin', label: 'Insulin (μU/mL)', min: 0, max: 900, step: 1, default: 80 },
    { key: 'bmi', label: 'BMI', min: 10, max: 70, step: 0.1, default: 28.0 },
    { key: 'diabetespedigreefunction', label: 'Pedigree Function', min: 0, max: 3, step: 0.01, default: 0.35 },
    { key: 'age', label: 'Age (years)', min: 18, max: 90, step: 1, default: 38 },
]

export default function Predict() {
    const [tab, setTab] = useState<'heart' | 'diabetes'>('heart')
    const fields = tab === 'heart' ? HEART_FIELDS : DIABETES_FIELDS

    // Form state: default values
    const [values, setValues] = useState<Record<string, number>>(() =>
        Object.fromEntries(fields.map((f) => [f.key, f.default]))
    )

    const [result, setResult] = useState<PredictionResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleTabChange = (newTab: 'heart' | 'diabetes') => {
        setTab(newTab)
        setResult(null)
        setError('')
        const newFields = newTab === 'heart' ? HEART_FIELDS : DIABETES_FIELDS
        setValues(Object.fromEntries(newFields.map((f) => [f.key, f.default])))
    }

    const handleChange = (key: string, value: string) =>
        setValues((v) => ({ ...v, [key]: parseFloat(value) }))

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            const fn = tab === 'heart' ? predictAPI.heart : predictAPI.diabetes
            const { data } = await fn(values)
            setResult(data)
        } catch (err: unknown) {
            const axiosErr = err as { response?: { data?: { detail?: string } } }
            const errMsg = axiosErr?.response?.data?.detail ??
                'No trained model available yet. Run scripts/run_training.py first.'
            setError(String(errMsg))
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <div className="page-header">
                <h1>Predict</h1>
                <p className="page-subtitle">Enter patient biomarkers to generate a risk prediction</p>
            </div>

            <div className="tab-bar" style={{ maxWidth: 400 }}>
                <button className={`tab ${tab === 'heart' ? 'active' : ''}`} onClick={() => handleTabChange('heart')}>
                    ♥ Heart Disease
                </button>
                <button className={`tab ${tab === 'diabetes' ? 'active' : ''}`} onClick={() => handleTabChange('diabetes')}>
                    ⬡ Diabetes
                </button>
            </div>

            <div className="grid-2" style={{ alignItems: 'start' }}>
                {/* Input form */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Patient Biomarkers</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>
                            {tab === 'heart' ? '10 features' : '8 features'}
                        </span>
                    </div>
                    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                            {fields.map((f) => (
                                <div key={f.key} className="form-group">
                                    <label className="form-label">{f.label}</label>
                                    <input
                                        id={`field-${f.key}`}
                                        type="number"
                                        className="form-input"
                                        value={values[f.key] ?? f.default}
                                        min={f.min}
                                        max={f.max}
                                        step={f.step}
                                        onChange={(e) => handleChange(f.key, e.target.value)}
                                    />
                                </div>
                            ))}
                        </div>
                        {error && (
                            <p style={{
                                color: '#f97316', fontSize: '0.8125rem',
                                background: 'rgba(249,115,22,0.08)',
                                border: '1px solid rgba(249,115,22,0.2)',
                                borderRadius: 8, padding: '10px 14px',
                            }}>
                                ⚠ {error}
                            </p>
                        )}
                        <button
                            id="predict-btn"
                            type="submit"
                            className="btn btn-primary"
                            disabled={loading}
                            style={{ alignSelf: 'flex-start', minWidth: 160 }}
                        >
                            {loading ? <><span className="spinner" /> Analysing…</> : '⚕ Run Prediction'}
                        </button>
                    </form>
                </div>

                {/* Result panel */}
                {result ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {/* Gauge card */}
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="card-header">
                                <span className="card-title">Risk Assessment</span>
                                <RiskBadge category={result.risk_category} size="sm" />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
                                <RiskGauge category={result.risk_category} score={result.composite_score} size={180} />
                            </div>
                            <p style={{ fontSize: '0.875rem', color: '#8b9ab5', marginBottom: 12 }}>
                                {result.plain_english_summary}
                            </p>
                            <div style={{
                                background: 'rgba(0,212,200,0.06)',
                                border: '1px solid rgba(0,212,200,0.15)',
                                borderRadius: 10, padding: '12px 16px', textAlign: 'left',
                            }}>
                                <p style={{ fontSize: '0.75rem', color: '#00d4c8', fontWeight: 600, marginBottom: 4 }}>
                                    CLINICAL ACTION — {result.clinical_action.urgency}
                                </p>
                                <p style={{ fontSize: '0.8125rem', color: '#f0f4f8' }}>{result.clinical_action.action}</p>
                            </div>
                            <p style={{ fontSize: '0.75rem', color: '#4a5568', marginTop: 10 }}>
                                95% CI: [{(result.confidence_interval[0] * 100).toFixed(1)}% —{' '}
                                {(result.confidence_interval[1] * 100).toFixed(1)}%]
                            </p>
                        </div>

                        {/* SHAP chart */}
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Feature Explanations (SHAP)</span>
                            </div>
                            {result.shap_contributions?.length ? (
                                <ShapChart contributions={result.shap_contributions} />
                            ) : (
                                <p style={{ color: '#4a5568', fontSize: '0.875rem' }}>
                                    SHAP explanations not available (model not yet trained).
                                </p>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
                        <div style={{ fontSize: '3rem', marginBottom: 16, opacity: 0.3 }}>⚕</div>
                        <p style={{ color: '#4a5568' }}>Fill in patient biomarkers and click Run Prediction to see the AI risk assessment here.</p>
                    </div>
                )}
            </div>
        </div>
    )
}
