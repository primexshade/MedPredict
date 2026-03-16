import { useState } from 'react'
import { predictAPI } from '../services/api'
import type { PredictionResponse, TopFeature } from '../services/api'
import RiskGauge from '../components/RiskGauge'
import RiskBadge from '../components/RiskBadge'

// ── Form field definitions ────────────────────────────────────────────────────

type DiseaseTab = 'heart' | 'diabetes' | 'cancer' | 'kidney'

const HEART_FIELDS = [
    { key: 'age', label: 'Age (years)', min: 20, max: 90, step: 1, default: 55 },
    { key: 'sex', label: 'Sex (1=M, 0=F)', min: 0, max: 1, step: 1, default: 1 },
    { key: 'cp', label: 'Chest Pain Type (0–3)', min: 0, max: 3, step: 1, default: 0 },
    { key: 'trestbps', label: 'Resting BP (mmHg)', min: 80, max: 220, step: 1, default: 130 },
    { key: 'chol', label: 'Serum Cholesterol (mg/dl)', min: 100, max: 600, step: 1, default: 240 },
    { key: 'fbs', label: 'Fasting BS > 120 (1/0)', min: 0, max: 1, step: 1, default: 0 },
    { key: 'restecg', label: 'Resting ECG (0–2)', min: 0, max: 2, step: 1, default: 0 },
    { key: 'thalach', label: 'Max Heart Rate', min: 60, max: 220, step: 1, default: 150 },
    { key: 'exang', label: 'Exercise Angina (1/0)', min: 0, max: 1, step: 1, default: 0 },
    { key: 'oldpeak', label: 'ST Depression', min: 0, max: 7, step: 0.1, default: 1.0 },
    { key: 'slope', label: 'ST Slope (0–2)', min: 0, max: 2, step: 1, default: 1 },
    { key: 'ca', label: 'Major Vessels (0–3)', min: 0, max: 3, step: 1, default: 0 },
    { key: 'thal', label: 'Thalassemia (0–2)', min: 0, max: 2, step: 1, default: 1 },
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

const CANCER_FIELDS = [
    { key: 'radius_mean', label: 'Radius Mean', min: 0, max: 40, step: 0.01, default: 14.0 },
    { key: 'texture_mean', label: 'Texture Mean', min: 0, max: 50, step: 0.01, default: 19.0 },
    { key: 'perimeter_mean', label: 'Perimeter Mean', min: 0, max: 250, step: 0.1, default: 92.0 },
    { key: 'area_mean', label: 'Area Mean', min: 0, max: 3000, step: 1, default: 655.0 },
    { key: 'smoothness_mean', label: 'Smoothness Mean', min: 0, max: 0.25, step: 0.001, default: 0.096 },
    { key: 'compactness_mean', label: 'Compactness Mean', min: 0, max: 0.5, step: 0.001, default: 0.104 },
    { key: 'concavity_mean', label: 'Concavity Mean', min: 0, max: 0.5, step: 0.001, default: 0.089 },
    { key: 'concave points_mean', label: 'Concave Points Mean', min: 0, max: 0.3, step: 0.001, default: 0.048 },
    { key: 'symmetry_mean', label: 'Symmetry Mean', min: 0, max: 0.4, step: 0.001, default: 0.181 },
    { key: 'fractal_dimension_mean', label: 'Fractal Dim. Mean', min: 0, max: 0.1, step: 0.001, default: 0.063 },
    { key: 'radius_worst', label: 'Radius Worst', min: 0, max: 50, step: 0.01, default: 16.0 },
    { key: 'texture_worst', label: 'Texture Worst', min: 0, max: 80, step: 0.01, default: 25.0 },
    { key: 'perimeter_worst', label: 'Perimeter Worst', min: 0, max: 300, step: 0.1, default: 107.0 },
    { key: 'area_worst', label: 'Area Worst', min: 0, max: 5000, step: 1, default: 880.0 },
    { key: 'smoothness_worst', label: 'Smoothness Worst', min: 0, max: 0.4, step: 0.001, default: 0.132 },
    { key: 'compactness_worst', label: 'Compactness Worst', min: 0, max: 1.5, step: 0.001, default: 0.254 },
    { key: 'concavity_worst', label: 'Concavity Worst', min: 0, max: 1.5, step: 0.001, default: 0.272 },
    { key: 'concave points_worst', label: 'Concave Points Worst', min: 0, max: 0.4, step: 0.001, default: 0.115 },
    { key: 'symmetry_worst', label: 'Symmetry Worst', min: 0, max: 0.7, step: 0.001, default: 0.290 },
    { key: 'fractal_dimension_worst', label: 'Fractal Dim. Worst', min: 0, max: 0.25, step: 0.001, default: 0.084 },
]

const KIDNEY_FIELDS = [
    { key: 'age', label: 'Age (years)', min: 1, max: 100, step: 1, default: 48 },
    { key: 'bp', label: 'Blood Pressure (mmHg)', min: 50, max: 180, step: 1, default: 80 },
    { key: 'sg', label: 'Specific Gravity', min: 1.005, max: 1.030, step: 0.001, default: 1.020 },
    { key: 'al', label: 'Albumin (0–5)', min: 0, max: 5, step: 1, default: 0 },
    { key: 'su', label: 'Sugar (0–5)', min: 0, max: 5, step: 1, default: 0 },
    { key: 'bgr', label: 'Blood Glucose (mg/dl)', min: 22, max: 500, step: 1, default: 121 },
    { key: 'bu', label: 'Blood Urea (mg/dl)', min: 1, max: 400, step: 1, default: 36 },
    { key: 'sc', label: 'Serum Creatinine (mg/dl)', min: 0.4, max: 76, step: 0.1, default: 1.2 },
    { key: 'sod', label: 'Sodium (mEq/L)', min: 100, max: 165, step: 1, default: 137 },
    { key: 'pot', label: 'Potassium (mEq/L)', min: 2, max: 10, step: 0.1, default: 4.6 },
    { key: 'hemo', label: 'Hemoglobin (g/dl)', min: 3, max: 18, step: 0.1, default: 13.3 },
    { key: 'pcv', label: 'Packed Cell Volume', min: 9, max: 54, step: 1, default: 40 },
    { key: 'wc', label: 'WBC Count (cells/μL)', min: 2200, max: 26400, step: 100, default: 7800 },
    { key: 'rc', label: 'RBC Count (millions/μL)', min: 2, max: 8, step: 0.1, default: 4.7 },
    { key: 'htn', label: 'Hypertension (1/0)', min: 0, max: 1, step: 1, default: 0 },
    { key: 'dm', label: 'Diabetes Mellitus (1/0)', min: 0, max: 1, step: 1, default: 0 },
]

const DISEASE_CONFIG: Record<DiseaseTab, { label: string; icon: string; fields: typeof HEART_FIELDS }> = {
    heart: { label: 'Heart Disease', icon: '♥', fields: HEART_FIELDS },
    diabetes: { label: 'Diabetes', icon: '⬡', fields: DIABETES_FIELDS },
    cancer: { label: 'Cancer', icon: '⬟', fields: CANCER_FIELDS },
    kidney: { label: 'Kidney', icon: '⬠', fields: KIDNEY_FIELDS },
}

// ── Helper: extract readable error message ────────────────────────────────────
function extractError(err: unknown): string {
    const ax = err as { response?: { data?: { detail?: unknown } }; message?: string }
    const detail = ax?.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
        // FastAPI 422 detail is an array of validation error objects
        return detail.map((d: { loc?: string[]; msg?: string }) =>
            `${d.loc?.slice(-1)[0] ?? 'field'}: ${d.msg ?? 'invalid'}`
        ).join(', ')
    }
    return ax?.message ?? 'Prediction failed. Check that the API is running.'
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function Predict() {
    const [tab, setTab] = useState<DiseaseTab>('heart')
    const { fields } = DISEASE_CONFIG[tab]

    const [values, setValues] = useState<Record<string, number>>(() =>
        Object.fromEntries(fields.map((f) => [f.key, f.default]))
    )
    const [result, setResult] = useState<PredictionResponse | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleTabChange = (newTab: DiseaseTab) => {
        setTab(newTab)
        setResult(null)
        setError('')
        const { fields: newFields } = DISEASE_CONFIG[newTab]
        setValues(Object.fromEntries(newFields.map((f) => [f.key, f.default])))
    }

    const handleChange = (key: string, value: string) =>
        setValues((v) => ({ ...v, [key]: parseFloat(value) }))

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            const { data } = await predictAPI.predict(tab, values)
            setResult(data)
        } catch (err: unknown) {
            setError(extractError(err))
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

            {/* Disease tabs */}
            <div className="tab-bar" style={{ maxWidth: 560, marginBottom: 24 }}>
                {(Object.entries(DISEASE_CONFIG) as [DiseaseTab, typeof DISEASE_CONFIG[DiseaseTab]][]).map(([key, cfg]) => (
                    <button
                        key={key}
                        className={`tab ${tab === key ? 'active' : ''}`}
                        onClick={() => handleTabChange(key)}
                    >
                        {cfg.icon} {cfg.label}
                    </button>
                ))}
            </div>

            <div className="grid-2" style={{ alignItems: 'start' }}>
                {/* Input form */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Patient Biomarkers</span>
                        <span style={{ fontSize: '0.75rem', color: '#8b9ab5' }}>
                            {fields.length} features
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
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="card-header">
                                <span className="card-title">Risk Assessment</span>
                                <RiskBadge category={result.risk_category} size="sm" />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
                                <RiskGauge category={result.risk_category} score={result.risk_score} size={180} />
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
                                    CLINICAL ACTION — {result.clinical_action?.urgency?.toUpperCase() ?? 'REVIEW'}
                                </p>
                                <p style={{ fontSize: '0.8125rem', color: '#f0f4f8' }}>
                                    {result.clinical_action?.action ?? 'Consult a physician.'}
                                </p>
                            </div>
                            {result.confidence_interval && (
                                <p style={{ fontSize: '0.75rem', color: '#4a5568', marginTop: 10 }}>
                                    95% CI: [{(result.confidence_interval[0] * 100).toFixed(1)}% —{' '}
                                    {(result.confidence_interval[1] * 100).toFixed(1)}%]
                                </p>
                            )}
                        </div>
                        {/* SHAP / top features */}
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Feature Explanations (SHAP)</span>
                            </div>
                            {result.top_features?.length ? (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
                                    {result.top_features.map((f: TopFeature) => {
                                        const isRisk = f.direction === 'increases_risk'
                                        const barWidth = Math.min(100, Math.abs(f.shap_value) * 30)
                                        return (
                                            <div key={f.feature} style={{ fontSize: '0.8rem' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                                    <span style={{ color: '#c8d6ef' }}>
                                                        {f.rank}. {f.feature.replace(/_/g, ' ')}
                                                    </span>
                                                    <span style={{ color: isRisk ? '#f97316' : '#22c55e', fontWeight: 600 }}>
                                                        {isRisk ? '▲ risk' : '▼ risk'}
                                                    </span>
                                                </div>
                                                <div style={{ background: 'rgba(255,255,255,0.06)', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                                                    <div style={{
                                                        width: `${barWidth}%`,
                                                        height: '100%',
                                                        background: isRisk
                                                            ? 'linear-gradient(90deg,#f97316,#ef4444)'
                                                            : 'linear-gradient(90deg,#22c55e,#14b8a6)',
                                                        borderRadius: 4,
                                                        transition: 'width 0.4s ease',
                                                    }} />
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            ) : (
                                <p style={{ color: '#4a5568', fontSize: '0.875rem' }}>
                                    SHAP explanations not available for this prediction.
                                </p>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
                        <div style={{ fontSize: '3rem', marginBottom: 16, opacity: 0.3 }}>⚕</div>
                        <p style={{ color: '#4a5568' }}>
                            Fill in patient biomarkers and click Run Prediction to see the AI risk assessment here.
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}
