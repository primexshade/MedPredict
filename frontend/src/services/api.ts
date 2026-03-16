import axios from 'axios'

// ── Axios Instance ────────────────────────────────────────────────────────────
export const api = axios.create({
    baseURL: '/api/v1',
    timeout: 30_000,
    headers: { 'Content-Type': 'application/json' },
})

// Inject Bearer token from localStorage on every request
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
        config.headers.Authorization = `Bearer ${token}`
    }
    return config
})

// On 401 → clear tokens and redirect to login
api.interceptors.response.use(
    (res) => res,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('access_token')
            localStorage.removeItem('refresh_token')
            window.location.href = '/login'
        }
        return Promise.reject(error)
    },
)

// ── Types ─────────────────────────────────────────────────────────────────────
// Matches actual API response fields from /predict/{disease}
export interface TopFeature {
    feature: string
    value: number
    shap_value: number
    direction: 'increases_risk' | 'decreases_risk'
    rank: number
}

export interface PredictionResponse {
    patient_id: string | null
    disease: string
    risk_score: number
    calibrated_probability: number
    risk_category: 'LOW' | 'BORDERLINE' | 'MODERATE' | 'HIGH' | 'CRITICAL'
    confidence_interval: [number, number] | null
    velocity: number | null
    top_features: TopFeature[]
    plain_english_summary: string
    clinical_action: { action: string; timeframe?: string; urgency: string }
    model_version?: string
    cached?: boolean
}

export interface PatientOut {
    id: string
    mrn: string
}

export interface AnalyticsSummaryResponse {
    total_predictions: number
    disease_breakdown: Record<string, number>
    high_risk_count: number
}

export interface ComorbidityRule {
    antecedents: string[]
    consequents: string[]
    support: number
    confidence: number
    lift: number
}

// ── API calls ─────────────────────────────────────────────────────────────────
export const authAPI = {
    login: (email: string, password: string) =>
        api.post<{ access_token: string; refresh_token: string; expires_in: number }>(
            '/auth/login', { email, password }
        ),
    logout: () => api.post('/auth/logout'),
}

export const predictAPI = {
    heart: (payload: Record<string, number | string>) =>
        api.post<PredictionResponse>('/predict/heart', payload),
    diabetes: (payload: Record<string, number | string>) =>
        api.post<PredictionResponse>('/predict/diabetes', payload),
    cancer: (payload: Record<string, number | string>) =>
        api.post<PredictionResponse>('/predict/cancer', payload),
    kidney: (payload: Record<string, number | string>) =>
        api.post<PredictionResponse>('/predict/kidney', payload),
    // Generic dispatcher used by Predict.tsx
    predict: (disease: string, payload: Record<string, number | string>) =>
        api.post<PredictionResponse>(`/predict/${disease}`, payload),
}

export const analyticsAPI = {
    summary: () => api.get<AnalyticsSummaryResponse>('/analytics/summary'),
    clusters: () => api.get('/analytics/clusters'),
    comorbidityRules: () => api.get<{ rules: ComorbidityRule[] }>('/analytics/comorbidity-rules'),
}

export const patientsAPI = {
    list: () => api.get<PatientOut[]>('/patients/'),
    get: (id: string) => api.get<PatientOut>(`/patients/${id}`),
}
