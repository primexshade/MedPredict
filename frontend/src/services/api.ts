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
export interface ShapContribution {
    feature: string
    value: number
    direction: 'increases' | 'decreases'
}

export interface PredictionResponse {
    disease: string
    probability: number
    risk_category: 'LOW' | 'BORDERLINE' | 'MODERATE' | 'HIGH' | 'CRITICAL'
    composite_score: number
    confidence_interval: [number, number]
    shap_contributions: ShapContribution[]
    plain_english_summary: string
    velocity: number | null
    clinical_action: { action: string; urgency: string }
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
