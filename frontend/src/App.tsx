import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Analytics from './pages/Analytics'
import Patients from './pages/Patients'
import { onAuthLogout } from './services/api'

// Wrapper component to handle auth logout events
function AuthLogoutHandler({ children }: { children: React.ReactNode }) {
    const navigate = useNavigate()
    
    useEffect(() => {
        return onAuthLogout(() => {
            navigate('/login', { replace: true })
        })
    }, [navigate])
    
    return <>{children}</>
}

export default function App() {
    return (
        <BrowserRouter>
            <AuthLogoutHandler>
                <Routes>
                    {/* Public */}
                    <Route path="/login" element={<Login />} />

                    {/* Protected */}
                    <Route element={<ProtectedRoute />}>
                        <Route
                            path="/*"
                            element={
                                <div className="layout">
                                    <Navbar />
                                    <main className="main-content">
                                        <Routes>
                                            <Route path="/" element={<Dashboard />} />
                                            <Route path="/predict" element={<Predict />} />
                                            <Route path="/analytics" element={<Analytics />} />
                                            <Route path="/patients" element={<Patients />} />
                                            <Route path="*" element={<Navigate to="/" replace />} />
                                        </Routes>
                                    </main>
                                </div>
                            }
                        />
                    </Route>
                </Routes>
            </AuthLogoutHandler>
        </BrowserRouter>
    )
}
