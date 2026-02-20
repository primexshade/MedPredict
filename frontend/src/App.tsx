import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Analytics from './pages/Analytics'
import Patients from './pages/Patients'

export default function App() {
    return (
        <BrowserRouter>
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
        </BrowserRouter>
    )
}
