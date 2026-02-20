import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { ShapContribution } from '../services/api'

interface Props {
    contributions: ShapContribution[]
}

export default function ShapChart({ contributions }: Props) {
    const data = contributions
        .slice(0, 10)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .map((c) => ({
            feature: c.feature,
            value: parseFloat(c.value.toFixed(4)),
            direction: c.direction,
        }))

    return (
        <div>
            <p style={{ fontSize: '0.75rem', color: '#8b9ab5', marginBottom: 12 }}>
                Top feature contributions (SHAP values). <span style={{ color: '#22c55e' }}>Green = increases risk</span>,{' '}
                <span style={{ color: '#64748b' }}>Grey = decreases risk</span>.
            </p>
            <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data} layout="vertical" margin={{ left: 8, right: 32 }}>
                    <XAxis type="number" tick={{ fill: '#8b9ab5', fontSize: 11 }} />
                    <YAxis
                        type="category"
                        dataKey="feature"
                        width={120}
                        tick={{ fill: '#f0f4f8', fontSize: 11 }}
                    />
                    <Tooltip
                        contentStyle={{
                            background: '#141920',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 8,
                            color: '#f0f4f8',
                            fontSize: 12,
                        }}
                        formatter={(v: number) => [v.toFixed(4), 'SHAP']}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {data.map((entry, idx) => (
                            <Cell
                                key={idx}
                                fill={entry.direction === 'increases' ? '#ef4444' : '#64748b'}
                            />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}
