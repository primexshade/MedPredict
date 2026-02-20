type RiskCategory = 'LOW' | 'BORDERLINE' | 'MODERATE' | 'HIGH' | 'CRITICAL'

const COLOR_MAP: Record<RiskCategory, string> = {
    LOW: '#22c55e',
    BORDERLINE: '#eab308',
    MODERATE: '#f97316',
    HIGH: '#ef4444',
    CRITICAL: '#dc2626',
}

interface Props {
    category: RiskCategory
    score: number      // 0–1
    size?: number
}

export default function RiskGauge({ category, score, size = 180 }: Props) {
    const color = COLOR_MAP[category]
    const r = 70
    const cx = size / 2
    const cy = size / 2

    // Arc spans from -220° to 40° (240° total sweep)
    const startAngle = -220 * (Math.PI / 180)
    const sweepAngle = 240 * (Math.PI / 180)
    const endFull = startAngle + sweepAngle

    const toXY = (angle: number) => ({
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
    })

    const buildArc = (start: number, end: number) => {
        const s = toXY(start)
        const e = toXY(end)
        const large = end - start > Math.PI ? 1 : 0
        return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
    }

    const filledEnd = startAngle + sweepAngle * Math.min(Math.max(score, 0), 1)

    return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            {/* Background track */}
            <path
                d={buildArc(startAngle, endFull)}
                fill="none"
                stroke="rgba(255,255,255,0.07)"
                strokeWidth={10}
                strokeLinecap="round"
            />
            {/* Filled arc */}
            <path
                d={buildArc(startAngle, filledEnd)}
                fill="none"
                stroke={color}
                strokeWidth={10}
                strokeLinecap="round"
                style={{ filter: `drop-shadow(0 0 8px ${color})` }}
            />
            {/* Score text */}
            <text
                x={cx} y={cy - 8}
                textAnchor="middle"
                fill={color}
                fontSize={24}
                fontWeight={700}
                fontFamily="Inter, sans-serif"
            >
                {(score * 100).toFixed(0)}%
            </text>
            {/* Category label */}
            <text
                x={cx} y={cy + 14}
                textAnchor="middle"
                fill="#8b9ab5"
                fontSize={11}
                fontFamily="Inter, sans-serif"
                textDecoration="none"
                letterSpacing="1"
            >
                {category}
            </text>
        </svg>
    )
}
