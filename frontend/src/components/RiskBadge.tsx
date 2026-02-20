type RiskCategory = 'LOW' | 'BORDERLINE' | 'MODERATE' | 'HIGH' | 'CRITICAL'

const LABEL_MAP: Record<RiskCategory, string> = {
    LOW: '● Low',
    BORDERLINE: '● Borderline',
    MODERATE: '● Moderate',
    HIGH: '● High',
    CRITICAL: '⚠ Critical',
}

interface Props {
    category: RiskCategory
    size?: 'sm' | 'md' | 'lg'
}

export default function RiskBadge({ category, size = 'md' }: Props) {
    const padding = size === 'sm' ? '3px 8px' : size === 'lg' ? '6px 16px' : '4px 12px'
    const fontSize = size === 'sm' ? '0.7rem' : size === 'lg' ? '0.875rem' : '0.75rem'

    return (
        <span
            className={`risk-badge ${category}`}
            style={{ padding, fontSize }}
        >
            {LABEL_MAP[category]}
        </span>
    )
}
