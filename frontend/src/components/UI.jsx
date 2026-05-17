import { clsx } from 'clsx'

export function Card({ children, className = '', ...props }) {
  return (
    <div className={clsx('card', className)} {...props}>
      {children}
    </div>
  )
}

export function Badge({ variant = 'slate', children }) {
  const map = {
    indigo: 'badge-indigo',
    green: 'badge-green',
    amber: 'badge-amber',
    rose: 'badge-rose',
    slate: 'badge-slate',
  }
  return <span className={map[variant] || 'badge-slate'}>{children}</span>
}

export function QualityScore({ score, label = 'Quality Score', size = 'md' }) {
  const color = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#f43f5e'
  const ring = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#f43f5e'
  const r = size === 'lg' ? 38 : 28
  const cx = size === 'lg' ? 44 : 34
  const dim = size === 'lg' ? 88 : 68
  const stroke = size === 'lg' ? 5 : 4
  const circ = 2 * Math.PI * r
  const dash = (score / 100) * circ

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={dim} height={dim} viewBox={`0 0 ${dim} ${dim}`}>
        <circle cx={cx} cy={cx} r={r} fill="none" stroke="#1e293b" strokeWidth={stroke} />
        <circle
          cx={cx} cy={cx} r={r}
          fill="none"
          stroke={ring}
          strokeWidth={stroke}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cx})`}
          style={{ transition: 'stroke-dasharray 0.8s ease' }}
        />
        <text
          x={cx} y={cx + (size === 'lg' ? 6 : 4)}
          textAnchor="middle"
          fill={color}
          fontSize={size === 'lg' ? '14' : '11'}
          fontWeight="600"
          fontFamily="DM Sans"
        >
          {Math.round(score)}
        </text>
      </svg>
      {label && <span className="text-xs text-slate-500">{label}</span>}
    </div>
  )
}

export function Skeleton({ className = '' }) {
  return <div className={clsx('shimmer rounded-lg bg-slate-800', className)} />
}

export function Divider() {
  return <div className="border-t border-slate-800 my-1" />
}

export function SectionLabel({ children }) {
  return <p className="section-label mb-3">{children}</p>
}

export function StatusDot({ status }) {
  const map = {
    auto_applied: 'bg-emerald-400',
    suggested: 'bg-amber-400',
    flagged: 'bg-rose-400',
  }
  return <span className={clsx('inline-block w-2 h-2 rounded-full', map[status] || 'bg-slate-500')} />
}

export function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 90 ? 'bg-emerald-500' : pct >= 70 ? 'bg-amber-400' : 'bg-rose-500'
  return (
    <div className="flex items-center gap-2">
      <div className="quality-bar flex-1">
        <div className={clsx('quality-bar-fill', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {Icon && <Icon className="w-10 h-10 text-slate-700 mb-4" />}
      <p className="text-slate-400 font-medium">{title}</p>
      {description && <p className="text-slate-600 text-sm mt-1 max-w-xs">{description}</p>}
    </div>
  )
}

export function IntentBadge({ intent }) {
  const labels = {
    fraud_detection: { label: 'Fraud Detection', variant: 'rose' },
    classification: { label: 'Classification', variant: 'indigo' },
    regression: { label: 'Regression', variant: 'indigo' },
    forecasting: { label: 'Forecasting', variant: 'amber' },
    analytics: { label: 'Analytics', variant: 'green' },
    recommendation: { label: 'Recommendation', variant: 'indigo' },
    nlp: { label: 'NLP', variant: 'amber' },
    general_ml: { label: 'General ML', variant: 'slate' },
  }
  const info = labels[intent] || { label: intent, variant: 'slate' }
  return <Badge variant={info.variant}>{info.label}</Badge>
}
