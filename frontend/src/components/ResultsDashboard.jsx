import { useState } from 'react'
import {
  BarChart2, ShieldCheck, Zap, AlertTriangle,
  Download, ChevronDown, ChevronRight,
  CheckCircle2, Clock, AlertCircle, Info,
  TrendingUp, Database, Layers, FileText
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts'
import {
  Card, Badge, QualityScore, SectionLabel, StatusDot,
  ConfidenceBar, IntentBadge, Skeleton
} from './UI.jsx'

// ── Custom tooltip for recharts ─────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="font-medium">{p.value}</p>
      ))}
    </div>
  )
}

// ── Transformation log entry ─────────────────────────────────────────────────
function LogEntry({ entry }) {
  const [open, setOpen] = useState(false)
  const icons = {
    auto_applied: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />,
    suggested: <Clock className="w-3.5 h-3.5 text-amber-400" />,
    flagged: <AlertCircle className="w-3.5 h-3.5 text-rose-400" />,
  }
  return (
    <div className="border border-slate-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-800/50 transition-colors"
      >
        <StatusDot status={entry.status} />
        <span className="flex-1 text-sm text-slate-300 font-mono truncate">{entry.action}</span>
        <span className="text-xs text-slate-500 font-mono mr-2">{entry.column}</span>
        <ConfidenceBar value={entry.confidence} />
        <span className="ml-2 text-slate-600">{open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}</span>
      </button>
      {open && (
        <div className="px-4 pb-3 pt-1 bg-slate-900/50 border-t border-slate-800">
          <p className="text-xs text-slate-400">{entry.details || entry.reason}</p>
          <div className="flex gap-3 mt-2">
            <span className="text-xs text-slate-600">Rows affected: <span className="text-slate-400">{entry.rows_affected}</span></span>
            <span className="text-xs text-slate-600">Reason: <span className="text-slate-400">{entry.reason}</span></span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Dataset recommendation card ──────────────────────────────────────────────
function DatasetCard({ ds, index }) {
  const diffColor = {
    beginner: 'badge-green',
    intermediate: 'badge-amber',
    advanced: 'badge-rose',
  }
  return (
    <a
      href={ds.url}
      target="_blank"
      rel="noopener noreferrer"
      className="card p-4 hover:bg-slate-800/80 transition-all duration-200 hover:border-slate-700 block group"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <p className="text-sm font-semibold text-slate-200 group-hover:text-indigo-300 transition-colors leading-tight">
          {ds.name}
        </p>
        <span className={clsx('badge shrink-0', diffColor[ds.difficulty] || 'badge-slate')}>
          {ds.difficulty}
        </span>
      </div>
      <p className="text-xs text-slate-500 mb-3 line-clamp-2">{ds.description}</p>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {ds.tags?.slice(0, 4).map(tag => (
          <span key={tag} className="badge badge-slate text-slate-500">{tag}</span>
        ))}
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-600">
        <span>{ds.rows?.toLocaleString()} rows</span>
        <span>·</span>
        <span>{ds.features} features</span>
        <span>·</span>
        <span className="text-indigo-400">{ds.source}</span>
        <span className="ml-auto text-slate-700">
          {Math.round(ds.relevance_score * 100)}% match
        </span>
      </div>
    </a>
  )
}

// ── Main results component ───────────────────────────────────────────────────
export function ResultsDashboard({ result, onDownload, downloadLoading, file, prompt }) {
  const [activeTab, setActiveTab] = useState('overview')

  if (!result) return null

  const { quality_scores, report, schema, anomaly_detection, feature_importance, recommendations } = result
  const transformations = report?.transformations || {}
  const allLogs = [
    ...(transformations.auto_applied || []),
    ...(transformations.suggested || []),
    ...(transformations.flagged || []),
  ]

  const TABS = [
    { id: 'overview', label: 'Overview', icon: BarChart2 },
    { id: 'transforms', label: `Transformations (${allLogs.length})`, icon: Zap },
    { id: 'anomalies', label: `Anomalies (${anomaly_detection?.anomaly_count || 0})`, icon: ShieldCheck },
    { id: 'features', label: 'Features', icon: Layers },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'report', label: 'Report', icon: FileText },
  ]

  // Null/missing chart data
  const missingData = Object.entries(schema?.columns || {})
    .filter(([, v]) => v.null_pct > 0)
    .sort((a, b) => b[1].null_pct - a[1].null_pct)
    .slice(0, 10)
    .map(([col, v]) => ({ col: col.slice(0, 12), pct: v.null_pct }))

  const featureData = (feature_importance || [])
    .slice(0, 10)
    .map(f => ({ name: f.feature.slice(0, 14), score: Math.abs(f.importance_score) }))

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Header strip */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <IntentBadge intent={result.intent?.intent} />
          <span className="text-sm text-slate-500">
            Confidence: <span className="text-slate-300 font-mono">
              {Math.round((result.intent?.confidence || 0) * 100)}%
            </span>
          </span>
        </div>
        <div className="flex gap-2">
          {['csv', 'excel'].map(fmt => (
            <button
              key={fmt}
              onClick={() => onDownload(fmt)}
              disabled={downloadLoading}
              className="btn-secondary text-sm py-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Quality score row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 stagger-children">
        <Card className="p-4 flex items-center gap-4">
          <QualityScore score={quality_scores?.before || 0} label="Before" />
          <div className="text-slate-500 text-xl">→</div>
          <QualityScore score={quality_scores?.after || 0} label="After" size="lg" />
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="section-label">Shape</span>
          <div>
            <p className="text-slate-200 font-mono text-sm">{result.shape?.original?.[0]?.toLocaleString()} × {result.shape?.original?.[1]}</p>
            <p className="text-xs text-slate-500">→ {result.shape?.engineered?.[0]?.toLocaleString()} × {result.shape?.engineered?.[1]}</p>
          </div>
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="section-label">Auto-Fixed</span>
          <div className="flex items-end gap-2">
            <span className="text-3xl font-display font-bold text-emerald-400">{transformations.auto_applied?.length || 0}</span>
            <span className="text-slate-500 text-sm mb-1">actions</span>
          </div>
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="section-label">Anomalies</span>
          <div className="flex items-end gap-2">
            <span className="text-3xl font-display font-bold text-amber-400">{anomaly_detection?.anomaly_count || 0}</span>
            <span className="text-slate-500 text-sm mb-1">detected</span>
          </div>
        </Card>
      </div>

      {/* Policy note */}
      {result.policy?.description && (
        <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-indigo-500/5 border border-indigo-500/20">
          <Info className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
          <p className="text-sm text-indigo-300">{result.policy.description}</p>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-slate-900 rounded-xl p-1 overflow-x-auto">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm whitespace-nowrap transition-all',
              activeTab === tab.id
                ? 'bg-slate-700 text-slate-100'
                : 'text-slate-500 hover:text-slate-300'
            )}
          >
            <tab.icon className="w-3.5 h-3.5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="animate-fade-in">

        {/* ── OVERVIEW ── */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Missing values chart */}
            {missingData.length > 0 && (
              <Card className="p-5">
                <SectionLabel>Missing Values by Column</SectionLabel>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={missingData} layout="vertical" margin={{ left: 0, right: 20 }}>
                    <XAxis type="number" tick={{ fill: '#475569', fontSize: 10 }} tickFormatter={v => `${v}%`} />
                    <YAxis type="category" dataKey="col" tick={{ fill: '#94a3b8', fontSize: 10 }} width={80} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="pct" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}

            {/* Schema overview */}
            <Card className="p-5">
              <SectionLabel>Column Roles</SectionLabel>
              <div className="space-y-2 max-h-52 overflow-y-auto">
                {Object.entries(schema?.columns || {}).map(([col, info]) => (
                  <div key={col} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300 font-mono truncate max-w-32">{col}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant={info.role === 'identifier' ? 'rose' : info.role === 'potential_target' ? 'green' : 'slate'}>
                        {info.role}
                      </Badge>
                      <span className="text-slate-600 text-xs font-mono">{info.inferred_type}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ── TRANSFORMATIONS ── */}
        {activeTab === 'transforms' && (
          <div className="space-y-3">
            <div className="flex gap-3 text-sm">
              <span className="flex items-center gap-1.5"><StatusDot status="auto_applied" /> {transformations.auto_applied?.length || 0} Auto-applied</span>
              <span className="flex items-center gap-1.5"><StatusDot status="suggested" /> {transformations.suggested?.length || 0} Suggested</span>
              <span className="flex items-center gap-1.5"><StatusDot status="flagged" /> {transformations.flagged?.length || 0} Flagged</span>
            </div>
            {allLogs.length === 0 ? (
              <div className="text-center py-12 text-slate-500">Dataset was already clean — no transformations needed.</div>
            ) : (
              allLogs.map((entry, i) => <LogEntry key={i} entry={entry} />)
            )}
          </div>
        )}

        {/* ── ANOMALIES ── */}
        {activeTab === 'anomalies' && (
          <div className="space-y-4">
            <div className={clsx(
              'rounded-xl p-4 border',
              anomaly_detection?.preserved
                ? 'bg-amber-500/5 border-amber-500/20'
                : 'bg-slate-900 border-slate-800'
            )}>
              <p className="text-sm text-slate-300">{anomaly_detection?.note}</p>
            </div>
            {(anomaly_detection?.anomaly_details || []).map((a, i) => (
              <Card key={i} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-slate-500">Row #{a.row_index}</span>
                  <Badge variant="rose">Score: {a.anomaly_score}</Badge>
                </div>
                <div className="flex flex-wrap gap-3">
                  {Object.entries(a.data || {}).slice(0, 6).map(([k, v]) => (
                    <div key={k} className="text-xs">
                      <span className="text-slate-600">{k}: </span>
                      <span className="text-slate-300 font-mono">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </Card>
            ))}
            {!anomaly_detection?.anomaly_details?.length && (
              <div className="text-center py-12 text-slate-500">No anomaly details available.</div>
            )}
          </div>
        )}

        {/* ── FEATURES ── */}
        {activeTab === 'features' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {featureData.length > 0 && (
              <Card className="p-5">
                <SectionLabel>Top Feature Importance</SectionLabel>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={featureData} layout="vertical" margin={{ left: 0, right: 20 }}>
                    <XAxis type="number" tick={{ fill: '#475569', fontSize: 10 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} width={90} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="score" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}
            <Card className="p-5">
              <SectionLabel>Engineering Log</SectionLabel>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {(report?.feature_engineering?.details || []).map((f, i) => (
                  <div key={i} className="border border-slate-800 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="indigo">{f.action}</Badge>
                      <span className="text-xs text-slate-500 font-mono">{f.source_column}</span>
                    </div>
                    <p className="text-xs text-slate-500">{f.reason}</p>
                  </div>
                ))}
                {!report?.feature_engineering?.details?.length && (
                  <p className="text-slate-500 text-sm">No features engineered.</p>
                )}
              </div>
            </Card>
          </div>
        )}

        {/* ── DATASETS ── */}
        {activeTab === 'datasets' && (
          <div className="space-y-4">
            <SectionLabel>Recommended Datasets for Your Goal</SectionLabel>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 stagger-children">
              {(recommendations || []).map((ds, i) => (
                <DatasetCard key={ds.id} ds={ds} index={i} />
              ))}
            </div>
            {/* Learning path */}
            {report?.model_recommendations?.length > 0 && (
              <Card className="p-5 mt-4">
                <SectionLabel>Recommended Models</SectionLabel>
                <div className="space-y-2">
                  {report.model_recommendations.map((m, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-slate-400 font-mono shrink-0">{m.priority}</span>
                      <span className="text-sm font-medium text-slate-200">{m.model}</span>
                      <span className="text-xs text-slate-500">{m.why}</span>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ── REPORT ── */}
        {activeTab === 'report' && (
          <div className="space-y-5">
            {/* Risks */}
            {(report?.risks || []).length > 0 && (
              <Card className="p-5">
                <SectionLabel>Risk Assessment</SectionLabel>
                <div className="space-y-2">
                  {report.risks.map((r, i) => (
                    <div key={i} className={clsx(
                      'flex items-start gap-3 p-3 rounded-lg border',
                      r.severity === 'high'
                        ? 'bg-rose-500/5 border-rose-500/20'
                        : 'bg-amber-500/5 border-amber-500/20'
                    )}>
                      <AlertTriangle className={clsx(
                        'w-4 h-4 mt-0.5 shrink-0',
                        r.severity === 'high' ? 'text-rose-400' : 'text-amber-400'
                      )} />
                      <div>
                        <p className="text-sm text-slate-300">{r.message}</p>
                        <p className="text-xs text-slate-500 mt-1">{r.recommendation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Next steps */}
            <Card className="p-5">
              <SectionLabel>Next Steps</SectionLabel>
              <div className="space-y-2">
                {(report?.next_steps || []).map((step, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm text-slate-300">
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            </Card>

            {/* Policy notes */}
            {(report?.policy_notes || []).length > 0 && (
              <Card className="p-5">
                <SectionLabel>Policy Notes ({result.intent?.intent})</SectionLabel>
                <ul className="space-y-1.5">
                  {report.policy_notes.map((note, i) => (
                    <li key={i} className="text-sm text-amber-300/80 flex items-start gap-2">
                      <span className="text-amber-500 mt-0.5">⚠</span> {note}
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
