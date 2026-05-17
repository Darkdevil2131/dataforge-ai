import { clsx } from 'clsx'
import { ExternalLink, Star, Database, Layers, BookOpen, TrendingUp } from 'lucide-react'
import { Badge, IntentBadge, SectionLabel } from './UI.jsx'

const DIFF = {
  beginner:     { cls: 'badge-green',  label: 'Beginner' },
  intermediate: { cls: 'badge-amber',  label: 'Intermediate' },
  advanced:     { cls: 'badge-rose',   label: 'Advanced' },
}

function DatasetCard({ ds, rank }) {
  const diff = DIFF[ds.difficulty] || DIFF.intermediate
  const score = Math.round((ds.relevance_score || 0.7) * 100)

  return (
    <a href={ds.url} target="_blank" rel="noopener noreferrer"
      style={{ textDecoration: 'none', display: 'block' }}
      className={clsx('card-3d group', rank === 0 && 'ring-1 ring-indigo-500/30')}
    >
      <div style={{ padding: '20px 24px' }}>
        {rank === 0 && (
          <div style={{ marginBottom: 10 }}>
            <span className="badge-indigo" style={{ fontSize: 10 }}>
              <Star size={9} style={{ display: 'inline', marginRight: 3 }} />Best Match
            </span>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
          <p style={{ fontWeight: 700, fontSize: 14, color: '#e2e8f0', lineHeight: 1.3, fontFamily: 'DM Sans, system-ui', margin: 0 }}>
            {ds.name}
            <ExternalLink size={11} style={{ display: 'inline', marginLeft: 5, opacity: 0.4, verticalAlign: 'middle' }} />
          </p>
          <span className={diff.cls} style={{ whiteSpace: 'nowrap', fontSize: 10 }}>{diff.label}</span>
        </div>

        {ds.description && (
          <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.5, marginBottom: 12, fontFamily: 'DM Sans', margin: '0 0 12px' }}>
            {String(ds.description).slice(0, 140)}{ds.description?.length > 140 ? '…' : ''}
          </p>
        )}

        {/* Stats */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          {ds.rows && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#334155' }}>
              <Database size={11} color="#475569" />
              {Number(ds.rows).toLocaleString()} rows
            </div>
          )}
          {ds.features && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#334155' }}>
              <Layers size={11} color="#475569" />
              {ds.features} features
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#6366f1' }}>
            <BookOpen size={11} />
            {ds.source || 'Kaggle'}
          </div>
        </div>

        {/* Tags */}
        {ds.tags?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
            {ds.tags.slice(0, 5).map(t => (
              <span key={t} className="badge-slate" style={{ fontSize: 10 }}>{t}</span>
            ))}
            {ds.imbalanced && <span className="badge-amber" style={{ fontSize: 10 }}>Imbalanced</span>}
            {ds.time_series && <span className="badge-indigo" style={{ fontSize: 10 }}>Time Series</span>}
          </div>
        )}

        {/* Recommended models */}
        {ds.recommended_models?.length > 0 && (
          <div style={{ borderTop: '1px solid rgba(148,163,184,0.06)', paddingTop: 10 }}>
            <p style={{ fontSize: 10, color: '#334155', marginBottom: 6, fontFamily: 'DM Sans' }}>Recommended models</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {ds.recommended_models.map(m => (
                <span key={m} style={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', padding: '2px 8px', borderRadius: 4, background: 'rgba(99,102,241,0.08)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.15)' }}>
                  {m}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Relevance bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12 }}>
          <span style={{ fontSize: 10, color: '#1e293b', fontFamily: 'DM Sans' }}>Match</span>
          <div style={{ flex: 1, height: 3, borderRadius: 2, background: 'rgba(30,41,59,0.8)', overflow: 'hidden' }}>
            <div style={{ width: `${score}%`, height: '100%', background: 'linear-gradient(90deg, #6366f1, #8b5cf6)', borderRadius: 2 }} />
          </div>
          <span style={{ fontSize: 10, color: '#475569', fontFamily: 'JetBrains Mono, monospace', width: 28, textAlign: 'right' }}>{score}%</span>
        </div>
      </div>
    </a>
  )
}

function LearningPath({ path }) {
  if (!path?.steps?.length) return null
  return (
    <div className="glass" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <TrendingUp size={14} color="#818cf8" />
        <SectionLabel>{path.title || 'Your Learning Path'}</SectionLabel>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {path.steps.map((step, i) => (
          <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{ width: 24, height: 24, borderRadius: '50%', background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: '#818cf8' }}>{step.step}</span>
            </div>
            <div>
              <p style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', margin: '0 0 2px', fontFamily: 'DM Sans' }}>{step.dataset}</p>
              <p style={{ fontSize: 11, color: '#475569', margin: 0, fontFamily: 'DM Sans' }}>{step.task}</p>
              {step.why && <p style={{ fontSize: 11, color: '#334155', margin: '2px 0 0', fontStyle: 'italic', fontFamily: 'DM Sans' }}>{step.why}</p>}
            </div>
          </div>
        ))}
      </div>
      {path.recommended_models?.length > 0 && (
        <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid rgba(148,163,184,0.06)' }}>
          <p style={{ fontSize: 10, color: '#334155', marginBottom: 8, fontFamily: 'DM Sans' }}>RECOMMENDED MODELS</p>
          {path.recommended_models.map((m, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: '#475569', minWidth: 14, fontFamily: 'JetBrains Mono' }}>{i + 1}</span>
              <div>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1', fontFamily: 'DM Sans' }}>{m.model}</span>
                <span style={{ fontSize: 11, color: '#334155', marginLeft: 8, fontFamily: 'DM Sans' }}>{m.why}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function RecommendResults({ result }) {
  if (!result) return null
  const { intent, recommendations, learning_path } = result
  if (!recommendations?.length) return (
    <div className="glass" style={{ padding: 40, textAlign: 'center' }}>
      <p style={{ color: '#334155', fontFamily: 'DM Sans' }}>No datasets found. Try a different description.</p>
    </div>
  )

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Intent strip */}
      {intent?.intent && (
        <div className="glass" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: '#475569', fontFamily: 'DM Sans' }}>Detected goal:</span>
          <IntentBadge intent={intent.intent} />
          <span style={{ fontSize: 11, color: '#334155', fontFamily: 'JetBrains Mono, monospace', marginLeft: 'auto' }}>
            {Math.round((intent.confidence || 0) * 100)}% confidence
          </span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {/* Dataset cards — 2/3 width feel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, gridColumn: learning_path ? '1' : '1/-1' }}>
          <SectionLabel>Matching Datasets ({recommendations.length})</SectionLabel>
          {recommendations.map((ds, i) => <DatasetCard key={ds.id || i} ds={ds} rank={i} />)}
        </div>

        {/* Learning path sidebar */}
        {learning_path && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <SectionLabel>Learning Path</SectionLabel>
            <LearningPath path={learning_path} />
          </div>
        )}
      </div>
    </div>
  )
}
