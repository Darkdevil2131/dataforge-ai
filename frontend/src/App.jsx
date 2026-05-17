import { useState, useRef } from 'react'
import {
  Database, Sparkles, BookOpen, Github, ChevronRight,
  LogOut, User, Menu, X, Zap, Shield, BarChart2
} from 'lucide-react'
import { clsx } from 'clsx'
import { AuthProvider, useAuth } from './context/AuthContext.jsx'
import { AuthPage } from './components/AuthPage.jsx'
import { IntroScreen, useIntro } from './components/IntroScreen.jsx'
import { UploadPanel, RecommendPanel } from './components/UploadPanel.jsx'
import { ResultsDashboard } from './components/ResultsDashboard.jsx'
import { RecommendResults } from './components/RecommendResults.jsx'
import { useProcessing, useRecommend } from './hooks/useDataForge.js'

// ── Navbar ────────────────────────────────────────────────────────────────────
function Navbar({ page, setPage }) {
  const { user, logout } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)

  const navItems = [
    { id: 'process',   label: 'Process',      icon: Database },
    { id: 'recommend', label: 'Datasets',      icon: Sparkles },
    { id: 'docs',      label: 'How It Works',  icon: BookOpen },
  ]

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 100,
      borderBottom: '1px solid rgba(148,163,184,0.06)',
      background: 'rgba(2,4,8,0.85)',
      backdropFilter: 'blur(20px)',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 20px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Logo */}
        <button onClick={() => setPage('home')} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none', cursor: 'pointer' }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(99,102,241,0.3)',
          }}>
            <Database size={14} color="white" />
          </div>
          <span style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 16, color: '#f1f5f9', letterSpacing: '-0.02em' }}>
            DataForge <span style={{ color: '#818cf8' }}>AI</span>
          </span>
        </button>

        {/* Desktop nav */}
        <nav className="hide-mobile" style={{ display: 'flex', gap: 4 }}>
          {navItems.map(item => (
            <button key={item.id} onClick={() => setPage(item.id)} style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px',
              borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
              fontFamily: 'DM Sans, system-ui', transition: 'all 0.15s',
              background: page === item.id ? 'rgba(99,102,241,0.12)' : 'transparent',
              color: page === item.id ? '#818cf8' : '#64748b',
            }}>
              <item.icon size={13} /> {item.label}
            </button>
          ))}
        </nav>

        {/* User */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {user ? (
            <>
              <div className="hide-mobile" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {user.photoURL
                  ? <img src={user.photoURL} alt="" style={{ width: 28, height: 28, borderRadius: '50%', border: '1px solid rgba(148,163,184,0.15)' }} />
                  : <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(99,102,241,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><User size={13} color="#818cf8" /></div>
                }
                <span style={{ fontSize: 13, color: '#64748b', fontFamily: 'DM Sans' }}>
                  {user.displayName?.split(' ')[0] || user.email?.split('@')[0]}
                </span>
              </div>
              <button onClick={logout} className="btn-ghost" style={{ fontSize: 12 }}>
                <LogOut size={13} /> Sign out
              </button>
            </>
          ) : (
            <button onClick={() => setPage('auth')} className="btn-primary" style={{ padding: '7px 16px', fontSize: 13 }}>
              Sign In
            </button>
          )}

          {/* Mobile menu */}
          <button className="hide-desktop" onClick={() => setMobileOpen(o => !o)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', padding: 4 }}>
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {mobileOpen && (
        <div className="hide-desktop" style={{ borderTop: '1px solid rgba(148,163,184,0.06)', padding: '8px 20px 16px' }}>
          {navItems.map(item => (
            <button key={item.id} onClick={() => { setPage(item.id); setMobileOpen(false) }} style={{
              display: 'flex', alignItems: 'center', gap: 10, width: '100%',
              padding: '12px 8px', background: 'none', border: 'none', cursor: 'pointer',
              color: '#94a3b8', fontSize: 14, fontFamily: 'DM Sans', borderBottom: '1px solid rgba(148,163,184,0.04)',
            }}>
              <item.icon size={15} /> {item.label}
            </button>
          ))}
        </div>
      )}
    </header>
  )
}

// ── Hero Section ──────────────────────────────────────────────────────────────
function Hero({ onStart }) {
  return (
    <div style={{ position: 'relative', overflow: 'hidden', minHeight: '88vh', display: 'flex', alignItems: 'center' }}>
      {/* Grid background */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
      }} />
      {/* Gradient fade bottom */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '40%', background: 'linear-gradient(to bottom, transparent, #020408)' }} />
      {/* Orbs */}
      <div style={{ position: 'absolute', top: '10%', left: '15%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)', filter: 'blur(60px)' }} />
      <div style={{ position: 'absolute', top: '20%', right: '10%', width: 300, height: 300, borderRadius: '50%', background: 'radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)', filter: 'blur(50px)' }} />

      <div style={{ position: 'relative', maxWidth: 1200, margin: '0 auto', padding: '60px 24px', width: '100%' }}>
        <div style={{ maxWidth: 760, margin: '0 auto', textAlign: 'center' }}>
          {/* Pill */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '6px 16px', borderRadius: 40, border: '1px solid rgba(99,102,241,0.2)', background: 'rgba(99,102,241,0.06)', marginBottom: 40, fontSize: 12, color: '#818cf8', fontFamily: 'DM Sans', letterSpacing: '0.04em' }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#818cf8', animation: 'pulse-ring 2s ease-out infinite' }} />
            6 Trained ML Models · Real Datasets · Zero API Cost
          </div>

          {/* Headline */}
          <h1 style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 'clamp(36px, 6vw, 68px)', lineHeight: 1.05, letterSpacing: '-0.03em', color: '#f1f5f9', marginBottom: 28 }}>
            From messy data to
            <br />
            <span style={{ background: 'linear-gradient(135deg, #818cf8 0%, #c4b5fd 50%, #818cf8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundSize: '200%', animation: 'shimmer 4s linear infinite' }}>
              production-ready
            </span>{' '}
            <span style={{ color: '#f1f5f9' }}>in one prompt.</span>
          </h1>

          {/* Sub */}
          <p style={{ fontSize: 18, color: '#64748b', lineHeight: 1.7, maxWidth: 560, margin: '0 auto 44px', fontFamily: 'DM Sans' }}>
            Upload any dataset. Describe your goal. DataForge AI cleans it intelligently,
            engineers features, detects anomalies, and gives you production-ready data — instantly.
          </p>

          {/* CTAs */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginBottom: 72 }}>
            <button onClick={() => onStart('process')} className="btn-primary" style={{ padding: '13px 28px', fontSize: 15 }}>
              <Database size={16} /> Start Processing <ChevronRight size={15} />
            </button>
            <button onClick={() => onStart('recommend')} className="btn-secondary" style={{ padding: '13px 24px', fontSize: 15 }}>
              <Sparkles size={16} /> Find Datasets
            </button>
          </div>

          {/* Stats row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 8 }}>
            {[
              { v: '92.9%', l: 'Intent Accuracy' },
              { v: 'AUC 1.0', l: 'Fraud Detection' },
              { v: 'R² 0.92', l: 'Price Prediction' },
              { v: '6 Models', l: 'Trained on Real Data' },
              { v: '5 Datasets', l: 'Production Quality' },
            ].map(s => (
              <div key={s.v} style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid rgba(148,163,184,0.07)', background: 'rgba(15,23,42,0.6)', textAlign: 'center', minWidth: 110 }}>
                <div style={{ fontFamily: 'Syne, system-ui', fontWeight: 700, fontSize: 16, color: '#818cf8' }}>{s.v}</div>
                <div style={{ fontSize: 11, color: '#334155', fontFamily: 'DM Sans', marginTop: 2 }}>{s.l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── How It Works ──────────────────────────────────────────────────────────────
function HowItWorks() {
  const steps = [
    { icon: Database, title: 'Upload & Describe', body: 'Drop any CSV, Excel, JSON, or Parquet file. Type one sentence describing your goal.' },
    { icon: Zap, title: 'AI Understands Intent', body: 'Our NLP model classifies your goal and selects the exact right cleaning policy for your use case.' },
    { icon: Shield, title: '6-Stage Pipeline', body: 'Schema inference → cleaning → anomaly detection → feature engineering → quality report.' },
    { icon: BarChart2, title: 'Download & Deploy', body: 'Get production-ready CSV, Excel, or Parquet with full transformation log and model recommendations.' },
  ]
  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '80px 24px' }}>
      <div style={{ textAlign: 'center', marginBottom: 60 }}>
        <p className="section-label" style={{ marginBottom: 12 }}>Pipeline</p>
        <h2 style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 'clamp(28px, 4vw, 42px)', color: '#f1f5f9', letterSpacing: '-0.02em', marginBottom: 16 }}>
          Six engines. One pipeline.
        </h2>
        <p style={{ color: '#475569', maxWidth: 480, margin: '0 auto', fontFamily: 'DM Sans', lineHeight: 1.6 }}>
          Not one model trying to do everything. Specialized engines each doing exactly what they do best.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16, marginBottom: 48 }}>
        {steps.map((s, i) => (
          <div key={i} className="card-3d" style={{ padding: 28 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
              <s.icon size={20} color="#818cf8" />
            </div>
            <div style={{ fontFamily: 'Syne, system-ui', fontWeight: 700, fontSize: 15, color: '#e2e8f0', marginBottom: 8 }}>{s.title}</div>
            <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, fontFamily: 'DM Sans' }}>{s.body}</p>
          </div>
        ))}
      </div>

      {/* Model table */}
      <div className="glass" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid rgba(148,163,184,0.06)' }}>
          <span className="section-label">Trained Models & Accuracy</span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, fontFamily: 'DM Sans' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(148,163,184,0.06)' }}>
                {['Model', 'Algorithm', 'Dataset', 'Score'].map(h => (
                  <th key={h} style={{ padding: '12px 20px', textAlign: 'left', color: '#334155', fontWeight: 500, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['Intent Classifier',    'TF-IDF + LogReg',         '210 samples / 8 classes', 'CV 92.9%'],
                ['Fraud Detector',       'GradientBoosting + SMOTE','10K rows real fraud data', 'AUC 1.00'],
                ['Anomaly Detector',     'IsolationForest + LOF',   'Ensemble on fraud data',  'AUC 1.00'],
                ['Churn Classifier',     'RandomForest + SMOTE',    '7K customer records',     'Acc 71.7%'],
                ['Price Regressor',      'GradientBoosting',        '1.5K house records',      'R² 0.92'],
                ['Cancer Classifier',    'RF + SVM Ensemble',       '569 clinical samples',    'AUC 0.997'],
              ].map(([m, a, d, s]) => (
                <tr key={m} style={{ borderBottom: '1px solid rgba(148,163,184,0.04)', transition: 'background 0.15s' }}>
                  <td style={{ padding: '12px 20px', color: '#cbd5e1', fontWeight: 500 }}>{m}</td>
                  <td style={{ padding: '12px 20px', color: '#6366f1', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, whiteSpace: 'nowrap' }}>{a}</td>
                  <td style={{ padding: '12px 20px', color: '#334155', whiteSpace: 'nowrap' }}>{d}</td>
                  <td style={{ padding: '12px 20px' }}><span className="badge-green">{s}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Main App Shell ────────────────────────────────────────────────────────────
function AppShell() {
  const { user, loading } = useAuth()
  const { show: showIntro, done: introDone } = useIntro()
  const [page, setPage]   = useState('home')
  const [dlLoading, setDlLoading] = useState(false)
  const processing = useProcessing()
  const recommend  = useRecommend()

  if (loading) return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#020408' }}>
      <div style={{ width: 32, height: 32, border: '2px solid rgba(99,102,241,0.2)', borderTopColor: '#6366f1', borderRadius: '50%' }} className="animate-spin" />
    </div>
  )

  const goTo = (p) => {
    if (!user && p !== 'home' && p !== 'docs') { setPage('auth'); return }
    setPage(p)
  }

  const handleDownload = async (format) => {
    setDlLoading(true)
    await processing.download(format)
    setDlLoading(false)
  }

  return (
    <div style={{ minHeight: '100vh', background: '#020408' }}>
      {showIntro && <IntroScreen onDone={introDone} />}

      <Navbar page={page} setPage={goTo} />

      {/* HOME */}
      {page === 'home' && (
        <>
          <Hero onStart={goTo} />
          <HowItWorks />
        </>
      )}

      {/* AUTH */}
      {page === 'auth' && !user && <AuthPage />}

      {/* Redirect to home after auth */}
      {page === 'auth' && user && (() => { setPage('process'); return null })()}

      {/* PROCESS */}
      {page === 'process' && (
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
          <div style={{ marginBottom: 32 }}>
            <h2 style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 28, color: '#f1f5f9', letterSpacing: '-0.02em', marginBottom: 8 }}>Process Dataset</h2>
            <p style={{ color: '#475569', fontFamily: 'DM Sans', fontSize: 14 }}>Upload your raw data and describe your ML goal. The full pipeline runs automatically.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,2fr) minmax(0,3fr)', gap: 24, alignItems: 'start' }}>
            <div className="glass" style={{ padding: 24, position: 'sticky', top: 76 }}>
              <UploadPanel onSubmit={processing.run} loading={processing.loading} />
            </div>
            <div>
              {processing.error && (
                <div className="glass" style={{ padding: 16, marginBottom: 16, borderColor: 'rgba(244,63,94,0.2)', background: 'rgba(244,63,94,0.05)' }}>
                  <p style={{ color: '#fb7185', fontWeight: 600, fontSize: 13, marginBottom: 4 }}>Processing failed</p>
                  <p style={{ color: 'rgba(251,113,133,0.7)', fontSize: 12 }}>{processing.error}</p>
                </div>
              )}
              {processing.loading && !processing.result && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {[260, 180, 140].map((h, i) => <div key={i} className="shimmer" style={{ height: h, borderRadius: 16 }} />)}
                </div>
              )}
              {processing.result && (
                <ResultsDashboard result={processing.result} onDownload={handleDownload} downloadLoading={dlLoading} />
              )}
              {!processing.result && !processing.loading && !processing.error && (
                <div className="glass" style={{ padding: 64, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', minHeight: 320 }}>
                  <div style={{ width: 64, height: 64, borderRadius: 18, background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(148,163,184,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
                    <Database size={28} color="#1e293b" />
                  </div>
                  <p style={{ color: '#334155', fontWeight: 500, fontFamily: 'DM Sans' }}>Results appear here</p>
                  <p style={{ color: '#1e293b', fontSize: 13, marginTop: 6, fontFamily: 'DM Sans' }}>Upload a file and describe your goal</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* RECOMMEND */}
      {page === 'recommend' && (
        <div style={{ maxWidth: 900, margin: '0 auto', padding: '40px 20px' }}>
          <div style={{ marginBottom: 32 }}>
            <h2 style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 28, color: '#f1f5f9', letterSpacing: '-0.02em', marginBottom: 8 }}>Find a Dataset</h2>
            <p style={{ color: '#475569', fontFamily: 'DM Sans', fontSize: 14 }}>Describe your ML goal. DataForge searches Kaggle, HuggingFace, and curated sources for the best match.</p>
          </div>
          <div className="glass" style={{ padding: 24, marginBottom: 24 }}>
            <RecommendPanel onSearch={recommend.search} loading={recommend.loading} />
          </div>
          {recommend.error && <div className="glass" style={{ padding: 16, color: '#fb7185', fontSize: 13 }}>{recommend.error}</div>}
          {recommend.loading && <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>{[1,2,3].map(i => <div key={i} className="shimmer" style={{ height: 110, borderRadius: 16 }} />)}</div>}
          {recommend.result && <RecommendResults result={recommend.result} />}
        </div>
      )}

      {/* DOCS */}
      {page === 'docs' && <HowItWorks />}

      {/* Footer */}
      <footer style={{ borderTop: '1px solid rgba(148,163,184,0.05)', marginTop: 80, padding: '32px 24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 24, height: 24, borderRadius: 6, background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Database size={12} color="white" />
            </div>
            <span style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 14, color: '#334155' }}>DataForge AI</span>
          </div>
          <p style={{ fontSize: 12, color: '#1e293b', fontFamily: 'DM Sans' }}>FastAPI · scikit-learn · React 18 · Tailwind — 100% free to run</p>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ color: '#334155' }}><Github size={16} /></a>
        </div>
      </footer>
    </div>
  )
}

// ── Mobile responsive fix for process grid ───────────────────────────────────
const mobileStyle = document.createElement('style')
mobileStyle.textContent = `
  @media (max-width: 768px) {
    .process-grid { grid-template-columns: 1fr !important; }
    .sticky-panel { position: static !important; }
  }
`
document.head.appendChild(mobileStyle)

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  )
}
