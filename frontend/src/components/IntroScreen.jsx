import { useEffect, useState } from 'react'

const STORAGE_KEY = 'dataforge_intro_seen'

export function IntroScreen({ onDone }) {
  const [phase, setPhase] = useState(0)
  // 0 = black, 1 = particles + logo in, 2 = tagline, 3 = fade out

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 300)
    const t2 = setTimeout(() => setPhase(2), 1800)
    const t3 = setTimeout(() => setPhase(3), 3400)
    const t4 = setTimeout(() => { sessionStorage.setItem(STORAGE_KEY, '1'); onDone() }, 4200)
    return () => [t1,t2,t3,t4].forEach(clearTimeout)
  }, [onDone])

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: '#020408',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      transition: 'opacity 0.8s ease',
      opacity: phase === 3 ? 0 : 1,
      overflow: 'hidden',
    }}>
      {/* Animated grid */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'linear-gradient(rgba(99,102,241,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.06) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
        opacity: phase >= 1 ? 1 : 0,
        transition: 'opacity 1s ease',
      }} />

      {/* Glow orb */}
      <div style={{
        position: 'absolute',
        width: 500, height: 500,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)',
        filter: 'blur(40px)',
        transform: `scale(${phase >= 1 ? 1 : 0.3})`,
        transition: 'transform 1.5s cubic-bezier(0.16,1,0.3,1)',
      }} />

      {/* Ring */}
      <div style={{
        position: 'absolute',
        width: 320, height: 320,
        borderRadius: '50%',
        border: '1px solid rgba(99,102,241,0.2)',
        opacity: phase >= 1 ? 1 : 0,
        transform: `scale(${phase >= 1 ? 1 : 0.5})`,
        transition: 'all 1.2s cubic-bezier(0.16,1,0.3,1)',
        boxShadow: '0 0 60px rgba(99,102,241,0.1)',
      }} />
      <div style={{
        position: 'absolute',
        width: 420, height: 420,
        borderRadius: '50%',
        border: '1px solid rgba(99,102,241,0.08)',
        opacity: phase >= 1 ? 1 : 0,
        transform: `scale(${phase >= 1 ? 1 : 0.5})`,
        transition: 'all 1.4s cubic-bezier(0.16,1,0.3,1) 0.1s',
      }} />

      {/* Logo icon */}
      <div style={{
        position: 'relative',
        width: 72, height: 72,
        borderRadius: 20,
        background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 0 40px rgba(99,102,241,0.5), 0 20px 60px rgba(99,102,241,0.3)',
        opacity: phase >= 1 ? 1 : 0,
        transform: `translateY(${phase >= 1 ? 0 : 30}px) scale(${phase >= 1 ? 1 : 0.8})`,
        transition: 'all 1s cubic-bezier(0.16,1,0.3,1) 0.2s',
        marginBottom: 28,
      }}>
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
          <path d="M4 6h16M4 10h16M4 14h10M4 18h6" stroke="white" strokeWidth="2" strokeLinecap="round"/>
          <circle cx="19" cy="16" r="3" fill="white" fillOpacity="0.9"/>
          <path d="M19 14v2l1 1" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </div>

      {/* Name */}
      <div style={{
        position: 'relative',
        opacity: phase >= 1 ? 1 : 0,
        transform: `translateY(${phase >= 1 ? 0 : 20}px)`,
        transition: 'all 1s cubic-bezier(0.16,1,0.3,1) 0.4s',
        textAlign: 'center',
      }}>
        <div style={{
          fontSize: 42,
          fontWeight: 800,
          fontFamily: 'Syne, system-ui, sans-serif',
          letterSpacing: '-0.03em',
          background: 'linear-gradient(135deg, #f1f5f9 30%, #818cf8)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          lineHeight: 1,
          marginBottom: 12,
        }}>
          DataForge <span style={{
            background: 'linear-gradient(135deg, #6366f1, #a78bfa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>AI</span>
        </div>

        {/* Tagline - appears later */}
        <div style={{
          color: '#64748b',
          fontSize: 15,
          fontFamily: 'DM Sans, system-ui',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          opacity: phase >= 2 ? 1 : 0,
          transform: `translateY(${phase >= 2 ? 0 : 10}px)`,
          transition: 'all 0.8s ease',
        }}>
          Intelligent Data Preprocessing
        </div>
      </div>

      {/* Loading bar */}
      <div style={{
        position: 'absolute', bottom: 48,
        width: 120, height: 2,
        background: 'rgba(99,102,241,0.15)',
        borderRadius: 2,
        overflow: 'hidden',
        opacity: phase >= 1 ? 1 : 0,
        transition: 'opacity 0.5s ease 0.8s',
      }}>
        <div style={{
          height: '100%',
          background: 'linear-gradient(90deg, #6366f1, #a78bfa)',
          borderRadius: 2,
          animation: 'introBar 2.8s ease forwards',
        }} />
      </div>

      <style>{`
        @keyframes introBar {
          from { width: 0% }
          to   { width: 100% }
        }
      `}</style>
    </div>
  )
}

export function useIntro() {
  const seen = sessionStorage.getItem(STORAGE_KEY)
  const [show, setShow] = useState(!seen)
  return { show, done: () => setShow(false) }
}
