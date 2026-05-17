import { useState } from 'react'
import { useAuth } from '../context/AuthContext.jsx'
import { Eye, EyeOff, Mail, Lock, ArrowRight, Loader2 } from 'lucide-react'

function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  )
}

function FacebookIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="#1877F2">
      <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
    </svg>
  )
}

export function AuthPage() {
  const { loginGoogle, loginFacebook, loginEmail, signupEmail, error, clearError } = useAuth()
  const [mode, setMode]         = useState('login')   // 'login' | 'signup' | 'reset'
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw]     = useState(false)
  const [loading, setLoading]   = useState(false)
  const [resetSent, setResetSent] = useState(false)
  const { resetPassword } = useAuth()

  const handleEmail = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      if (mode === 'login')  await loginEmail(email, password)
      else                   await signupEmail(email, password)
    } catch {}
    setLoading(false)
  }

  const handleReset = async (e) => {
    e.preventDefault()
    const ok = await resetPassword(email)
    if (ok) setResetSent(true)
  }

  const switchMode = (m) => { clearError(); setMode(m); setResetSent(false) }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#020408',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px 16px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Background grid */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
      }} />
      {/* Glow */}
      <div style={{
        position: 'absolute', top: '-20%', left: '50%', transform: 'translateX(-50%)',
        width: 600, height: 400,
        background: 'radial-gradient(ellipse, rgba(99,102,241,0.12) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }} />

      <div style={{ position: 'relative', width: '100%', maxWidth: 400 }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{
            width: 52, height: 52, borderRadius: 14,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 30px rgba(99,102,241,0.4)',
            marginBottom: 16,
          }}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
              <path d="M4 6h16M4 10h16M4 14h10M4 18h6" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="19" cy="16" r="3" fill="white" fillOpacity="0.9"/>
            </svg>
          </div>
          <div style={{ fontFamily: 'Syne, system-ui', fontWeight: 800, fontSize: 24, color: '#f1f5f9', letterSpacing: '-0.02em' }}>
            DataForge AI
          </div>
          <div style={{ color: '#475569', fontSize: 14, marginTop: 6, fontFamily: 'DM Sans, system-ui' }}>
            {mode === 'login'  ? 'Sign in to your workspace' :
             mode === 'signup' ? 'Create your free account' :
             'Reset your password'}
          </div>
        </div>

        {/* Card */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.8)',
          border: '1px solid rgba(148,163,184,0.08)',
          borderRadius: 20,
          padding: '32px 28px',
          backdropFilter: 'blur(20px)',
          boxShadow: '0 25px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.05)',
        }}>
          {/* Error */}
          {error && (
            <div style={{
              background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)',
              borderRadius: 10, padding: '10px 14px', marginBottom: 20,
              color: '#fb7185', fontSize: 13, fontFamily: 'DM Sans, system-ui',
            }}>
              {error}
            </div>
          )}

          {resetSent ? (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <div style={{ fontSize: 40, marginBottom: 16 }}>📧</div>
              <div style={{ color: '#a5f3fc', fontWeight: 600, marginBottom: 8 }}>Check your email</div>
              <div style={{ color: '#64748b', fontSize: 14 }}>Password reset link sent to {email}</div>
              <button onClick={() => switchMode('login')} style={linkStyle}>Back to sign in</button>
            </div>
          ) : mode !== 'reset' ? (
            <>
              {/* Social buttons */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 24 }}>
                <button onClick={loginGoogle} style={socialBtnStyle}>
                  <GoogleIcon /> Continue with Google
                </button>
                <button onClick={loginFacebook} style={{ ...socialBtnStyle, background: 'rgba(24,119,242,0.08)', borderColor: 'rgba(24,119,242,0.2)' }}>
                  <FacebookIcon /> Continue with Facebook
                </button>
              </div>

              {/* Divider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
                <div style={{ flex: 1, height: 1, background: 'rgba(148,163,184,0.08)' }} />
                <span style={{ color: '#334155', fontSize: 12, fontFamily: 'DM Sans' }}>or with email</span>
                <div style={{ flex: 1, height: 1, background: 'rgba(148,163,184,0.08)' }} />
              </div>

              {/* Email form */}
              <form onSubmit={handleEmail} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ position: 'relative' }}>
                  <Mail style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: '#475569', width: 16, height: 16 }} />
                  <input
                    type="email" placeholder="Email address" value={email}
                    onChange={e => setEmail(e.target.value)} required
                    style={inputStyle}
                  />
                </div>
                <div style={{ position: 'relative' }}>
                  <Lock style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: '#475569', width: 16, height: 16 }} />
                  <input
                    type={showPw ? 'text' : 'password'} placeholder="Password" value={password}
                    onChange={e => setPassword(e.target.value)} required minLength={6}
                    style={{ ...inputStyle, paddingRight: 44 }}
                  />
                  <button type="button" onClick={() => setShowPw(s => !s)}
                    style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#475569', display: 'flex' }}>
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>

                <button type="submit" disabled={loading} style={primaryBtnStyle}>
                  {loading ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> :
                   <ArrowRight size={16} />}
                  {mode === 'login' ? 'Sign In' : 'Create Account'}
                </button>
              </form>

              {/* Footer links */}
              <div style={{ marginTop: 20, textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 8 }}>
                {mode === 'login' && (
                  <button onClick={() => switchMode('reset')} style={linkStyle}>Forgot password?</button>
                )}
                <div style={{ color: '#334155', fontSize: 13, fontFamily: 'DM Sans' }}>
                  {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
                  <button onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')} style={linkStyle}>
                    {mode === 'login' ? 'Sign up free' : 'Sign in'}
                  </button>
                </div>
              </div>
            </>
          ) : (
            /* Reset password form */
            <form onSubmit={handleReset} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div style={{ position: 'relative' }}>
                <Mail style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: '#475569', width: 16, height: 16 }} />
                <input type="email" placeholder="Your email" value={email}
                  onChange={e => setEmail(e.target.value)} required style={inputStyle} />
              </div>
              <button type="submit" style={primaryBtnStyle}>Send Reset Link</button>
              <button type="button" onClick={() => switchMode('login')} style={linkStyle}>Back to sign in</button>
            </form>
          )}
        </div>

        <div style={{ textAlign: 'center', marginTop: 24, color: '#1e293b', fontSize: 12, fontFamily: 'DM Sans' }}>
          By continuing, you agree to our Terms of Service & Privacy Policy
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

const inputStyle = {
  width: '100%', boxSizing: 'border-box',
  background: 'rgba(15,23,42,0.8)',
  border: '1px solid rgba(148,163,184,0.1)',
  borderRadius: 10, padding: '12px 14px 12px 42px',
  color: '#e2e8f0', fontSize: 14, fontFamily: 'DM Sans, system-ui',
  outline: 'none', transition: 'border-color 0.2s',
}

const socialBtnStyle = {
  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
  gap: 10, padding: '11px 16px', borderRadius: 10,
  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
  color: '#cbd5e1', fontSize: 14, fontFamily: 'DM Sans, system-ui',
  cursor: 'pointer', transition: 'all 0.15s',
}

const primaryBtnStyle = {
  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
  gap: 8, padding: '12px 20px', borderRadius: 10,
  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
  border: 'none', color: 'white', fontSize: 14, fontWeight: 600,
  fontFamily: 'DM Sans, system-ui', cursor: 'pointer',
  boxShadow: '0 4px 20px rgba(99,102,241,0.35)',
  transition: 'all 0.2s',
}

const linkStyle = {
  background: 'none', border: 'none', color: '#818cf8',
  fontSize: 13, fontFamily: 'DM Sans, system-ui',
  cursor: 'pointer', textDecoration: 'underline',
}
