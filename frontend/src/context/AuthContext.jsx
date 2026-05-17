import { createContext, useContext, useEffect, useState } from 'react'
import {
  onAuthStateChanged, signInWithPopup, signInWithEmailAndPassword,
  createUserWithEmailAndPassword, signOut, sendPasswordResetEmail
} from 'firebase/auth'
import { auth, googleProvider, facebookProvider } from '../firebase.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, u => { setUser(u); setLoading(false) })
    return unsub
  }, [])

  const clearError = () => setError('')

  const loginGoogle = async () => {
    try { clearError(); await signInWithPopup(auth, googleProvider) }
    catch (e) { setError(e.code === 'auth/popup-closed-by-user' ? '' : 'Google sign-in failed. Try again.') }
  }

  const loginFacebook = async () => {
    try { clearError(); await signInWithPopup(auth, facebookProvider) }
    catch (e) { setError('Facebook sign-in failed. Make sure Facebook auth is enabled in Firebase.') }
  }

  const loginEmail = async (email, password) => {
    try { clearError(); await signInWithEmailAndPassword(auth, email, password) }
    catch (e) {
      const msgs = { 'auth/user-not-found': 'No account with that email.', 'auth/wrong-password': 'Wrong password.', 'auth/invalid-email': 'Invalid email address.' }
      setError(msgs[e.code] || 'Sign-in failed. Check your credentials.')
      throw e
    }
  }

  const signupEmail = async (email, password) => {
    try { clearError(); await createUserWithEmailAndPassword(auth, email, password) }
    catch (e) {
      const msgs = { 'auth/email-already-in-use': 'Account already exists.', 'auth/weak-password': 'Password must be 6+ characters.' }
      setError(msgs[e.code] || 'Sign-up failed.')
      throw e
    }
  }

  const logout = () => signOut(auth)

  const resetPassword = async (email) => {
    try { await sendPasswordResetEmail(auth, email); return true }
    catch { setError('Could not send reset email.'); return false }
  }

  return (
    <AuthContext.Provider value={{ user, loading, error, loginGoogle, loginFacebook, loginEmail, signupEmail, logout, resetPassword, clearError }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
