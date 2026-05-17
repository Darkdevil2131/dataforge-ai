// src/firebase.js
// ─────────────────────────────────────────────────────────────────────────────
// SETUP INSTRUCTIONS (5 minutes, completely free):
//
// 1. Go to https://console.firebase.google.com
// 2. Click "Add project" → name it "dataforge-ai" → Continue
// 3. Click the </> (Web) icon → register app → copy the firebaseConfig object
// 4. Replace the values below with YOUR values
// 5. In Firebase console → Authentication → Sign-in method → Enable:
//      - Google
//      - Email/Password
//      - Facebook (requires Facebook Developer App — see README)
// ─────────────────────────────────────────────────────────────────────────────

import { initializeApp } from 'firebase/app'
import { getAuth, GoogleAuthProvider, FacebookAuthProvider } from 'firebase/auth'

const firebaseConfig = {
  apiKey:            import.meta.env.VITE_FIREBASE_API_KEY            || "YOUR_API_KEY",
  authDomain:        import.meta.env.VITE_FIREBASE_AUTH_DOMAIN        || "YOUR_AUTH_DOMAIN",
  projectId:         import.meta.env.VITE_FIREBASE_PROJECT_ID         || "YOUR_PROJECT_ID",
  storageBucket:     import.meta.env.VITE_FIREBASE_STORAGE_BUCKET     || "YOUR_STORAGE_BUCKET",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID|| "YOUR_SENDER_ID",
  appId:             import.meta.env.VITE_FIREBASE_APP_ID             || "YOUR_APP_ID",
}

const app      = initializeApp(firebaseConfig)
export const auth             = getAuth(app)
export const googleProvider   = new GoogleAuthProvider()
export const facebookProvider = new FacebookAuthProvider()

googleProvider.setCustomParameters({ prompt: 'select_account' })
