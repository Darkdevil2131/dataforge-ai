import { useState, useCallback } from 'react'
import { processDataset, recommendDatasets, searchDatasets, getDownloadUrl } from '../utils/api.js'

export function useProcessing() {
  const [state, setState] = useState({
    loading: false, result: null, error: null, file: null, prompt: '',
  })

  const run = useCallback(async (file, prompt, options = {}) => {
    setState(s => ({ ...s, loading: true, error: null, result: null, file, prompt }))
    try {
      const result = await processDataset(file, prompt, options)
      setState(s => ({ ...s, loading: false, result }))
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Processing failed.'
      setState(s => ({ ...s, loading: false, error: msg }))
    }
  }, [])

  const download = useCallback(async (format = 'csv') => {
    if (!state.file) return
    try { await getDownloadUrl(state.file, state.prompt, format)() }
    catch (err) { console.error('Download failed:', err) }
  }, [state.file, state.prompt])

  const reset = useCallback(() => {
    setState({ loading: false, result: null, error: null, file: null, prompt: '' })
  }, [])

  return { ...state, run, download, reset }
}

export function useRecommend() {
  const [state, setState] = useState({
    loading: false, result: null, error: null,
  })

  const search = useCallback(async (prompt) => {
    setState(s => ({ ...s, loading: true, error: null, result: null }))
    try {
      // Run both: intent-based recommendations AND real dataset search in parallel
      const [intentResult, searchResult] = await Promise.allSettled([
        recommendDatasets(prompt),
        searchDatasets(prompt, 'general_ml'),
      ])
      const base = intentResult.status === 'fulfilled' ? intentResult.value : {}
      const extra = searchResult.status === 'fulfilled' ? searchResult.value.results : []

      // Merge: deduplicate by name, real search results first
      const seen = new Set()
      const merged = [...extra, ...(base.recommendations || [])]
        .filter(d => { const k = d.name?.toLowerCase().slice(0,25); if (seen.has(k)) return false; seen.add(k); return true })
        .slice(0, 8)

      setState(s => ({ ...s, loading: false, result: { ...base, recommendations: merged } }))
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Search failed.'
      setState(s => ({ ...s, loading: false, error: msg }))
    }
  }, [])

  return { ...state, search }
}
