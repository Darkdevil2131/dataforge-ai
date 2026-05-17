import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Upload, FileSpreadsheet, X, Sparkles,
  ArrowRight, Loader2, Database
} from 'lucide-react'
import { clsx } from 'clsx'

const ACCEPTED = {
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.ms-excel': ['.xls'],
  'application/json': ['.json'],
  'application/octet-stream': ['.parquet'],
}

const QUICK_PROMPTS = [
  "Prepare this for fraud detection",
  "Clean for a classification model",
  "Optimize for time-series forecasting",
  "Prepare for a sales analytics dashboard",
  "Build a customer churn model",
  "I want to learn ML with this data",
]

export function UploadPanel({ onSubmit, loading }) {
  const [file, setFile] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [options, setOptions] = useState({ anomaly: true, features: true })

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) setFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    multiple: false,
  })

  const handleSubmit = () => {
    if (!file || !prompt.trim()) return
    onSubmit(file, prompt, options)
  }

  const canSubmit = file && prompt.trim().length > 3

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={clsx(
          'border-2 border-dashed rounded-xl p-10 cursor-pointer transition-all duration-200',
          'flex flex-col items-center justify-center text-center',
          isDragActive
            ? 'border-indigo-500 bg-indigo-500/10 glow-indigo'
            : file
            ? 'border-emerald-500/40 bg-emerald-500/5'
            : 'border-slate-700 hover:border-slate-500 bg-slate-900/50 hover:bg-slate-800/50'
        )}
      >
        <input {...getInputProps()} />
        {file ? (
          <>
            <FileSpreadsheet className="w-10 h-10 text-emerald-400 mb-3" />
            <p className="text-emerald-400 font-semibold text-lg">{file.name}</p>
            <p className="text-slate-500 text-sm mt-1">
              {(file.size / 1024).toFixed(1)} KB — click to replace
            </p>
            <button
              className="mt-3 text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
              onClick={(e) => { e.stopPropagation(); setFile(null) }}
            >
              <X className="w-3 h-3" /> Remove
            </button>
          </>
        ) : (
          <>
            <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center mb-4">
              <Upload className="w-6 h-6 text-slate-400" />
            </div>
            <p className="text-slate-300 font-medium text-lg">
              {isDragActive ? 'Drop your dataset here' : 'Upload your dataset'}
            </p>
            <p className="text-slate-500 text-sm mt-1">
              CSV, Excel, JSON, Parquet — up to 100K rows
            </p>
          </>
        )}
      </div>

      {/* Prompt input */}
      <div className="space-y-2">
        <label className="section-label">What do you want to do with this data?</label>
        <div className="relative">
          <Sparkles className="absolute left-3 top-3.5 w-4 h-4 text-indigo-400 pointer-events-none" />
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="e.g. Prepare this for a fraud detection model…"
            rows={2}
            className={clsx(
              'w-full bg-slate-900 border rounded-xl pl-10 pr-4 py-3 resize-none',
              'text-slate-200 placeholder-slate-600 text-sm',
              'focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all',
              'border-slate-700 focus:border-indigo-500/60'
            )}
          />
        </div>

        {/* Quick prompts */}
        <div className="flex flex-wrap gap-2 pt-1">
          {QUICK_PROMPTS.map(q => (
            <button
              key={q}
              onClick={() => setPrompt(q)}
              className={clsx(
                'text-xs px-3 py-1.5 rounded-lg border transition-all duration-150',
                prompt === q
                  ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500'
              )}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Options */}
      <div className="flex gap-4">
        {[
          { key: 'anomaly', label: 'Anomaly Detection' },
          { key: 'features', label: 'Feature Engineering' },
        ].map(({ key, label }) => (
          <label key={key} className="flex items-center gap-2 cursor-pointer group">
            <div
              onClick={() => setOptions(o => ({ ...o, [key]: !o[key] }))}
              className={clsx(
                'w-9 h-5 rounded-full transition-all duration-200 relative',
                options[key] ? 'bg-indigo-600' : 'bg-slate-700'
              )}
            >
              <div className={clsx(
                'absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200',
                options[key] ? 'left-4' : 'left-0.5'
              )} />
            </div>
            <span className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
              {label}
            </span>
          </label>
        ))}
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!canSubmit || loading}
        className={clsx(
          'btn-primary w-full py-3 text-base justify-center',
          !canSubmit && 'opacity-40 cursor-not-allowed'
        )}
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Processing your dataset…
          </>
        ) : (
          <>
            <Database className="w-5 h-5" />
            Run DataForge AI
            <ArrowRight className="w-4 h-4 ml-auto" />
          </>
        )}
      </button>
    </div>
  )
}

export function RecommendPanel({ onSearch, loading }) {
  const [prompt, setPrompt] = useState('')

  return (
    <div className="space-y-4">
      <label className="section-label">Describe your ML goal</label>
      <div className="flex gap-2">
        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && prompt.trim() && onSearch(prompt)}
          placeholder="e.g. I want to build a heart disease prediction model…"
          className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/60"
        />
        <button
          onClick={() => prompt.trim() && onSearch(prompt)}
          disabled={!prompt.trim() || loading}
          className="btn-primary px-5"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
        </button>
      </div>
    </div>
  )
}
