import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000,
})

export const analyzeDataset = async (file, prompt) => {
  const form = new FormData()
  form.append('file', file)
  form.append('prompt', prompt)
  const { data } = await api.post('/api/analyze', form)
  return data
}

export const processDataset = async (file, prompt, options = {}) => {
  const form = new FormData()
  form.append('file', file)
  form.append('prompt', prompt)
  form.append('run_anomaly_detection', options.anomaly ?? true)
  form.append('run_feature_engineering', options.features ?? true)
  const { data } = await api.post('/api/process', form)
  return data
}

export const recommendDatasets = async (prompt) => {
  const form = new FormData()
  form.append('prompt', prompt)
  const { data } = await api.post('/api/recommend', form)
  return data
}

export const getDownloadUrl = (file, prompt, format = 'csv') => {
  // Returns a function that triggers download
  return async () => {
    const form = new FormData()
    form.append('file', file)
    form.append('prompt', prompt)
    form.append('format', format)
    const response = await api.post('/api/download/cleaned', form, {
      responseType: 'blob',
    })
    const url = URL.createObjectURL(response.data)
    const a = document.createElement('a')
    a.href = url
    const ext = format === 'excel' ? 'xlsx' : format === 'parquet' ? 'parquet' : 'csv'
    a.download = `cleaned_dataset.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }
}

export const searchDatasets = async (prompt, intent = 'general_ml') => {
  const form = new FormData()
  form.append('prompt', prompt)
  form.append('intent', intent)
  const { data } = await api.post('/api/search-datasets', form)
  return data
}
