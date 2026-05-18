import axios from "axios";

const api = axios.create({
  baseURL: "https://dataforge-ai-gzdt.onrender.com",
  timeout: 600000,
});

export default api;

// ======================================================
// ANALYZE DATASET
// ======================================================

export const analyzeDataset = async (
  file,
  prompt
) => {
  const form = new FormData();

  form.append("file", file);

  form.append("prompt", prompt);

  const { data } = await api.post(
    "/api/analyze",
    form
  );

  return data;
};

// ======================================================
// PROCESS DATASET
// ======================================================

export const processDataset = async (
  file,
  prompt,
  options = {}
) => {
  const form = new FormData();

  form.append("file", file);

  form.append("prompt", prompt);

  form.append(
    "run_anomaly_detection",
    options.anomaly ?? true
  );

  form.append(
    "run_feature_engineering",
    options.features ?? true
  );

  const { data } = await api.post(
    "/api/process",
    form
  );

  return data;
};

// ======================================================
// RECOMMEND DATASETS
// ======================================================

export const recommendDatasets = async (
  prompt
) => {
  const form = new FormData();

  form.append("prompt", prompt);

  const { data } = await api.post(
    "/api/recommend",
    form
  );

  return data;
};

// ======================================================
// DOWNLOAD CLEANED DATA
// ======================================================

export const getDownloadUrl = (
  file,
  prompt,
  format = "csv"
) => {
  return async () => {
    const form = new FormData();

    form.append("file", file);

    form.append("prompt", prompt);

    form.append("format", format);

    const response = await api.post(
      "/api/download/cleaned",
      form,
      {
        responseType: "blob",
      }
    );

    const url = URL.createObjectURL(
      response.data
    );

    const a = document.createElement("a");

    a.href = url;

    const ext =
      format === "excel"
        ? "xlsx"
        : format === "parquet"
        ? "parquet"
        : "csv";

    a.download = `cleaned_dataset.${ext}`;

    a.click();

    URL.revokeObjectURL(url);
  };
};

// ======================================================
// SEARCH DATASETS
// ======================================================

export const searchDatasets = async (
  prompt,
  intent = "general_ml"
) => {
  const form = new FormData();

  form.append("prompt", prompt);

  form.append("intent", intent);

  const { data } = await api.post(
    "/api/search-datasets",
    form
  );

  return data;
};