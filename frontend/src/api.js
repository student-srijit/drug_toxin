import axios from 'axios'

const api = axios.create({ baseURL: '' })

export const predict = (smiles) =>
  api.post('/predict', { smiles }).then((r) => r.data)

export const getShap = (smiles, task) =>
  api.post('/shap', { smiles, task }).then((r) => r.data)
