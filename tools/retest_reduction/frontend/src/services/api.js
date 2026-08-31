import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export const getHealth = () => api.get('/health').then(r => r.data);
export const getModelInfo = () => api.get('/model/info').then(r => r.data);
export const getSingleEventOptions = () => api.get('/datasets/single-event-options').then(r => r.data);

export const predictSingleWithShap = (event) => api.post('/predict/single-with-shap', event).then(r => r.data);

export const getMonth12Batch = (costPerHour) => 
  api.get('/analysis/month12-batch', { params: { cost_per_hour: costPerHour } }).then(r => r.data);

export const uploadPreRetest = (file, costPerHour) => {
  const fd = new FormData();
  fd.append('file', file);
  if (costPerHour) fd.append('cost_per_hour', costPerHour);
  return api.post('/analysis/upload-pre-retest', fd, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data);
};

export const validateOutcomes = (file, useLocal, predictionsJson) => {
  const fd = new FormData();
  if (file) fd.append('file', file);
  fd.append('use_local_file', useLocal ? 'True' : 'False');
  fd.append('predictions', predictionsJson);
  return api.post('/analysis/validate-outcomes', fd, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data);
};

export const getHistoricalValidation = () => api.get('/analysis/historical-validation').then(r => r.data);
export const getReferenceAudit = () => api.get('/analysis/reference-audit').then(r => r.data);

export const getOnlineLearningStatus = () => api.get('/online-learning/status').then(r => r.data);
export const learnFromOutcomes = (validatedEvents) => api.post('/online-learning/learn', { validated_events: validatedEvents }).then(r => r.data);
export const resetOnlineLearning = () => api.post('/online-learning/reset').then(r => r.data);

export const getCostImpact = (events, costPerHour) => 
  api.post('/cost-impact', { events, cost_per_hour: costPerHour }).then(r => r.data);
