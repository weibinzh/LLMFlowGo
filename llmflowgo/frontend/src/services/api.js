import axios from 'axios';

const apiClient = axios.create({
    baseURL: '/api', // The base URL for all API requests
    headers: {
        'Content-Type': 'application/json',
    },
});

// Example API service for problems
export const problemsApi = {
    getAll: () => apiClient.get('/problems/'),
    getById: (id) => apiClient.get(`/problems/${id}`),
    analyze: (id, llmConfig) => apiClient.post(`/problems/${id}/analyze`, { llm_config: llmConfig }),
    configure: (id, configData) => apiClient.put(`/problems/${id}/configure`, configData),
    create: (formData) => apiClient.post('/problems/', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    }),
    recommendServerCounts: (dagConfig, serverTypes, llmConfig, bounds) =>
        apiClient.post('/dag/recommend-server-counts', {
            dagConfig,
            serverTypes,
            llm_config: llmConfig,
            bounds,
        }),
    getAlgorithmPresets: () => apiClient.get('/problems/algorithm-presets'),
    recommendAlgorithmPreset: (dagConfig, environmentConfig, llmConfig) =>
        apiClient.post('/problems/recommend-algorithm-preset', {
            dagConfig,
            environmentConfig,
            llm_config: llmConfig,
        }),
    suggestOptimizationDescription: (dagConfig, environmentConfig, llmConfig) =>
        apiClient.post('/problems/suggest-optimization-description', {
            dagConfig,
            environmentConfig,
            llm_config: llmConfig,
        }),
    analyzePresetFramework: (preset, dagConfig, environmentConfig, serverTypes, llmConfig) =>
        apiClient.post('/problems/analyze-preset-framework', {
            preset,
            llm_config: llmConfig,
            dagConfig,
            environmentConfig,
            serverTypes,
        }),
    createProblemPackage: async (data) => {
      try {
        const response = await apiClient.post('/problems/create-problem-package', data);
        return response.data;
      } catch (error) {
        console.error('Error creating edge workflow:', error);
        throw error;
      }
    },
    preciseBuildRun: (environmentConfig, dagConfig, llmConfig, name = null, description = null, meohConfig = {}, bounds = null, algorithmPreset = null) =>
        apiClient.post('/problems/precise-build-run', {
            environmentConfig,
            dagConfig,
            llm_config: llmConfig,
            name,
            description,
            meoh_config: meohConfig,
            bounds,
            algorithmPreset,
        }),
    delete: (id) => apiClient.delete(`/problems/${id}`),
};

// API service for runs
export const runsApi = {
    start: (runData) => apiClient.post('/runs/', runData),
    getAll: () => apiClient.get('/runs/'),
    getById: (id) => apiClient.get(`/runs/${id}`),
    getStatus: (id) => apiClient.get(`/runs/${id}/status`),
    getFinalCounts: (id) => apiClient.get(`/runs/${id}/final-counts`),
    getResults: (id) => apiClient.get(`/runs/${id}/results`),
};
