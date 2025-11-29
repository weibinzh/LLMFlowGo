import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Use the persist middleware to automatically save the state to localStorage
const useLLMConfigStore = create(
    persist(
        (set) => ({
            // Default values
            apiKey: '',
            baseUrl: 'https://api.openai.com/v1',
            modelName: 'gpt-3.5-turbo',
            
            // Action to update the configuration
            setConfig: (config) => set(config),
        }),
        {
            name: 'llm-config-storage', // localStorage 中的 key
        }
    )
);

export default useLLMConfigStore;
