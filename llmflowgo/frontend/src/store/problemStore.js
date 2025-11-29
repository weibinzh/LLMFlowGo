import { create } from 'zustand';
import { problemsApi } from '../services/api';

const useProblemStore = create((set) => ({
    problems: [],
    isLoading: false,
    error: null,
    
    fetchProblems: async () => {
        try {
            set({ isLoading: true, error: null });
            const response = await problemsApi.getAll();
            set({ problems: response.data, isLoading: false });
        } catch (error) {
            console.error("Failed to fetch problems:", error);
            set({ isLoading: false, error: 'Failed to load problems.' });
        }
    },
    deleteProblem: async (id) => {
        try {
            set({ isLoading: true, error: null });
            await problemsApi.delete(id);
            set((state) => ({
                problems: Array.isArray(state.problems)
                    ? state.problems.filter((p) => p.id !== id)
                    : [],
                isLoading: false,
            }));
        } catch (error) {
            console.error("Failed to delete problem:", error);
            set({ isLoading: false, error: 'Failed to delete problem.' });
            throw error;
        }
    },
}));

export default useProblemStore;
