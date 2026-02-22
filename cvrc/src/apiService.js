import axios from 'axios';

// Set base API URL - change this according to your backend server
const API_BASE_URL = ' http://127.0.0.1:5000/api';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service methods
const apiService = {
  // Get available object classes
  getClasses: async () => {
    try {
      const response = await apiClient.get('/classes');
      return response.data.classes;
    } catch (error) {
      console.error('Error fetching classes:', error);
      throw error;
    }
  },

  // Get available masking techniques
  getTechniques: async () => {
    try {
      const response = await apiClient.get('/techniques');
      return response.data.techniques;
    } catch (error) {
      console.error('Error fetching techniques:', error);
      throw error;
    }
  },

  // Upload video file
  uploadVideo: async (file, onUploadProgress) => {
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Upload with progress tracking
      const response = await apiClient.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress,
      });

      return response.data;
    } catch (error) {
      console.error('Error uploading video:', error);
      throw error;
    }
  },

  // Process video with selected parameters
  processVideo: async (fileId, selectedObject, selectedTechnique, confidenceThreshold) => {
    try {
      const response = await apiClient.post('/process', {
        file_id: fileId,
        selected_object: selectedObject,
        selected_technique: selectedTechnique,
        confidence_threshold: confidenceThreshold,
      });

      return response.data;
    } catch (error) {
      console.error('Error processing video:', error);
      throw error;
    }
  },

  // Check processing job status
  checkJobStatus: async (jobId) => {
    try {
      const response = await apiClient.get(`/status/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Error checking job status:', error);
      throw error;
    }
  },

  // Get video URL for processed video
  getVideoUrl: (videoFilename) => {
    return `${API_BASE_URL}/video/${videoFilename}`;
  },

  // Get detections for a processed video
  getDetections: async (detectionsFilename) => {
    try {
      const response = await apiClient.get(`/detections/${detectionsFilename}`);
      return response.data.detections;
    } catch (error) {
      console.error('Error fetching detections:', error);
      throw error;
    }
  },
};

export default apiService;