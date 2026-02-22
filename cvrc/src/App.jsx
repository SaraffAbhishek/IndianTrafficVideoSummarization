import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoUploader from './components/VideoUploader';
import ControlPanel from './components/ControlPanel';
import ProcessingStatus from './components/ProcessingStatus';
import VideoPlayer from './components/VideoPlayer';
import Footer from './components/Footer';
import axios from 'axios';
import './App.css';

// API base URL - adjust this if your Flask backend runs on a different port
const API_BASE_URL =  'http://127.0.0.1:5000/api';

function App() {
  // State management
  const [classNames, setClassNames] = useState([]);
  const [techniques, setTechniques] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [selectedObject, setSelectedObject] = useState('car');
  const [selectedTechnique, setSelectedTechnique] = useState('Black & White');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [outputVideo, setOutputVideo] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [processingStats, setProcessingStats] = useState(null);

  // Fetch available classes and techniques from the API on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch available object classes from backend
        const classesResponse = await axios.get(`${API_BASE_URL}/classes`);
        if (classesResponse.data && classesResponse.data.classes) {
          setClassNames(classesResponse.data.classes);
          // Set default selected object if available
          if (classesResponse.data.classes.length > 0) {
            setSelectedObject(classesResponse.data.classes[0]);
          }
        }

        // Fetch available masking techniques from backend
        const techniquesResponse = await axios.get(`${API_BASE_URL}/techniques`);
        if (techniquesResponse.data && techniquesResponse.data.techniques) {
          setTechniques(techniquesResponse.data.techniques);
          // Set default selected technique if available
          if (techniquesResponse.data.techniques.length > 0) {
            setSelectedTechnique(techniquesResponse.data.techniques[0]);
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        setErrorMessage('Failed to fetch application data from server');
      }
    };

    fetchData();
  }, []);

  // Handle file upload
  const handleFileUpload = async (file) => {
    if (!file) return;

    setSelectedFile(file);
    setOutputVideo(null);
    setErrorMessage('');
    setProcessingStats(null);
    setFileId(null);

    // Upload file to backend
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      if (response.data && response.data.file_id) {
        setFileId(response.data.file_id);
        console.log('File uploaded successfully, ID:', response.data.file_id);
      } else {
        setErrorMessage('File upload succeeded but received invalid response');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setErrorMessage('Failed to upload video file. Please try again.');
    }
  };

  // Process video function
  const processVideo = async () => {
    if (!fileId) {
      setErrorMessage('Please upload a video first.');
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setErrorMessage('');
    setOutputVideo(null);
    setProcessingStats(null);

    try {
      // Start simulating progress for processing (since we don't have real-time updates)
      let progressCounter = 0;
      const progressInterval = setInterval(() => {
        progressCounter += 1;
        setProgress(Math.min(progressCounter, 99)); // Don't reach 100% until we're done
        
        if (progressCounter >= 99) {
          clearInterval(progressInterval);
        }
      }, 300);

      // Send processing request to backend
      const response = await axios.post(`${API_BASE_URL}/process`, {
        file_id: fileId,
        selected_object: selectedObject,
        selected_technique: selectedTechnique,
        confidence_threshold: confidenceThreshold
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (response.data && response.data.output_video) {
        setOutputVideo(`${API_BASE_URL}/video/${response.data.output_video}`);
        setProcessingStats(response.data.stats);
        console.log('Processing stats:', response.data.stats);
      } else {
        setErrorMessage('Processing succeeded but received invalid response');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setErrorMessage('Failed to process video. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Poll for job status (if we implement async processing)
  const checkJobStatus = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status/${jobId}`);
      if (response.data && response.data.status === 'complete') {
        // Job is complete, fetch results
        setProgress(100);
        return true;
      } else {
        // Job still in progress
        setProgress(response.data.progress || 0);
        return false;
      }
    } catch (error) {
      console.error('Error checking job status:', error);
      return false;
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <div className="controls-section">
          <VideoUploader onFileUpload={handleFileUpload} />
          
          <ControlPanel 
            classNames={classNames.length > 0 ? classNames : ['Loading...']}
            techniques={techniques.length > 0 ? techniques : ['Loading...']}
            selectedObject={selectedObject}
            selectedTechnique={selectedTechnique}
            confidenceThreshold={confidenceThreshold}
            onObjectChange={setSelectedObject}
            onTechniqueChange={setSelectedTechnique}
            onConfidenceChange={setConfidenceThreshold}
            onProcessClick={processVideo}
            disabled={isProcessing || !fileId}
          />
        </div>
        
        {isProcessing && (
          <ProcessingStatus progress={progress} />
        )}
        
        {errorMessage && (
          <div className="error-message">{errorMessage}</div>
        )}
        
        {outputVideo && (
          <div className="results-container">
            <VideoPlayer videoUrl={outputVideo} />
            
            {processingStats && (
              <div className="processing-stats">
                <h3>Processing Statistics</h3>
                <ul>
                  <li>Total Frames: {processingStats.total_frames}</li>
                  <li>Frames with Detections: {processingStats.frames_with_detections}</li>
                  <li>Total Objects Detected: {processingStats.total_objects_detected}</li>
                  <li>Processing Time: {processingStats.processing_time.toFixed(2)} seconds</li>
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
      
      <Footer />
    </div>
  );
}

export default App;