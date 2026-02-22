import React, { useRef, useState } from 'react';

function VideoUploader({ onFileUpload }) {
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      onFileUpload(file);
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      // Check if file is a video
      if (file.type.startsWith('video/')) {
        setFileName(file.name);
        onFileUpload(file);
      }
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  return (
    <div className="uploader-container">
      <div 
        className="drop-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current.click()}
      >
        <div className="drop-icon">
          <i className="fas fa-cloud-upload-alt"></i>
        </div>
        <p>Drag & drop a video file here or click to browse</p>
        {fileName && <p className="file-name">Selected: {fileName}</p>}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/avi,video/mov,video/mkv"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default VideoUploader;