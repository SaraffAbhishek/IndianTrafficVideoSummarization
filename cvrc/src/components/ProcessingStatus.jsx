import React from 'react';

function ProcessingStatus({ progress }) {
  return (
    <div className="processing-status">
      <div className="progress-container">
        <div 
          className="progress-bar" 
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      <p className="progress-text">{Math.round(progress)}% Complete</p>
    </div>
  );
}

export default ProcessingStatus;