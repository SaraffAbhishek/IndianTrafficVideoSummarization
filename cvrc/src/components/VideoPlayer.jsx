import React from 'react';

function VideoPlayer({ videoUrl }) {
  return (
    <div className="video-player">
      <h3>Processed Video:</h3>
      <video 
        controls
        src={videoUrl}
        className="video-display"
      />
      <div className="video-controls">
        <a 
          href={videoUrl} 
          download="processed-video.mp4"
          className="download-button"
        >
          Download Video
        </a>
      </div>
    </div>
  );
}

export default VideoPlayer;