import React from 'react';

function ControlPanel({
  classNames,
  techniques,
  selectedObject,
  selectedTechnique,
  confidenceThreshold,
  onObjectChange,
  onTechniqueChange,
  onConfidenceChange,
  onProcessClick,
  disabled
}) {
  return (
    <div className="control-panel">
      <div className="control-group">
        <label htmlFor="object-select">Select object to isolate:</label>
        <select 
          id="object-select"
          value={selectedObject}
          onChange={(e) => onObjectChange(e.target.value)}
          disabled={disabled}
        >
          {classNames.map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>
      
      <div className="control-group">
        <label htmlFor="technique-select">Masking Technique:</label>
        <select 
          id="technique-select"
          value={selectedTechnique}
          onChange={(e) => onTechniqueChange(e.target.value)}
          disabled={disabled}
        >
          {techniques.map(tech => (
            <option key={tech} value={tech}>{tech}</option>
          ))}
        </select>
      </div>
      
      <div className="control-group">
        <label htmlFor="confidence-slider">
          Confidence Threshold: {confidenceThreshold.toFixed(2)}
        </label>
        <input 
          id="confidence-slider"
          type="range"
          min="0.1"
          max="1.0"
          step="0.05"
          value={confidenceThreshold}
          onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="slider"
        />
      </div>
      
      <button 
        className="process-button"
        onClick={onProcessClick}
        disabled={disabled}
      >
        {disabled ? 'Processing...' : 'Process Video'}
      </button>
    </div>
  );
}

export default ControlPanel;