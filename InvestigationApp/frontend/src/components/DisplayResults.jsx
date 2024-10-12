// src/components/DisplayResults.jsx

import React from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const DisplayResults = ({ results }) => {
  // Default values when no results are available
  const confidence = results ? results.confidence : 0;
  const classification = results ? results.classification : 'Neither';

  return (
    <div className="bg-white p-4 h-1/2 w-full flex flex-col justify-center items-center">
      <h2 className="text-2xl font-bold font-martel text-center mb-4">Model Results</h2>
      <div className="flex flex-col items-center">
        {/* Circular progress bar for confidence */}
        <div className="w-24 h-24 mb-2">
          <CircularProgressbar
            value={confidence} // Directly use the confidence value as it is already a percentage
            text={`${confidence.toFixed(2)}%`} // Display the percentage with two decimal points
            styles={buildStyles({
              pathColor: '#1e3a8a', // Blue color for the progress path
              textColor: '#1e3a8a', // Matching text color
              trailColor: '#d6d6d6', // Light gray trail color
              textSize: '16px', // Size of the text inside the circle
            })}
          />
        </div>
        {/* Label indicating what the circle represents */}
        <p className="text-sm text-gray-600 font-mono font-semibold mb-4">Model Confidence</p>
        {/* Classification text */}
        <p className="text-lg font-mono font-bold text-center">
          Classification: {classification}
        </p>
      </div>
    </div>
  );
};

export default DisplayResults;
