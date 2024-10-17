// src/components/DisplaySaliencyMap.jsx

import React from 'react';

const DisplaySaliencyMap = ({ saliencyMap }) => {
  return (
    <div className="bg-black p-2 h-1/2 w-full flex flex-col justify-center items-center box-border text-white font-martel font-bold relative rounded-b-lg">
      <h2 className="text-2xl text-center mb-2">Saliency Map</h2>
      {saliencyMap === "N/A" ? (
        <p className="text-lg">N/A</p>
      ) : (
        saliencyMap && (
          <img
            src={saliencyMap}
            alt="Saliency Map"
            className="max-h-full max-w-full object-contain"
            style={{ flex: '1 0 auto' }}
          />
        )
      )}
    </div>
  );
};

export default DisplaySaliencyMap;
