// src/components/DisplayExudates.jsx

import React from 'react';

const DisplayExudates = ({ exudates }) => {
  return (
    <div className="bg-black p-2 h-1/2 w-full flex flex-col justify-center items-center box-border text-white font-martel font-bold relative rounded-t-lg">
      <h2 className="text-2xl text-center mb-2">Exudates</h2>
      {exudates === "N/A" ? (
        <p className="text-lg">N/A</p>
      ) : (
        exudates && (
          <img
            src={exudates}
            alt="Exudates"
            className="max-h-full max-w-full object-contain"
            style={{ flex: '1 0 auto' }}
          />
        )
      )}
      <div className="absolute bottom-0 left-[10%] right-[10%] h-[2px] bg-white"></div>
    </div>
  );
};

export default DisplayExudates;
