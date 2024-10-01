// src/components/DisplayUploadedImage.jsx

import React from 'react';

const DisplayUploadedImage = ({ uploadedImage }) => {
  return (
    <div className="bg-black p-2 h-1/2 w-full flex flex-col justify-center items-center box-border text-white font-martel font-bold relative rounded-t-lg">
      <h2 className="text-2xl text-center mb-2">Uploaded Image</h2>
      {uploadedImage && (
        <img
          src={uploadedImage}
          alt="Uploaded"
          className="max-h-full max-w-full object-contain"
          style={{ flex: '1 0 auto' }}
        />
      )}
      {/* Incomplete white line at the bottom, slightly incomplete on both ends */}
      <div className="absolute bottom-0 left-[10%] right-[10%] h-[2px] bg-white"></div>
    </div>
  );
};

export default DisplayUploadedImage;
