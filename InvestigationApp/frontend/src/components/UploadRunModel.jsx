// src/components/UploadRunModel.jsx

import React, { useState, useRef } from 'react';
import uploadIcon from '../assets/upload.png';

const UploadRunModel = ({ setUploadedImage, setSaliencyMap, setExudates, setResults, uploadedImage, clearResults }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState('');
  const dropRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous results and reset to initial state
    clearResults();

    setSelectedFile(file);
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:5000/resize-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to resize the image.');
      }

      const result = await response.json();
      setUploadedImage(result.resized_image_url); // Display resized image
      setMessage('Image uploaded successfully.');
    } catch (error) {
      console.error('Error resizing image:', error);
      setMessage('Error resizing image.');
    }
  };

  const handleRunModel = async () => {
    if (!selectedFile) {
      setMessage('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/run-model', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to run the model.');
      }

      const result = await response.json();
      setResults(result.results);
      setSaliencyMap(result.saliency_url);
      setExudates(result.exudates_url);
      setMessage('Model has run successfully.');
    } catch (error) {
      console.error('Error running model:', error);
      setMessage('Error running model.');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    dropRef.current.classList.add('border-blue-500');
  };

  const handleDragLeave = () => {
    dropRef.current.classList.remove('border-blue-500');
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dropRef.current.classList.remove('border-blue-500');
    const file = event.dataTransfer.files[0];
    handleFileChange({ target: { files: [file] } });
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="bg-white p-4 h-1/2 w-full flex flex-col justify-center items-center">
      <h2 className="text-2xl font-bold font-martel text-center mb-4">Upload and Run Model</h2>
      <div
        ref={dropRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className="border-2 border-dashed border-gray-300 p-4 rounded-lg flex flex-col items-center justify-center cursor-pointer"
      >
        <img src={uploadIcon} alt="Upload Icon" className="w-16 mb-2" />
        <p className="text-m text-gray-500 font-mono text-center">Drag & Drop Your Image Here Or Click To Select</p>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileChange}
          className="hidden"
          accept="image/png, image/jpeg"
        />
      </div>
      {selectedFile && !message && (
        <p className="text-sm font-mono text-gray-700 mt-2">
          Selected file: <span className="font-semibold font-mono">{selectedFile.name}</span>
        </p>
      )}
      {message && <p className="text-sm text-gray-700 mt-2 font-mono font-semibold">{message}</p>}
      <button
        onClick={handleRunModel}
        className="w-full bg-blue-900 text-white font-mono py-2 rounded-lg hover:bg-blue-800 mt-4"
        disabled={!uploadedImage} // Disable button until resized image is displayed
      >
        Run Model
      </button>
    </div>
  );
};

export default UploadRunModel;
