// src/components/UploadRunModel.jsx

import React, { useState, useRef } from 'react';
import uploadIcon from '../assets/upload.png'; // Import the upload icon

const UploadRunModel = ({ setUploadedImage, setSaliencyMap, setResults }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState('');
  const dropRef = useRef(null);
  const fileInputRef = useRef(null); // Reference for the hidden input element

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setMessage(''); // Clear any previous messages when a new file is selected
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
    setSelectedFile(file);
    setMessage(''); // Clear any previous messages when a new file is dropped
  };

  const handleRunModel = async () => {
    if (!selectedFile) {
      setMessage('Please select or drop a file first.');
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
        throw new Error('Failed to upload and run model.');
      }

      const result = await response.json();
      setResults(result.results);
      setUploadedImage(result.image_url);
      setSaliencyMap(result.saliency_url);
      setMessage(`"${selectedFile.name}" has run successfully.`);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage('Error uploading file.');
    }
  };

  // Function to trigger the file input click when clicking the drop area
  const handleClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="bg-white p-4 h-1/2 w-full flex flex-col justify-center items-center">
      <h2 className="text-3xl font-bold font-martel text-center mb-4">Upload and Run Model</h2>
      <div
        ref={dropRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className="border-2 border-dashed border-gray-300 p-4 rounded-lg flex flex-col items-center justify-center cursor-pointer"
      >
        <img src={uploadIcon} alt="Upload Icon" className="w-16 mb-2" />
        <p className="text-m text-gray-500 font-mono">Drag & drop your image here or click to select</p>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileChange}
          className="hidden"
          accept="image/png, image/jpeg"
        />
      </div>
      {/* Display the selected file name or success message */}
      {selectedFile && !message && (
        <p className="text-sm font-mono text-gray-700 mt-2">
          Selected file: <span className="font-semibold font-mono">{selectedFile.name}</span>
        </p>
      )}
      {message && <p className="text-sm text-gray-700 mt-2 font-mono font-semibold">{message}</p>}
      <button
        onClick={handleRunModel}
        className="w-full bg-blue-500 text-white font-mono py-2 rounded-lg hover:bg-blue-600 mt-4"
      >
        Run Model
      </button>
    </div>
  );
};

export default UploadRunModel;
