// src/pages/DashboardPage.jsx

import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import UploadRunModel from '../components/UploadRunModel';
import DisplayResults from '../components/DisplayResults';
import DisplayUploadedImage from '../components/DisplayUploadedImage';
import DisplaySaliencyMap from '../components/DisplaySaliencyMap';
import DisplayExudates from '../components/DisplayExudates';
import GuessDiabeticStatus from '../components/GuessDiabeticStatus';

const DashboardPage = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [saliencyMap, setSaliencyMap] = useState(null);
  const [exudates, setExudates] = useState(null);
  const [results, setResults] = useState(null);
  const [guessFeedback, setGuessFeedback] = useState("");
  const [resetGuess, setResetGuess] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);

  const clearResults = () => {
    setSaliencyMap(null);
    setExudates(null);
    setResults(null);
    setGuessFeedback("");
    setResetGuess(true);
  };

  const handleImageUpload = (imageUrl) => {
    setUploadedImage(imageUrl);
    setResetGuess(false);
  };

  const toggleFullScreen = () => {
    if (!isFullScreen) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen();
      } else if (document.documentElement.webkitRequestFullscreen) { 
        document.documentElement.webkitRequestFullscreen();
      } else if (document.documentElement.msRequestFullscreen) { 
        document.documentElement.msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) { 
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) { 
        document.msExitFullscreen();
      }
    }
    setIsFullScreen(!isFullScreen);
  };

  useEffect(() => {
    const handleFullScreenChange = () => {
      const isCurrentlyFullScreen = !!document.fullscreenElement;
      setIsFullScreen(isCurrentlyFullScreen);
    };

    document.addEventListener('fullscreenchange', handleFullScreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullScreenChange);
    document.addEventListener('msfullscreenchange', handleFullScreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullScreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullScreenChange);
      document.removeEventListener('msfullscreenchange', handleFullScreenChange);
    };
  }, []);

  return (
    <div className={`min-h-screen flex flex-col items-center bg-white p-2 ${isFullScreen ? 'fixed inset-0 z-50' : ''}`}>
      <Header />
      <div className={`flex ${isFullScreen ? 'flex-grow' : 'h-full'} flex-row w-full p-4`}>
        <div className="flex flex-col items-start w-1/3">
          <DisplayUploadedImage uploadedImage={uploadedImage} />
          <GuessDiabeticStatus 
            modelResult={results} 
            guessFeedback={guessFeedback} 
            setGuessFeedback={setGuessFeedback} 
            resetGuess={resetGuess} 
          />
        </div>

        <div className="flex flex-col items-start w-1/3 px-20 relative">
          <UploadRunModel
            setUploadedImage={handleImageUpload}
            setSaliencyMap={setSaliencyMap}
            setExudates={setExudates}
            setResults={setResults}
            uploadedImage={uploadedImage}
            clearResults={clearResults}
          />
          <DisplayResults results={results} />
          <button
            onClick={toggleFullScreen}
            className={`absolute bottom-0 left-3 bg-blue-900 text-white px-1 py-0 rounded hover:bg-blue-700 focus:outline-none`}
          >
            {isFullScreen ? 'Exit Full Screen' : 'Full Screen'}
          </button>
        </div>

        <div className="flex flex-col items-start w-1/3">
          <DisplayExudates exudates={exudates} />
          <DisplaySaliencyMap saliencyMap={saliencyMap} />
        </div>
      </div>
      {!isFullScreen && <Footer />}
    </div>
  );
};

export default DashboardPage;
