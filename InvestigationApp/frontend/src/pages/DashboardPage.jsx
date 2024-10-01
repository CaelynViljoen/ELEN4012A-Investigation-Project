// src/pages/DashboardPage.jsx

import React, { useState } from 'react';
import Header from '../components/Header';
import ProjectDetails from '../components/ProjectDetails';
import Eyeball3D from '../components/Eyeball3D';
import UploadRunModel from '../components/UploadRunModel';
import DisplayResults from '../components/DisplayResults';
import DisplayUploadedImage from '../components/DisplayUploadedImage';
import DisplaySaliencyMap from '../components/DisplaySaliencyMap';

const DashboardPage = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [saliencyMap, setSaliencyMap] = useState(null);
  const [results, setResults] = useState(null);

  return (
    <div className="min-h-screen flex flex-col items-center bg-white p-2">
      <Header />

      <div className="flex flex-row w-full h-full p-4">
        {/* Column 1 - Project Details and Eyeball */}
        <div className="flex flex-col items-start w-1/3">
          <ProjectDetails />
          <Eyeball3D />
        </div>

        {/* Column 2 - Upload/Run Model and Display Results */}
        <div className="flex flex-col items-start w-1/3 px-4">
          <UploadRunModel
            setUploadedImage={setUploadedImage}
            setSaliencyMap={setSaliencyMap}
            setResults={setResults}
          />
          <DisplayResults results={results} />
        </div>

        {/* Column 3 - Display Uploaded Image and Saliency Map */}
        <div className="flex flex-col items-start w-1/3">
          <DisplayUploadedImage uploadedImage={uploadedImage} />
          <DisplaySaliencyMap saliencyMap={saliencyMap} />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
