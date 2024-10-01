// src/components/ProjectDetails.jsx

import React from 'react';

const ProjectDetails = () => {
  return (
    <div className="flex flex-col items-start justify-center p-6 bg-white w-full h-1/2 text-left"> {/* Adjusted width and alignment */}
      <h1 className="text-5xl font-martel font-bold mb-6">Diabetic Retinopathy Detection System</h1> {/* Title with Martel font */}
      <p className="text-m font-mono mb-1">Authors: Caelyn Viljoen, Liam G. Hermanus</p> {/* Authors with Assistant font */}
      <p className="text-m font-mono mb-1">Supervisor: Dr Martin Bekker</p> {/* Supervisor with Assistant font */}
      <p className="text-m font-mono mb-1">Group: 30</p> {/* Group with Assistant font */}
    </div>
  );
};

export default ProjectDetails;
