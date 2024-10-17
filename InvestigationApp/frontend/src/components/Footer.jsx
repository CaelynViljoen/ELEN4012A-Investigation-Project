// src/components/ProjectDetails.jsx

import React from 'react';

const Footer = () => {
  return (
    <footer className="w-full flex items-center text-xs font-mono justify-between border-t border-gray-300 p-4"> {/* Moved the border to the top */}
      <div className="w-3/4"> {/* Limits the width of the disclaimer to 2/3 */}
        <p className="text-left">
          Disclaimer: This system is intended for research and educational purposes only. It should not be used for diagnostic purposes or as a substitute for professional medical advice. Users should consult with qualified healthcare providers for accurate and personalised medical recommendations.
        </p>
      </div>
      <div className="text-left">
        <p>Authors: Caelyn Viljoen, Liam G. Hermanus</p>
        <p>Supervisor: Dr. Martin Bekker</p>
        <p>Group: 24G30</p>
      </div>
    </footer>
  );
};

export default Footer;

