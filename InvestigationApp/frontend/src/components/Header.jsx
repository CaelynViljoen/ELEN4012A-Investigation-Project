// src/components/Header.jsx

import React from 'react';
import logoA from '../assets/LogoA.png';
import logoB from '../assets/LogoB.png';

const Header = () => {
  return (
    <header className="w-full flex items-center justify-center border-b border-gray-300">
      <div className="flex items-center justify-start">
        <img src={logoA} alt="University Logo" className="h-14" /> {/* Align logo to the right */}
      </div>
      <div className="flex-grow text-4xl font-semibold font-martel text-center"> 
        DIABETIC RETINOPATHY DETECTION SYSTEM
      </div>
      <div className="flex items-center">
        <img src={logoB} alt="University Logo" className="h-14 ml-auto" /> {/* Align logo to the right */}
      </div>
    </header>
  );
};

export default Header;
