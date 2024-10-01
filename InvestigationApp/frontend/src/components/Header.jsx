// src/components/Header.jsx

import React from 'react';
import logo from '../assets/logo.png'; // Adjust the path if needed

const Header = () => {
  return (
    <header className="w-full flex items-center justify-between border-b border-gray-300">
      <div className="flex items-center">
        <img src={logo} alt="University Logo" className="h-14" /> {/* Increased height slightly */}
      </div>
      <div className="text-xl font-semibold font-martel"> {/* Changed to a code-like font */}
        Investigation Project
      </div>
    </header>
  );
};

export default Header;
