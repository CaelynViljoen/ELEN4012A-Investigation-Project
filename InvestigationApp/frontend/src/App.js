// src/App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './styles/tailwind.css';
import DashboardPage from './pages/DashboardPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<DashboardPage />} /> {/* Set Dashboard as the landing page */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
