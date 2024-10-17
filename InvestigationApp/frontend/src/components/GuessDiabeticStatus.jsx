// src/components/GuessDiabeticStatus.jsx

import React, { useState, useEffect } from "react";

const GuessDiabeticStatus = ({ modelResult, guessFeedback, setGuessFeedback, resetGuess }) => {
  const [selectedOption, setSelectedOption] = useState(null);

  const handleOptionSelect = (option) => {
    setSelectedOption(option);
    setGuessFeedback(""); // Clear previous feedback
  };

  useEffect(() => {
    if (modelResult && selectedOption) {
      const checkGuess = async () => {
        try {
          const response = await fetch('http://127.0.0.1:5000/check-guess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              classification: modelResult.classification,
              guess: selectedOption,
            }),
          });

          const data = await response.json();
          if (data.correct) {
            setGuessFeedback("Correct prediction!");
          } else {
            setGuessFeedback("Incorrect prediction.");
          }
        } catch (error) {
          setGuessFeedback("Error checking the prediction.");
        }
      };

      checkGuess();
    }
  }, [modelResult, selectedOption]);

  // Reset the selected option when resetGuess is true
  useEffect(() => {
    if (resetGuess) {
      setSelectedOption(null);
    }
  }, [resetGuess]);

  return (
    <div className="bg-black p-4 h-1/2 w-full flex flex-col justify-center items-center text-white font-martel font-bold relative rounded-b-lg box-border">
      <h2 className="text-2xl text-center mb-2">Make Your Prediction:</h2>
      <div className="flex space-x-4 mb-6">
        <button
          className={`px-4 py-2 font-bold rounded-lg ${
            selectedOption === "diabetic" ? "bg-blue-500" : "bg-gray-400"
          }`}
          onClick={() => handleOptionSelect("diabetic")}
        >
          Diabetic
        </button>
        <button
          className={`px-4 py-2 font-bold rounded-lg ${
            selectedOption === "non-diabetic" ? "bg-blue-500" : "bg-gray-400"
          }`}
          onClick={() => handleOptionSelect("non-diabetic")}
        >
          Non-Diabetic
        </button>
      </div>
      {guessFeedback && <p className="mt-4 text-3xl">{guessFeedback}</p>}
    </div>
  );
};

export default GuessDiabeticStatus;
