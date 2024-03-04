import React, { useState } from 'react';
import TypingText from './TypingText';
import { predictAge } from '../services/apiService';

import '../styling/App.css'; // Import CSS file for styling
import '../styling/TypingText.css'; // Import CSS file for styling


const App = () => {
  const [textDisplayed, setTextDisplayed] = useState(false);
  const [predictedAge, setPredictedAge] = useState(null);
  const [text, setText] = useState("Welcome to our age detector")
  const [flag, setFlag] = useState(true)

  const handleTextDisplayComplete = () => {
    if (flag) {
      setTimeout(async () => {
        setText("Upload your face")
        setTimeout(async () => {
          setTextDisplayed(true);
        }, 1500)
      }, 2000);
    } 
    setFlag(false)
  };

  const handleImageUpload = async (e) => {
    setText("Scanning wrinkles...");

    try {
      setTimeout(async () => {
        // setText("You look fucking stupid...")
        const response = await predictAge(e.target.files[0]);
        setText(`You look like you are about ${response.predicted_age} years old`)
      }, 3000);
        
    } catch (error) {
      console.error('Error predicting age: ', error);
    }
  }

  return (
    <div className="App">
      <div className="content-container">
        <TypingText text={text} key={text} onTypingComplete={handleTextDisplayComplete} />
        {textDisplayed && (
          <input
            type="file"
            accept=".jpg"
            className="image-dropzone"
            onChange={handleImageUpload}
          />
        )}
      </div>
    </div>
  );
};

export default App;
