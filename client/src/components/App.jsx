import React, { useState } from 'react';
import TypingText from './TypingText';
import Dropzone from './Dropzone';
import { predictAge } from '../services/apiService';
import GitHubIcon from '@mui/icons-material/GitHub';

import '../styling/App.css'; // Import CSS file for styling
import '../styling/TypingText.css'; // Import CSS file for styling

const App = () => {
  const [textDisplayed, setTextDisplayed] = useState(false);
  const [text, setText] = useState("Welcome to Age-I");
  const [flag, setFlag] = useState(true);

  const handleTextDisplayComplete = () => {
    if (flag) {
      setTimeout(() => {
        setText("Upload a face");
        setTimeout(() => {
          setTextDisplayed(true);
        }, 1500);
      }, 1800);
    }
    setFlag(false);
  };

  const handleImageUpload = (filename) => {
    setText("Scanning for wrinkles...");
    
    setTimeout(() => {
      setText("Hmm...");
    }, 2200);

    try {
      setTimeout(async () => {
        const response = await predictAge(filename);
        setText(`You look like you are about ${response.predicted_age} years old`);
        
        setTimeout(() => {
          setText("Want to try again?");
          setTextDisplayed(false); // Reset textDisplayed state to allow user to upload again
          setFlag(true); // Reset flag to restart the text display sequence
        }, 10000); // Wait 10 seconds before asking to try again
      }, 3000);
    } catch (error) {
      console.error('Error predicting age: ', error);
    }
  };

  return (
    <div className="App">
      <div className="content-container">
        <div className="typing-dropzone-container">
          <TypingText text={text} key={text} onTypingComplete={handleTextDisplayComplete} />
          {textDisplayed && <Dropzone onImageUpload={handleImageUpload} />}
        </div>
      </div>
      <footer className="footer">
        <a href="https://github.com/huangk430/age-detector" target="_blank" rel="noopener noreferrer">
          <GitHubIcon />
        </a>
      </footer>
    </div>
  );
};

export default App;
