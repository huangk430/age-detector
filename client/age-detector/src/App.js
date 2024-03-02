import { useState, useEffect } from 'react';
import './App.css';

const App = () => {
  const words = ['Hello', 'World', 'This', 'Is', 'A', 'Typing', 'Effect'];
  const [currentIndex, setCurrentIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    let timeout;

    const typeWord = (word, index) => {
      timeout = setTimeout(() => {
        setDisplayedText((prevText) => prevText + word[index]);
        if (index < word.length - 1) {
          typeWord(word, index + 1);
        } else {
          setTimeout(() => {
            typeNextWord();
          }, 80); // Adjust delay between words
        }
      }, 40); // Adjust typing speed as needed
    };

    const typeNextWord = () => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % words.length);
      // setDisplayedText('');
      typeWord(words[(currentIndex + 1) % words.length], 0);
    };

    typeWord(words[currentIndex], 0);

    return () => clearTimeout(timeout);
  }, [currentIndex]);

  return (
    <div className="typing-effect">
      <span>{displayedText}</span>
      <span className="cursor" /> {/* Cursor blinking effect */}
    </div>
  );
};

export default App;
