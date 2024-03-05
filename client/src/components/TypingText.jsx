import { useState, useEffect } from 'react';


const TypingText = ({ text, onTypingComplete }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDisplayText(text.slice(0, currentIndex + 1));
      setCurrentIndex(prevIndex => prevIndex + 1);
      if (currentIndex === text.length - 1) {
        clearInterval(interval);
        onTypingComplete(); // Call the callback when typing is complete
      }
    }, 60); // Adjust typing speed here (milliseconds per character)

    return () => clearInterval(interval);
  }, [currentIndex, text, onTypingComplete]);

  return (
    <div className="typing-text-container">
      <span className="typing-text">{displayText}</span>
      <span className="cursor" />
    </div>
  );
};

export default TypingText;
