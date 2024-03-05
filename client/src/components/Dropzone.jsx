import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import "../styling/Dropzone.css";

const Dropzone = ({ onImageUpload }) => {
  const [uploaded, setUploaded] = useState(false);
  const [files, setFiles] = useState([]);

  const onDrop = useCallback((acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    setFiles(
      acceptedFiles.map((file) =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
        })
      )
    );

    const reader = new FileReader();
    reader.onload = () => {
      const fileAsArrayBuffer = reader.result;
      const uploadFile = new File([fileAsArrayBuffer], uploadedFile.name, { type: uploadedFile.type });
      onImageUpload(uploadFile);
      setUploaded(true);
    };
    reader.onabort = () => console.log("File reading was aborted");
    reader.onerror = () => console.log("File reading has failed");
    reader.readAsArrayBuffer(uploadedFile);
  }, [onImageUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg'],
     },
    maxSize: 1024 * 1024 * 5,
    maxFiles: 1,
  });

  const fileList = files.map((file) => (
    <li key={file.name}>
      <img className="image-preview" src={file.preview} alt={file.name} />
    </li>
  ));

  return (
    <div className="dropzone-container">
      {uploaded ? (
        <div className="image-container">
          <ul>{fileList}</ul>
        </div>
      ) : (
        <div
          className={`dropzone ${isDragActive ? "active-dropzone" : ""}`}
          {...getRootProps()}
        >
          <input {...getInputProps()} />
          <p className="dropzone-text">
            Here please...
          </p>
        </div>
      )}
    </div>
  );
};

export default Dropzone;
