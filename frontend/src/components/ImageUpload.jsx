import { useEffect, useState } from "react";

function ImageDropzone({ label, file, previewUrl, onSelect }) {
  const handleChange = (event) => {
    const selected = event.target.files?.[0];
    if (selected) onSelect(selected);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onSelect(dropped);
  };

  return (
    <div className="upload-card">
      <p className="upload-label">{label}</p>
      <label
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input type="file" accept="image/*" hidden onChange={handleChange} />
        {previewUrl ? (
          <img src={previewUrl} alt={label} className="preview-image" />
        ) : (
          <span>Click or drop an image</span>
        )}
      </label>
      {file && <p className="file-name">{file.name}</p>}
    </div>
  );
}

export default function ImageUpload({ preFile, postFile, onPreChange, onPostChange }) {
  const [preUrl, setPreUrl] = useState(null);
  const [postUrl, setPostUrl] = useState(null);

  useEffect(() => {
    if (!preFile) {
      setPreUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(preFile);
    setPreUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [preFile]);

  useEffect(() => {
    if (!postFile) {
      setPostUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(postFile);
    setPostUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [postFile]);

  return (
    <div className="panel">
      <h2>1. Upload images</h2>
      <div className="upload-grid">
        <ImageDropzone
          label="Before scrubbing"
          file={preFile}
          previewUrl={preUrl}
          onSelect={onPreChange}
        />
        <ImageDropzone
          label="After scrubbing"
          file={postFile}
          previewUrl={postUrl}
          onSelect={onPostChange}
        />
      </div>
    </div>
  );
}
