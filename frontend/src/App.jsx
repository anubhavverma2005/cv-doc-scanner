import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, Image as ImageIcon, Activity, Cpu, ScanLine, Type } from 'lucide-react';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [removeMoire, setRemoveMoire] = useState(false);
  const [fftCutoff, setFftCutoff] = useState(30);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const processDocument = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    if (removeMoire) {
      formData.append('remove_moire', 'true');
      formData.append('fft_cutoff', fftCutoff);
    }

    try {
      // Ensure your Python backend is running on port 5000
      const response = await axios.post('http://127.0.0.1:5000/process-document', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to process document. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const MetricsCard = ({ title, metrics, icon: Icon }) => (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 p-4 flex items-center gap-2 text-white">
        <Icon className="w-5 h-5" />
        <h3 className="font-bold">{title}</h3>
      </div>
      <div className="p-6 space-y-4">
        {Object.entries(metrics).map(([key, value]) => {
          let displayValue = value;
          if (typeof value === 'number') {
            displayValue = value.toFixed(2);
          }
          return (
            <div key={key} className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-sm font-medium text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="text-sm font-bold text-gray-900">{displayValue}</span>
            </div>
          );
        })}
      </div>
    </div>
  );

  const ImageCard = ({ title, imageSrc, icon: Icon, desc }) => (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200 flex flex-col h-full">
      <div className="bg-gray-50 p-3 border-b border-gray-200 flex items-center gap-2">
        <Icon className="w-5 h-5 text-blue-600" />
        <h3 className="font-bold text-gray-700">{title}</h3>
      </div>
      <div className="relative group flex-grow bg-gray-100 min-h-[200px] flex items-center justify-center">
        {imageSrc ? (
          <img 
            src={`data:image/jpeg;base64,${imageSrc}`} 
            alt={title} 
            className="max-w-full max-h-[300px] object-contain" 
          />
        ) : (
          <div className="text-gray-400 text-sm">Pending Processing...</div>
        )}
      </div>
      <div className="p-3 text-xs text-gray-500 bg-gray-50 border-t border-gray-100">
        {desc}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-100 p-8 font-sans text-slate-900">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-extrabold text-slate-800">
            Neural<span className="text-blue-600">Scan</span> Pipeline
          </h1>
          <p className="text-slate-600">
            CPU-Based Document Reconstruction System
          </p>
        </header>

        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-sm p-8 border border-slate-200 flex flex-col items-center gap-6">
          <div className="w-full max-w-md">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-blue-300 border-dashed rounded-lg cursor-pointer bg-blue-50 hover:bg-blue-100 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-8 h-8 mb-3 text-blue-500" />
                <p className="mb-2 text-sm text-blue-600"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                <p className="text-xs text-blue-500">PNG, JPG (MAX. 10MB)</p>
              </div>
              <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
            </label>
          </div>

          {preview && (
            <div className="flex flex-col items-center gap-4">
              <img src={preview} alt="Preview" className="h-48 rounded shadow-md" />

              {/* Moire Removal Options */}
              <div className="flex items-center justify-center gap-4 p-4 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="moire-checkbox"
                    checked={removeMoire}
                    onChange={(e) => setRemoveMoire(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor="moire-checkbox" className="font-medium text-slate-700">
                    Fix Moiré Noise (FFT)
                  </label>
                </div>
                {removeMoire && (
                  <div className="flex items-center gap-2">
                    <label htmlFor="fft-cutoff" className="text-sm text-slate-600">Cutoff:</label>
                    <input
                      type="range"
                      id="fft-cutoff"
                      min="10"
                      max="50"
                      value={fftCutoff}
                      onChange={(e) => setFftCutoff(e.target.value)}
                      className="w-24"
                    />
                    <span className="text-sm font-bold text-slate-800 w-8">{fftCutoff}</span>
                  </div>
                )}
              </div>

              <button 
                onClick={processDocument}
                disabled={loading}
                className={`px-8 py-3 rounded-full font-bold text-white shadow-lg transition-all transform hover:scale-105 ${loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}
              >
                {loading ? 'Processing Pipeline...' : 'Run Pipeline'}
              </button>
            </div>
          )}

          {error && <div className="text-red-500 font-medium">{error}</div>}
        </div>

        {/* Results Grid */}
        {results && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-10 duration-700">
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
              <ImageCard 
                title="0. Pre-process" 
                imageSrc={results.corrected} 
                icon={ScanLine}
                desc={results.moire_removed ? `FFT Moire Removal (Cutoff: ${results.fft_cutoff}) + Skew Correction.` : "PCA-based automatic tilt detection and straightening."}
              />
              <ImageCard 
                title="1. Edge Detection" 
                imageSrc={results.edges} 
                icon={Activity}
                desc="Canny edge detection + Bilateral filtering to isolate document boundaries."
              />
              <ImageCard 
                title="2. Perspective Warp" 
                imageSrc={results.scanned} 
                icon={ScanLine}
                desc="4-Point transform applied to flatten the detected quadrilateral."
              />
              <ImageCard 
                title="3. AI Enhancement" 
                imageSrc={results.enhanced} 
                icon={Cpu}
                desc="FSRCNN Super-Resolution (4x Upscale) via dnn_superres."
              />
              <ImageCard 
                title="4. Binarization" 
                imageSrc={results.binary} 
                icon={ImageIcon}
                desc="Adaptive Gaussian Thresholding for high-contrast OCR prep."
              />
            </div>

            {/* Text Extraction Result */}
            <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
              <div className="bg-slate-800 p-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-white">
                  <FileText className="w-5 h-5" />
                  <h3 className="font-bold">Stage 5: OCR Extraction Output</h3>
                </div>
                <button 
                  onClick={() => navigator.clipboard.writeText(results.text)}
                  className="text-xs bg-slate-700 hover:bg-slate-600 text-white px-3 py-1 rounded transition-colors"
                >
                  Copy Text
                </button>
              </div>
              <div className="p-6 bg-slate-50">
                <pre className="whitespace-pre-wrap font-mono text-sm text-slate-700 bg-white p-4 rounded border border-slate-200 min-h-[150px]">
                  {results.text || "No text detected."}
                </pre>
              </div>
            </div>

            {/* Quality Metrics Section */}
            {results.metrics && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-slate-800">Performance Metrics</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {results.metrics.preprocessing && (
                    <MetricsCard 
                      title="Preprocessing Quality" 
                      metrics={results.metrics.preprocessing}
                      icon={Activity}
                    />
                  )}
                  {results.metrics.ocr && (
                    <MetricsCard 
                      title="OCR Confidence & Accuracy" 
                      metrics={results.metrics.ocr}
                      icon={Type}
                    />
                  )}
                </div>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
};

export default App;