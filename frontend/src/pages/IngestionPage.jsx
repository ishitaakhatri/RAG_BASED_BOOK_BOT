import React, { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  ArrowLeft,
  AlertCircle,
  CheckCircle,
  Loader,
  Book,
} from "lucide-react";

const API_BASE_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/ingest";

// Extract LogItem to a memoized component for performance
const LogItem = React.memo(({ log, getLogColor }) => (
  <div className={`${getLogColor(log.type)} leading-relaxed flex items-start hover:bg-purple-500/10 px-3 py-1.5 rounded-md transition-all duration-200`}>
    <span className="text-gray-500 text-xs mr-3 flex-shrink-0 select-none font-semibold">
      [{log.timestamp}]
    </span>
    <span className={`mr-3 flex-shrink-0 font-bold text-xs select-none px-2 py-0.5 rounded ${
      log.type === 'error' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
      log.type === 'warning' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30' :
      log.type === 'success' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
      'bg-purple-500/20 text-purple-300 border border-purple-500/30'
    }`}>
      {log.type === 'error' ? 'ERR' :
       log.type === 'warning' ? 'WRN' :
       log.type === 'success' ? 'OK' :
       'INF'}
    </span>
    <span className="flex-1 break-all text-gray-200">{log.message}</span>
  </div>
));

export default function IngestionPage({ books, onUploadSuccess }) {
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const [showLogs, setShowLogs] = useState(true);
  const [logs, setLogs] = useState([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  // Smooth animation state
  const [animatedPercentage, setAnimatedPercentage] = useState(0);
  
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  const processedLogsRef = useRef(new Set()); 
  const finishSequenceStarted = useRef(false); 
  const successTimerRef = useRef(null);
  
  // ✅ NEW: Ref to track ingestion state instantly inside event listeners
  const isIngestingRef = useRef(false);
  
  const navigate = useNavigate();

  // Add custom scrollbar and animation styles
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .scrollbar-thin::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      .scrollbar-thin::-webkit-scrollbar-track {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 4px;
      }
      .scrollbar-thin::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
        border-radius: 4px;
        transition: all 0.3s ease;
      }
      .scrollbar-thin::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #7c3aed, #db2777);
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
      }
      .progress-circle {
        transition: stroke-dashoffset 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      }
      @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(168, 85, 247, 0.4); }
        50% { box-shadow: 0 0 30px rgba(236, 72, 153, 0.6); }
      }
      .terminal-glow {
        animation: glow 3s ease-in-out infinite;
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current && showLogs) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

  // Cleanup Timer on unmount
  useEffect(() => {
    return () => {
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // ✅ FIXED: Robust Completion Logic
  useEffect(() => {
    if (!isIngesting) return;
    
    // Prevent re-entry if finish sequence is already running
    if (successTimerRef.current) return;

    // Check completion conditions
    const isFinishedInLogs = logs.some(l => l.message.includes("Ingestion completed successfully"));
    const isFinishedStatus = liveProgress?.status === "completed" || liveProgress?.status === "success";

    if (isFinishedInLogs || isFinishedStatus) {
      console.log("✅ Success detected, starting cleanup sequence...");
      finishSequenceStarted.current = true;
      
      // Force visual 100%
      if (animatedPercentage < 100) setAnimatedPercentage(100);

      // Start the timer
      successTimerRef.current = setTimeout(() => {
        if (!isIngestingRef.current) return; // Guard clause

        console.log("🎉 Cleanup timer triggered");
        
        const mockResult = {
          chunks: liveProgress?.chunks_created || 0,
          total_pages: liveProgress?.total_pages || 0,
          title: uploadFile?.name,
          method: "stream_verified"
        };

        // 1. Show Success Message
        setUploadProgress({
          status: "success",
          message: "Ingestion completed successfully!",
          percentage: 100,
          result: mockResult,
        });

        // 2. Wait a moment, then Reset UI & Fetch Books
        setTimeout(() => {
          console.log("🧹 Resetting UI state...");
          
          // A. Stop Ingestion Ref first (Blocks new WS messages)
          isIngestingRef.current = false;
          
          // B. Close WebSocket immediately
          if (wsRef.current) {
            wsRef.current.close(1000, "Ingestion complete");
            wsRef.current = null;
          }

          // C. Reset State (Enables buttons/inputs)
          setUploadFile(null);
          setUploadProgress(null);
          setLiveProgress(null);
          setIsIngesting(false); 
          setAnimatedPercentage(0);
          setLogs([]);
          
          // D. Reset Refs
          processedLogsRef.current.clear();
          finishSequenceStarted.current = false;
          successTimerRef.current = null;
          
          // E. Fetch Books (Delayed to ensure Pinecone indexing)
          if (onUploadSuccess) {
             console.log("📚 Refreshing book list...");
             onUploadSuccess();
          }
        }, 2000); // 2s delay for user to see green success message
      }, 1500); // 1.5s delay after completion detection
    }
  }, [logs, liveProgress, isIngesting, animatedPercentage, onUploadSuccess, uploadFile]);

  // Smooth percentage animation
  useEffect(() => {
    const targetPercentage = calculatePercentage();
    let animationFrameId;
    const animateProgress = () => {
      setAnimatedPercentage((prev) => {
        const diff = targetPercentage - prev;
        if (Math.abs(diff) < 0.1) return targetPercentage;
        return prev + diff * 0.15;
      });
      if (Math.abs(targetPercentage - animatedPercentage) > 0.1) {
        animationFrameId = requestAnimationFrame(animateProgress);
      }
    };
    animationFrameId = requestAnimationFrame(animateProgress);
    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [liveProgress, uploadProgress, isIngesting]);

  const addLog = (message, type = "info") => {
    const timestamp = new Date().toLocaleTimeString();
    const logKey = `${timestamp}-${message}`;
    
    if (processedLogsRef.current.has(logKey)) return;
    
    processedLogsRef.current.add(logKey);
    setLogs((prev) => {
      const newLogs = [...prev, { message, type, timestamp }];
      return newLogs.length > 100 ? newLogs.slice(-100) : newLogs;
    });
  };

  const connectWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }
        if (wsRef.current) wsRef.current.close();

        wsRef.current = new WebSocket(WS_URL);

        wsRef.current.onopen = () => {
          addLog("✅ Connected to progress stream", "success");
          resolve();
        };

        wsRef.current.onmessage = (event) => {
          // ✅ GUARD: Block updates if we are done
          if (!isIngestingRef.current) return;

          try {
            const data = JSON.parse(event.data);
            setLiveProgress(data);

            if (data.logs && Array.isArray(data.logs)) {
              data.logs.forEach((logLine) => {
                const match = logLine.match(/\[(\d{2}:\d{2}:\d{2})\] (\w+): (.+)/);
                if (match) {
                  const [, timestamp, level, message] = match;
                  // Only add if we are still ingesting
                  if (isIngestingRef.current) {
                      addLog(message, level.toLowerCase());
                  }
                }
              });
            }
          } catch (error) {
            console.error("Error parsing progress data:", error);
          }
        };

        wsRef.current.onerror = (error) => {
          console.error("WebSocket error:", error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    // Reset everything
    setIsIngesting(true);
    isIngestingRef.current = true; // Sync Ref
    setLogs([]);
    setShowLogs(true);
    setAnimatedPercentage(0);
    processedLogsRef.current.clear();
    finishSequenceStarted.current = false;
    successTimerRef.current = null;
    
    addLog("📋 Starting ingestion process...", "info");

    try {
      await connectWebSocket();
    } catch (error) {
      addLog("⚠️ Could not connect to live progress (will still work)", "warning");
    }

    const formData = new FormData();
    formData.append("file", uploadFile);

    setUploadProgress({
      status: "uploading",
      message: "Uploading and processing PDF...",
      percentage: 5,
    });

    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // We rely on the WebSocket/Log completion detector to handle success UI
        // But we update this state just in case WS failed
        setUploadProgress({
            status: "success",
            message: "Ingestion verified. Finalizing...",
            percentage: 95,
            result: data.result
        });
      } else {
        throw new Error(data.error || "Unknown error");
      }
    } catch (error) {
      // Only show error if we haven't already succeeded via the failsafe
      if (isIngestingRef.current && !finishSequenceStarted.current) {
        setUploadProgress({
            status: "error",
            message: `Upload failed: ${error.message}`,
            percentage: 0,
        });
        addLog(`❌ Upload failed: ${error.message}`, "error");
        setIsIngesting(false);
        isIngestingRef.current = false;
      }
    }
  };

  const handleReset = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, "User cancelled");
    }
    setUploadFile(null);
    setUploadProgress(null);
    setLiveProgress(null);
    setLogs([]);
    setShowLogs(false);
    setIsIngesting(false);
    isIngestingRef.current = false;
    navigate("/");
  };

  // Drag and Drop Handlers
  const handleDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); if (!isIngesting) setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); if (e.currentTarget === e.target) setIsDragging(false); };
  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    if (isIngesting) return;
    const file = e.dataTransfer.files[0];
    if (file && (file.type === "application/pdf" || file.name.endsWith(".pdf"))) {
      setUploadFile(file);
      addLog(`📎 File selected: ${file.name}`, "info");
    }
  };

  const getStatusColor = (status) => {
    if (status === "completed" || status === "success") return "text-green-400";
    if (status === "failed" || status === "error") return "text-red-400";
    return "text-blue-400";
  };

  const getLogColor = (type) => {
    if (type === "success") return "text-green-400";
    if (type === "error") return "text-red-400";
    if (type === "warning") return "text-yellow-400";
    return "text-purple-300";
  };

  const calculatePercentage = () => {
    if (uploadProgress?.status === "success" || liveProgress?.status === "completed") return 100;
    if (liveProgress?.status === "failed") return 0;
    if (!liveProgress) return uploadProgress?.percentage ?? 0;
    
    // Explicit percentage from backend
    if (liveProgress.percentage !== undefined && liveProgress.percentage !== null) {
      return Math.min(Math.max(Number(liveProgress.percentage), 0), 100);
    }
    
    // Heuristic fallback
    const status = liveProgress.status;
    if (status === "parsing_pdf") return 15;
    if (status === "chunking") return 20 + ((liveProgress.current_page || 0) / (liveProgress.total_pages || 1)) * 30;
    if (status === "embedding") return 50 + ((liveProgress.embeddings_generated || 0) / (liveProgress.chunks_created || 1)) * 30;
    if (status === "upserting") return 80 + ((liveProgress.vectors_upserted || 0) / (liveProgress.chunks_created || 1)) * 18;
    
    return Math.max(uploadProgress?.percentage ?? 0, 5);
  };

  const currentPercentage = Math.round(animatedPercentage);
  const currentStatus = liveProgress?.status || uploadProgress?.status || "idle";
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const renderedLogs = useMemo(() => logs.map((log, idx) => <LogItem key={idx} log={log} getLogColor={getLogColor} />), [logs]);

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-lg border-b border-white/10 sticky top-0 z-10 flex-shrink-0">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <button
                onClick={handleReset}
                className="p-2 hover:bg-white/10 rounded-lg transition-all text-white disabled:opacity-50"
                disabled={isIngesting}
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Ingest Books</h1>
                <p className="text-sm text-purple-200">
                  Add new books to your knowledge base
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-purple-300">
                Books in system:{" "}
                <span className="font-bold text-white">{books.length}</span>
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* LEFT SIDEBAR */}
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Book className="w-5 h-5 mr-2" />
                    Available Books
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin">
                    <button className="w-full text-left px-3 py-2 rounded-lg transition-all bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold">
                      All Books
                    </button>
                    {books.map((book, idx) => (
                      <button key={idx} className="w-full text-left px-3 py-2 rounded-lg transition-all bg-white/5 hover:bg-white/10 text-purple-200 hover:text-white">
                        <div className="text-sm font-semibold truncate">{book.title}</div>
                        <div className="text-xs opacity-75 truncate">by {book.author}</div>
                        {book.total_chunks > 0 && <div className="text-xs opacity-60 mt-1">{book.total_chunks} chunks</div>}
                      </button>
                    ))}
                  </div>
                </div>
                {/* Stats */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4">Stats</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center"><span className="text-purple-200 text-sm">Total Books:</span><span className="font-bold text-white text-lg">{books.length}</span></div>
                    <div className="flex justify-between items-center"><span className="text-blue-200 text-sm">Total Chunks:</span><span className="font-bold text-white text-lg">{books.reduce((sum, b) => sum + (b.total_chunks || 0), 0)}</span></div>
                  </div>
                </div>
              </div>

              {/* RIGHT SIDE - Upload Form */}
              <div className="lg:col-span-3 space-y-6">
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
                  <h2 className="text-2xl font-bold text-white mb-2">Upload PDF Books</h2>
                  <p className="text-purple-200 mb-6">Upload PDF files to ingest them into your knowledge base.</p>

                  <div className="space-y-6">
                    {/* File Input Zone */}
                    {!isIngesting && (
                      <>
                        <div>
                          <label className="block text-sm font-semibold text-purple-200 mb-3">Select PDF File</label>
                          <label 
                            htmlFor="file-upload" 
                            className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
                              isDragging 
                                ? 'border-pink-400 bg-gradient-to-br from-pink-500/30 to-purple-500/30 scale-[1.02] shadow-lg shadow-pink-500/50' 
                                : 'border-purple-400/70 bg-gradient-to-br from-purple-500/10 to-pink-500/10 hover:from-purple-500/20 hover:to-pink-500/20 hover:border-purple-300'
                            }`}
                            onDragEnter={handleDragEnter} onDragLeave={handleDragLeave} onDragOver={handleDragOver} onDrop={handleDrop}
                          >
                            <div className="flex flex-col items-center justify-center pt-8 pb-6">
                              <Upload className={`w-12 h-12 mb-2 transition-all duration-300 ${isDragging ? 'text-pink-300 scale-125 rotate-12' : 'text-purple-300'}`} />
                              <p className="text-sm font-semibold text-white">{isDragging ? '🚀 Drop your PDF here' : 'Click to upload or drag and drop'}</p>
                              <p className="text-xs text-purple-300 mt-1">PDF files only</p>
                            </div>
                            <input
                              id="file-upload" type="file" accept=".pdf"
                              onClick={(e) => (e.target.value = null)}
                              onChange={(e) => {
                                const file = e.target.files[0];
                                if (file) { setUploadFile(file); addLog(`📎 File selected: ${file.name}`, "info"); }
                              }}
                              className="hidden" disabled={isIngesting}
                            />
                          </label>
                          {uploadFile && (
                            <div className="mt-4 p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-400/50 rounded-lg shadow-lg">
                              <div className="flex items-center space-x-3">
                                <CheckCircle className="w-5 h-5 text-green-400" />
                                <div>
                                  <p className="text-sm font-semibold text-green-300">{uploadFile.name}</p>
                                  <p className="text-xs text-green-200">{(uploadFile.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                        {/* Requirements Box */}
                        <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-400/30 rounded-lg p-4">
                          <h4 className="text-sm font-semibold text-purple-300 mb-2 flex items-center"><AlertCircle className="w-4 h-4 mr-2" />File Requirements</h4>
                          <ul className="text-xs text-purple-200 space-y-1">
                            <li>• File format: PDF (.pdf)</li>
                            <li>• Recommended file size: Less than 50 MB</li>
                            <li>• Filename format: "Book Title - Author Name.pdf"</li>
                          </ul>
                        </div>
                      </>
                    )}

                    {/* Progress Section */}
                    {(uploadProgress || liveProgress || isIngesting) && (
                      <div className="bg-gradient-to-br from-purple-900/30 via-pink-900/20 to-purple-900/30 rounded-xl p-6 border border-purple-400/30 shadow-2xl">
                        <h3 className="text-xl font-bold text-white mb-6 flex items-center">
                          <div className={`w-2 h-2 rounded-full mr-3 ${currentPercentage === 100 ? 'bg-green-400' : 'bg-green-400 animate-pulse'}`}></div>
                          Ingestion Progress
                        </h3>
                        <div className="space-y-6">
                          <div className="flex items-center justify-center">
                            <div className="relative w-32 h-32">
                              <svg className="w-full h-full transform -rotate-90">
                                <circle cx="64" cy="64" r="56" stroke="rgba(139, 92, 246, 0.2)" strokeWidth="8" fill="none" />
                                <circle cx="64" cy="64" r="56" stroke="url(#grad)" strokeWidth="8" fill="none" strokeDasharray={circumference} strokeDashoffset={circumference - (animatedPercentage / 100) * circumference} strokeLinecap="round" className="progress-circle" style={{ filter: 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.6))' }} />
                                <defs>
                                  <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#8b5cf6" /><stop offset="50%" stopColor="#a855f7" /><stop offset="100%" stopColor="#ec4899" />
                                  </linearGradient>
                                </defs>
                              </svg>
                              <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <p className="text-3xl font-bold bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">{currentPercentage}%</p>
                                <p className="text-xs text-purple-300 mt-1">{currentStatus}</p>
                              </div>
                            </div>
                          </div>
                          
                          <div className="bg-black/30 rounded-lg p-4 border border-purple-500/30">
                            <p className={`text-center text-sm font-semibold ${getStatusColor(currentStatus)}`}>
                              {liveProgress?.current_task || uploadProgress?.message || "Ready"}
                            </p>
                          </div>

                          {uploadProgress?.status === "success" && (
                            <div className="p-4 rounded-lg bg-green-500/20 border border-green-400/50 flex items-center space-x-3">
                              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                              <div>
                                <p className="text-sm font-semibold text-green-300">Success!</p>
                                <p className="text-xs text-green-200">{uploadProgress.result?.chunks} chunks created</p>
                              </div>
                            </div>
                          )}

                          {isIngesting && (
                            <div className="mt-6 bg-black/90 rounded-xl border-2 border-purple-500/50 shadow-2xl terminal-glow overflow-hidden">
                              <div className="bg-gradient-to-r from-purple-900/80 via-pink-900/80 to-purple-900/80 px-4 py-3 border-b border-purple-500/50 flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                  <div className="flex space-x-2"><div className="w-3 h-3 rounded-full bg-red-500"></div><div className="w-3 h-3 rounded-full bg-yellow-500"></div><div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div></div>
                                  <span className="text-sm text-purple-300 font-mono font-semibold">backend@rag-bot:~$</span>
                                </div>
                                <span className="text-xs text-gray-400 font-mono bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/30">{logs.length} lines</span>
                              </div>
                              <div className="p-5 h-96 overflow-y-auto scrollbar-thin font-mono text-xs bg-gradient-to-b from-black/95 to-gray-900/95">
                                {logs.length === 0 ? (
                                  <div className="flex items-center justify-center h-full flex-col space-y-4">
                                    <Loader className="w-10 h-10 animate-spin text-purple-400" />
                                    <p className="text-purple-300">Connecting to logger...</p>
                                  </div>
                                ) : (
                                  <div className="space-y-1">{renderedLogs}<div ref={logsEndRef} /></div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Buttons */}
                    <div className="flex gap-3 pt-4">
                      <button
                        onClick={handleUploadSubmit} disabled={!uploadFile || isIngesting}
                        className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold flex items-center justify-center space-x-2"
                      >
                        {isIngesting ? <><Loader className="w-5 h-5 animate-spin" /><span>Processing...</span></> : <><Upload className="w-5 h-5" /><span>Upload & Ingest</span></>}
                      </button>
                      <button
                        onClick={handleReset} disabled={isIngesting}
                        className="px-6 py-3 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}