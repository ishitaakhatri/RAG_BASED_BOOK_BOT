import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  ArrowLeft,
  AlertCircle,
  CheckCircle,
  Loader,
  Book,
  Eye,
  EyeOff,
} from "lucide-react";

const API_BASE_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/ingest";

export default function IngestionPage({ books, onUploadSuccess }) {
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (logsEndRef.current && showLogs) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const addLog = (message, type = "info") => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { message, type, timestamp }]);
  };

  const connectWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        wsRef.current = new WebSocket(WS_URL);

        wsRef.current.onopen = () => {
          addLog("✅ Connected to progress stream", "success");
          resolve();
        };

        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setLiveProgress(data);
            console.log("📊 Live progress update:", data);

            if (data.status === "chunking") {
              addLog(
                `📊 Chunking: ${data.chunks_created} chunks created`,
                "info"
              );
            } else if (data.status === "embedding") {
              addLog(
                `🧠 Embedding: ${data.embeddings_generated}/${data.chunks_created} embeddings`,
                "info"
              );
            } else if (data.status === "upserting") {
              addLog(
                `📤 Upserting: ${data.vectors_upserted} vectors to Pinecone`,
                "info"
              );
            } else if (data.status === "failed") {
              addLog(`❌ Ingestion failed: ${data.errors.join(", ")}`, "error");
            } else if (data.status === "completed") {
              addLog(
                `✅ Ingestion completed! ${data.chunks_created} chunks indexed`,
                "success"
              );

              if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
              }
            }
          } catch (error) {
            console.error("Error parsing progress data:", error);
          }
        };

        wsRef.current.onerror = (error) => {
          console.error("WebSocket error:", error);
          addLog("❌ WebSocket connection error", "error");
          reject(error);
        };

        wsRef.current.onclose = () => {
          addLog("⚠️ Disconnected from progress stream", "warning");
        };
      } catch (error) {
        reject(error);
      }
    });
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIsIngesting(true);
    setLogs([]);
    setShowLogs(true);
    addLog("📋 Starting ingestion process...", "info");

    try {
      await connectWebSocket();
    } catch (error) {
      addLog(
        "⚠️ Could not connect to live progress (will still work)",
        "warning"
      );
    }

    const formData = new FormData();
    formData.append("file", uploadFile);

    setUploadProgress({
      status: "uploading",
      message: "Uploading and processing PDF...",
      percentage: 10,
    });

    addLog(`📁 Uploading: ${uploadFile.name}`, "info");

    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        const result = data.result;
        setUploadProgress({
          status: "success",
          message: `Successfully ingested "${
            result.title || uploadFile.name
          }"! Created ${result.chunks} chunks.`,
          percentage: 100,
          result,
        });

        addLog(
          `✅ Ingestion complete: ${result.chunks} chunks from ${result.total_pages} pages`,
          "success"
        );
        addLog(`📚 Method used: ${result.method}`, "success");

        if (onUploadSuccess) {
          setTimeout(() => {
            onUploadSuccess();
            setUploadFile(null);
            setUploadProgress(null);
            setLiveProgress(null);
            setIsIngesting(false);
          }, 2000);
        }
      } else {
        setUploadProgress({
          status: "error",
          message: data.error || "Upload failed",
          percentage: 0,
        });
        addLog(`❌ Upload error: ${data.error}`, "error");
        setIsIngesting(false);
      }
    } catch (error) {
      setUploadProgress({
        status: "error",
        message: `Upload failed: ${error.message}`,
        percentage: 0,
      });
      addLog(`❌ Upload failed: ${error.message}`, "error");
      setIsIngesting(false);
    }
  };

  const handleReset = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setUploadFile(null);
    setUploadProgress(null);
    setLiveProgress(null);
    setLogs([]);
    setShowLogs(false);
    setIsIngesting(false);
    navigate("/");
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "completed":
        return "text-green-400";
      case "failed":
        return "text-red-400";
      case "upserting":
      case "embedding":
      case "chunking":
        return "text-blue-400";
      case "parsing_pdf":
        return "text-yellow-400";
      default:
        return "text-purple-400";
    }
  };

  // Enhanced: Map all statuses to a progress percentage
  const calculatePercentage = () => {
    if (!liveProgress) {
      return uploadProgress?.percentage ?? 0;
    }

    // Use backend-provided percentage FIRST (most accurate)
    if (typeof liveProgress.percentage === "number") {
      return Math.round(liveProgress.percentage);
    }

    // Fallbacks (rarely needed)
    if (liveProgress.status === "completed") return 100;
    if (liveProgress.status === "failed") return 0;
    if (liveProgress.status === "parsing_pdf") return 10;

    return uploadProgress?.percentage ?? 0;
  };

  const currentPercentage = calculatePercentage();

  const currentStatus =
    liveProgress?.status || uploadProgress?.status || "idle";

  const radius = 48;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-lg border-b border-white/10 sticky top-0 z-10">
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

      {/* Main Content - Scrollable */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-white/5">
        <div className="px-4 sm:px-6 lg:px-8 py-8">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* LEFT SIDEBAR - Books & Stats */}
              <div className="lg:col-span-1 space-y-6">
                {/* Books List */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Book className="w-5 h-5 mr-2" />
                    Available Books
                  </h3>

                  {/* Books Vault - Scrollable */}
                  <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-white/5">
                    <button className="w-full text-left px-3 py-2 rounded-lg transition-all bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold">
                      All Books
                    </button>

                    {books.map((book, idx) => (
                      <button
                        key={idx}
                        className="w-full text-left px-3 py-2 rounded-lg transition-all bg-white/5 hover:bg-white/10 text-purple-200 hover:text-white"
                      >
                        <div className="text-sm font-semibold truncate">
                          {book.title}
                        </div>
                        <div className="text-xs opacity-75 truncate">
                          by {book.author}
                        </div>
                        {book.total_chunks > 0 && (
                          <div className="text-xs opacity-60 mt-1">
                            {book.total_chunks} chunks
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Stats Card */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Stats
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-purple-200 text-sm">
                        Total Books:
                      </span>
                      <span className="font-bold text-white text-lg">
                        {books.length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-purple-200 text-sm">
                        Total Chunks:
                      </span>
                      <span className="font-bold text-white text-lg">
                        {books.reduce(
                          (sum, b) => sum + (b.total_chunks || 0),
                          0
                        )}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* RIGHT SIDE - Upload Form & Progress */}
              <div className="lg:col-span-3 space-y-6">
                {/* Upload Form Card */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
                  <h2 className="text-2xl font-bold text-white mb-2">
                    Upload PDF Books
                  </h2>
                  <p className="text-purple-200 mb-6">
                    Upload PDF files to ingest them into your knowledge base.
                  </p>

                  <div className="space-y-6">
                    {/* File Input */}
                    <div>
                      <label className="block text-sm font-semibold text-purple-200 mb-3">
                        Select PDF File
                      </label>
                      <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-purple-400 rounded-xl cursor-pointer bg-white/5 hover:bg-white/10 transition-all">
                        <div className="flex flex-col items-center justify-center pt-8 pb-6">
                          <Upload className="w-12 h-12 text-purple-300 mb-2" />
                          <p className="text-sm font-semibold text-white">
                            Click to upload or drag and drop
                          </p>
                          <p className="text-xs text-purple-300">
                            PDF files only
                          </p>
                        </div>
                        <input
                          type="file"
                          accept=".pdf"
                          onChange={(e) => setUploadFile(e.target.files[0])}
                          className="hidden"
                          disabled={isIngesting}
                        />
                      </label>
                      {uploadFile && (
                        <div className="mt-4 p-4 bg-green-500/20 border border-green-400/50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <CheckCircle className="w-5 h-5 text-green-400" />
                            <div>
                              <p className="text-sm font-semibold text-green-300">
                                {uploadFile.name}
                              </p>
                              <p className="text-xs text-green-200">
                                {(uploadFile.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* File Requirements */}
                    <div className="bg-blue-500/10 border border-blue-400/30 rounded-lg p-4">
                      <h4 className="text-sm font-semibold text-blue-300 mb-2 flex items-center">
                        <AlertCircle className="w-4 h-4 mr-2" />
                        File Requirements
                      </h4>
                      <ul className="text-xs text-blue-200 space-y-1">
                        <li>• File format: PDF (.pdf)</li>
                        <li>• Recommended file size: Less than 50 MB</li>
                        <li>
                          • Filename format: "Book Title - Author Name.pdf"
                        </li>
                        <li>• System will auto-extract and chunk content</li>
                      </ul>
                    </div>

                    {/* Progress Section */}
                    {(uploadProgress || liveProgress || isIngesting) && (
                      <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                        <h3 className="text-lg font-bold text-white mb-4">
                          Ingestion Progress
                        </h3>

                        <div className="space-y-4">
                          {/* Circular Progress */}
                          <div className="flex items-center justify-center">
                            <div className="relative w-28 h-28">
                              <svg className="w-full h-full transform -rotate-90">
                                <circle
                                  cx="56"
                                  cy="56"
                                  r="48"
                                  stroke="rgba(255,255,255,0.1)"
                                  strokeWidth="6"
                                  fill="none"
                                />
                                <circle
                                  cx="56"
                                  cy="56"
                                  r={radius}
                                  stroke="url(#grad)"
                                  strokeWidth="6"
                                  fill="none"
                                  strokeDasharray={circumference}
                                  strokeDashoffset={
                                    circumference -
                                    (currentPercentage / 100) * circumference
                                  }
                                  strokeLinecap="round"
                                  className="transition-all duration-500 ease-out"
                                />

                                <defs>
                                  <linearGradient
                                    id="grad"
                                    x1="0%"
                                    y1="0%"
                                    x2="100%"
                                    y2="100%"
                                  >
                                    <stop offset="0%" stopColor="#a855f7" />
                                    <stop offset="100%" stopColor="#ec4899" />
                                  </linearGradient>
                                </defs>
                              </svg>
                              <div className="absolute inset-0 flex items-center justify-center">
                                <p className="text-2xl font-bold text-white">
                                  {Math.round(currentPercentage)}%
                                </p>
                              </div>
                            </div>
                          </div>

                          {/* Status */}
                          <p
                            className={`text-center text-sm font-semibold ${getStatusColor(
                              currentStatus
                            )}`}
                          >
                            {liveProgress?.status
                              ? `Processing: ${liveProgress.status}`
                              : uploadProgress?.message || "Ready"}
                          </p>

                          {/* Progress Details Grid */}
                          {liveProgress && (
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-white/5 rounded p-3 border border-white/10">
                                <p className="text-xs text-purple-300 mb-1">
                                  Pages
                                </p>
                                <p className="text-lg font-bold text-white">
                                  {liveProgress.current_page}/
                                  {liveProgress.total_pages}
                                </p>
                              </div>
                              <div className="bg-white/5 rounded p-3 border border-white/10">
                                <p className="text-xs text-purple-300 mb-1">
                                  Chunks
                                </p>
                                <p className="text-lg font-bold text-white">
                                  {liveProgress.chunks_created}
                                </p>
                              </div>
                            </div>
                          )}

                          {/* Success/Error Message */}
                          {uploadProgress?.status === "success" && (
                            <div className="p-4 rounded-lg bg-green-500/20 border border-green-400/50">
                              <div className="flex items-center space-x-3">
                                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                                <div>
                                  <p className="text-sm font-semibold text-green-300">
                                    Success!
                                  </p>
                                  <p className="text-xs text-green-200">
                                    {uploadProgress.result?.chunks} chunks
                                    created
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {uploadProgress?.status === "error" && (
                            <div className="p-4 rounded-lg bg-red-500/20 border border-red-400/50">
                              <div className="flex items-center space-x-3">
                                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                                <p className="text-sm font-semibold text-red-300">
                                  {uploadProgress.message}
                                </p>
                              </div>
                            </div>
                          )}

                          {/* Logs */}
                          {isIngesting && (
                            <div className="mt-4 pt-4 border-t border-white/10">
                              <button
                                onClick={() => setShowLogs(!showLogs)}
                                className="w-full flex items-center justify-center space-x-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all text-sm"
                              >
                                {showLogs ? (
                                  <EyeOff className="w-4 h-4" />
                                ) : (
                                  <Eye className="w-4 h-4" />
                                )}
                                <span className="font-medium">
                                  {showLogs ? "Hide" : "View"} Logs
                                </span>
                              </button>

                              {showLogs && (
                                <div className="mt-3 bg-black/40 rounded-lg border border-white/10 p-3 h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-purple-500 scrollbar-track-white/10 font-mono text-xs">
                                  {logs.length === 0 ? (
                                    <p className="text-purple-300">
                                      Waiting for logs...
                                    </p>
                                  ) : (
                                    logs.map((log, idx) => (
                                      <div
                                        key={idx}
                                        className={`${
                                          log.type === "success"
                                            ? "text-green-400"
                                            : log.type === "error"
                                            ? "text-red-400"
                                            : log.type === "warning"
                                            ? "text-yellow-400"
                                            : "text-purple-300"
                                        }`}
                                      >
                                        <span className="text-gray-500">
                                          [{log.timestamp}]
                                        </span>{" "}
                                        {log.message}
                                      </div>
                                    ))
                                  )}
                                  <div ref={logsEndRef} />
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Buttons */}
                    <div className="flex gap-3 pt-4">
                      <button
                        onClick={handleUploadSubmit}
                        disabled={!uploadFile || isIngesting}
                        className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold flex items-center justify-center space-x-2"
                      >
                        {isIngesting ? (
                          <>
                            <Loader className="w-5 h-5 animate-spin" />
                            <span>Processing...</span>
                          </>
                        ) : (
                          <>
                            <Upload className="w-5 h-5" />
                            <span>Upload & Ingest</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={handleReset}
                        disabled={isIngesting}
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
