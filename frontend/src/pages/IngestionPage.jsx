import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  ArrowLeft,
  AlertCircle,
  CheckCircle,
  Loader,
  Book,
  FileText,
  Eye,
  EyeOff,
} from "lucide-react";

const API_BASE_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/ingestion-progress";

export default function IngestionPage({ books, onBack, onUploadSuccess }) {
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logsEndRef.current && showLogs) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

  // Cleanup WebSocket on unmount
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

            // Log significant changes
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
            } else if (data.status === "completed") {
              addLog(
                `✅ Ingestion completed! ${data.chunks_created} chunks indexed`,
                "success"
              );
            } else if (data.status === "failed") {
              addLog(`❌ Ingestion failed: ${data.errors.join(", ")}`, "error");
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

    // Connect to WebSocket before uploading
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
      percentage: 0,
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

        // Call the callback to refresh books in parent component
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
    onBack();
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

  const getProgressBarColor = (status) => {
    switch (status) {
      case "completed":
        return "from-green-500 to-emerald-600";
      case "failed":
        return "from-red-500 to-pink-600";
      case "upserting":
      case "embedding":
      case "chunking":
        return "from-blue-500 to-cyan-600";
      default:
        return "from-purple-500 to-pink-600";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-lg border-b border-white/10">
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
      <div className="px-4 sm:px-6 lg:px-8 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Upload Form Card */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 mb-8">
            <h2 className="text-2xl font-bold text-white mb-2">
              Upload PDF Books
            </h2>
            <p className="text-purple-200 mb-6">
              Upload PDF files to ingest them into your knowledge base. The
              system will automatically extract text, create chunks, and index
              them for retrieval.
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
                    <p className="text-xs text-purple-300">PDF files only</p>
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
                    • Filename format: "Book Title - Author Name.pdf" (optional
                    but recommended)
                  </li>
                  <li>
                    • The system will automatically extract and chunk the
                    content
                  </li>
                </ul>
              </div>

              {/* Progress Section */}
              {(uploadProgress || liveProgress) && (
                <div className="space-y-4">
                  {/* Progress Bar */}
                  <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex-1">
                        <p className="text-sm font-semibold text-white mb-1">
                          Ingestion Progress
                        </p>
                        <p
                          className={`text-xs font-medium ${getStatusColor(
                            liveProgress?.status || uploadProgress?.status
                          )}`}
                        >
                          {liveProgress?.current_task ||
                            uploadProgress?.message}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-white">
                          {Math.round(
                            liveProgress?.percentage ||
                              uploadProgress?.percentage ||
                              0
                          )}
                          %
                        </p>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden border border-white/20">
                      <div
                        className={`h-full bg-gradient-to-r ${getProgressBarColor(
                          liveProgress?.status || uploadProgress?.status
                        )} transition-all duration-300`}
                        style={{
                          width: `${
                            liveProgress?.percentage ||
                            uploadProgress?.percentage ||
                            0
                          }%`,
                        }}
                      />
                    </div>

                    {/* Progress Details */}
                    {liveProgress && (
                      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                        <div className="bg-white/5 rounded p-2 border border-white/10">
                          <p className="text-purple-300">Pages</p>
                          <p className="text-white font-semibold">
                            {liveProgress.current_page}/
                            {liveProgress.total_pages}
                          </p>
                        </div>
                        <div className="bg-white/5 rounded p-2 border border-white/10">
                          <p className="text-purple-300">Chunks</p>
                          <p className="text-white font-semibold">
                            {liveProgress.chunks_created}
                          </p>
                        </div>
                        <div className="bg-white/5 rounded p-2 border border-white/10">
                          <p className="text-purple-300">Embeddings</p>
                          <p className="text-white font-semibold">
                            {liveProgress.embeddings_generated}
                          </p>
                        </div>
                        <div className="bg-white/5 rounded p-2 border border-white/10">
                          <p className="text-purple-300">Time</p>
                          <p className="text-white font-semibold">
                            {Math.round(liveProgress.elapsed_time)}s
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Speed Info */}
                    {liveProgress && liveProgress.speed_pages_per_sec > 0 && (
                      <div className="mt-3 text-xs text-purple-200">
                        <p>
                          Speed: {liveProgress.speed_pages_per_sec.toFixed(1)}{" "}
                          pages/sec
                        </p>
                        {liveProgress.estimated_time_remaining > 0 && (
                          <p>
                            ETA:{" "}
                            {Math.round(liveProgress.estimated_time_remaining)}s
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Message Display */}
                  {uploadProgress?.status === "success" && (
                    <div className="p-4 rounded-lg bg-green-500/20 border-green-500/50 border">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="w-5 h-5 text-green-400" />
                        <div>
                          <p className="font-semibold text-green-300">
                            {uploadProgress.message}
                          </p>
                          {uploadProgress.result && (
                            <div className="text-xs text-green-200 mt-1">
                              <p>
                                Total chunks: {uploadProgress.result.chunks}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {uploadProgress?.status === "error" && (
                    <div className="p-4 rounded-lg bg-red-500/20 border-red-500/50 border">
                      <div className="flex items-center space-x-3">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                        <p className="font-semibold text-red-300">
                          {uploadProgress.message}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Live Logs Toggle & Display */}
              {isIngesting && (
                <div className="space-y-2">
                  <button
                    onClick={() => setShowLogs(!showLogs)}
                    className="flex items-center space-x-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-all"
                  >
                    {showLogs ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                    <span className="text-sm font-medium">
                      {showLogs ? "Hide" : "View"} Live Logs
                    </span>
                  </button>

                  {showLogs && (
                    <div className="bg-black/40 rounded-lg border border-white/10 p-4 h-64 overflow-y-auto font-mono text-xs">
                      <div className="space-y-1">
                        {logs.length === 0 ? (
                          <p className="text-purple-300">Waiting for logs...</p>
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
                    </div>
                  )}
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

          {/* Books List */}
          {books.length > 0 && (
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6">
                Currently Ingested Books
              </h2>
              <div className="grid gap-4">
                {books.map((book, idx) => (
                  <div
                    key={idx}
                    className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-purple-400/50 transition-all"
                  >
                    <div className="flex items-start space-x-4">
                      <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-3 rounded-lg flex-shrink-0">
                        <Book className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white">
                          {book.title}
                        </h3>
                        <p className="text-sm text-purple-300">
                          by {book.author}
                        </p>
                        <div className="mt-2 flex items-center space-x-4 text-xs text-purple-200">
                          <span className="flex items-center space-x-1">
                            <FileText className="w-3 h-3" />
                            <span>{book.total_chunks} chunks</span>
                          </span>
                          <span className="text-purple-300">•</span>
                          <span>Ready for queries</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
