import React, { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@clerk/clerk-react";
import {
  Upload,
  ArrowLeft,
  AlertCircle,
  CheckCircle,
  Loader,
  Eye,
  EyeOff,
  FileText,
  X,
  Library,
  BookOpen,
  GraduationCap,
  Book,
} from "lucide-react";

// Determine the correct protocol (ws or wss) based on the current page
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// Dynamically set the URL to match the current domain/IP
const WS_URL = `${protocol}//${window.location.host}/api/ws/ingest`;

// For the API URL, we can use a relative path so Nginx proxies it
const API_BASE_URL = "/api";

// Extract LogItem to a memoized component for performance
const LogItem = React.memo(({ log, getLogColor }) => (
  <div
    className={`${getLogColor(
      log.type
    )} leading-relaxed flex items-start hover:bg-blue-50 px-3 py-1.5 rounded-md transition-all duration-200`}
  >
    <span className="text-gray-500 text-xs mr-3 flex-shrink-0 select-none font-semibold">
      [{log.timestamp}]
    </span>
    <span
      className={`mr-3 flex-shrink-0 font-bold text-xs select-none px-2 py-0.5 rounded ${log.type === "error"
        ? "bg-red-100 text-red-700 border border-red-300"
        : log.type === "warning"
          ? "bg-yellow-100 text-yellow-700 border border-yellow-300"
          : log.type === "success"
            ? "bg-green-100 text-green-700 border border-green-300"
            : "bg-blue-100 text-blue-700 border border-blue-300"
        }`}
    >
      {log.type === "error"
        ? "ERR"
        : log.type === "warning"
          ? "WRN"
          : log.type === "success"
            ? "OK"
            : "INF"}
    </span>
    <span className="flex-1 break-all text-gray-700">{log.message}</span>
  </div>
));

export default function IngestionPage({ books, onUploadSuccess }) {
  const { getToken } = useAuth();

  // --- QUEUE STATE MANAGEMENT ---
  const [fileQueue, setFileQueue] = useState([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(-1);
  const [completedCount, setCompletedCount] = useState(0);

  // --- EXISTING UI STATES ---
  const [uploadProgress, setUploadProgress] = useState(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Smooth animation state
  const [animatedPercentage, setAnimatedPercentage] = useState(0);
  const [searchMode, setSearchMode] = useState("all");

  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  const processedLogsRef = useRef(new Set());
  const finishSequenceStarted = useRef(false);
  const successTimerRef = useRef(null);
  const isIngestingRef = useRef(false);

  const processingStarted = useRef(false);
  const currentFileIdRef = useRef(null);

  const navigate = useNavigate();

  // Add animation styles
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = `
      .progress-circle {
        transition: stroke-dashoffset 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      }
      @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.4); }
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

  // ==================================================================================
  // QUEUE ORCHESTRATION
  // ==================================================================================

  useEffect(() => {
    if (currentFileIndex >= 0 && currentFileIndex < fileQueue.length) {
      const item = fileQueue[currentFileIndex];
      currentFileIdRef.current = item.id;

      setFileQueue((prev) =>
        prev.map((f, i) =>
          i === currentFileIndex ? { ...f, status: "processing" } : f
        )
      );
      uploadSingleFile(item);
    } else if (currentFileIndex >= fileQueue.length && currentFileIndex > 0) {
      finishQueue();
    }
  }, [currentFileIndex]);

  const startQueueProcessing = () => {
    if (fileQueue.length === 0) return;
    setIsIngesting(true);
    isIngestingRef.current = true;
    setCurrentFileIndex(0);
    setCompletedCount(0);
  };

  const proceedToNextFile = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, "File complete");
      wsRef.current = null;
    }
    setCompletedCount((prev) => prev + 1);
    setCurrentFileIndex((prev) => prev + 1);
  };

  const finishQueue = () => {
    setIsIngesting(false);
    isIngestingRef.current = false;
    setCurrentFileIndex(-1);
    currentFileIdRef.current = null;

    setUploadProgress({
      status: "success",
      message: `All ${fileQueue.length} files processed successfully!`,
      percentage: 100,
    });
    setLiveProgress({ status: "completed" });
    setAnimatedPercentage(100);

    addLog("🎉 All files in queue processed.", "success");

    if (onUploadSuccess) {
      console.log("📚 Refreshing book list...");
      onUploadSuccess();
    }
  };

  // ==================================================================================
  // CORE UPLOAD LOGIC (Single File)
  // ==================================================================================

  const uploadSingleFile = async (queueItem) => {
    setLiveProgress(null);
    setUploadProgress(null);
    setLogs([]);
    processedLogsRef.current.clear();

    finishSequenceStarted.current = false;
    successTimerRef.current = null;
    processingStarted.current = false;
    setAnimatedPercentage(0);

    addLog(`📋 Starting: ${queueItem.file.name}`, "info");

    const formData = new FormData();
    formData.append("file", queueItem.file);

    setUploadProgress({
      status: "uploading",
      message: `Uploading ${queueItem.file.name}...`,
      percentage: 5,
    });

    try {
      const token = await getToken();

      const requestPromise = fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const response = await requestPromise;
      const data = await response.json();

      if (currentFileIdRef.current !== queueItem.id) {
        console.warn(
          `🛑 Ignoring stale response for ${queueItem.file.name} (Current: ${currentFileIdRef.current})`
        );
        return;
      }

      if (data.success) {
        const taskId = data.result?.task_id;

        if (taskId) {
          addLog(`✅ Upload complete. Tracking Task ID: ${taskId}`, "success");

          setUploadProgress({
            status: "processing",
            message: "File uploaded. Starting processing...",
            percentage: 10,
          });

          if (isIngestingRef.current) {
            await connectWebSocket(taskId);
          }
        } else {
          addLog("⚠️ No task ID returned. Assuming immediate completion.", "warning");
          finishSequenceStarted.current = true;
          proceedToNextFile();
        }
      } else {
        throw new Error(data.error || "Unknown error");
      }
    } catch (error) {
      if (currentFileIdRef.current !== queueItem.id) return;

      addLog(`❌ Failed: ${queueItem.file.name} - ${error.message}`, "error");
      setFileQueue((prev) =>
        prev.map((f) => (f.id === queueItem.id ? { ...f, status: "error" } : f))
      );

      setTimeout(() => proceedToNextFile(), 2000);
    }
  };

  // ==================================================================================
  // WEBSOCKET & LOGIC
  // ==================================================================================

  const connectWebSocket = (taskId) => {
    return new Promise((resolve, reject) => {
      try {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.close();
        }

        const wsUrlWithId = `${WS_URL}/${taskId}`;

        wsRef.current = new WebSocket(wsUrlWithId);

        wsRef.current.onopen = () => {
          addLog("✅ Connected to progress stream", "success");
          resolve();
        };

        wsRef.current.onmessage = (event) => {
          if (!isIngestingRef.current) return;

          try {
            const data = JSON.parse(event.data);
            setLiveProgress(data);

            if (data.logs && Array.isArray(data.logs)) {
              data.logs.forEach((logLine) => {
                const match = logLine.match(
                  /\[(\d{2}:\d{2}:\d{2})\] (\w+): (.+)/
                );
                if (match) {
                  const [, timestamp, level, message] = match;
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

        wsRef.current.onclose = () => {
          console.log("WebSocket connection closed");
        };

      } catch (error) {
        reject(error);
      }
    });
  };

  // Monitoring Effect (Success Detection)
  useEffect(() => {
    if (!isIngesting) return;
    if (successTimerRef.current) return;

    const isActiveStatus = [
      "parsing_pdf",
      "chunking",
      "embedding",
      "upserting",
      "processing",
    ].includes(liveProgress?.status);

    if (!processingStarted.current && isActiveStatus) {
      processingStarted.current = true;
      setLogs((prevLogs) =>
        prevLogs.filter((l) => !l.message.includes("Ingestion completed"))
      );
    }

    const isFinishedInLogs = logs.some((l) =>
      l.message.includes("Ingestion completed successfully")
    );
    const isFinishedStatus =
      liveProgress?.status === "completed" ||
      liveProgress?.status === "success";

    if ((isFinishedInLogs || isFinishedStatus) && processingStarted.current) {
      finishSequenceStarted.current = true;
      if (animatedPercentage < 100) setAnimatedPercentage(100);

      successTimerRef.current = setTimeout(() => {
        if (!isIngestingRef.current) return;

        if (currentFileIndex >= 0 && currentFileIndex < fileQueue.length) {
          setFileQueue((prev) =>
            prev.map((f, i) =>
              i === currentFileIndex ? { ...f, status: "success" } : f
            )
          );
        }

        proceedToNextFile();
      }, 1500);
    }
  }, [logs, liveProgress, isIngesting, animatedPercentage, currentFileIndex]);

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

  // ==================================================================================
  // FILE HANDLING UTILS
  // ==================================================================================

  const handleFilesSelected = (files) => {
    const newFiles = Array.from(files)
      .filter((f) => f.type === "application/pdf" || f.name.endsWith(".pdf"))
      .map((f) => ({
        file: f,
        id: Math.random().toString(36).substr(2, 9),
        status: "pending",
      }));

    if (newFiles.length > 0) {
      setFileQueue((prev) => [...prev, ...newFiles]);
      addLog(`📎 Added ${newFiles.length} files to queue`, "info");
    }
  };

  const removeFile = (id) => {
    if (isIngesting) return;
    setFileQueue((prev) => prev.filter((f) => f.id !== id));
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isIngesting) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isIngesting) return;
    handleFilesSelected(e.dataTransfer.files);
  };

  const handleReset = () => {
    if (wsRef.current) {
      wsRef.current.close(1000, "User cancelled");
    }
    setFileQueue([]);
    setCurrentFileIndex(-1);
    setCompletedCount(0);
    setUploadProgress(null);
    setLiveProgress(null);
    setLogs([]);
    setShowLogs(false);
    setIsIngesting(false);
    isIngestingRef.current = false;
    currentFileIdRef.current = null;
    navigate("/");
  };

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

  const getStatusColor = (status) => {
    if (status === "completed" || status === "success") return "text-green-600";
    if (status === "failed" || status === "error") return "text-red-600";
    return "text-blue-600";
  };

  const getLogColor = (type) => {
    if (type === "success") return "text-green-700";
    if (type === "error") return "text-red-700";
    if (type === "warning") return "text-yellow-700";
    return "text-blue-700";
  };

  const calculatePercentage = () => {
    if (
      uploadProgress?.status === "success" &&
      (!liveProgress || liveProgress.status === "completed")
    )
      return 100;
    if (liveProgress?.status === "failed") return 0;
    if (!liveProgress) return uploadProgress?.percentage ?? 0;

    if (
      liveProgress.percentage !== undefined &&
      liveProgress.percentage !== null
    ) {
      return Math.min(Math.max(Number(liveProgress.percentage), 0), 100);
    }

    const status = liveProgress.status;
    if (status === "parsing_pdf") return 15;
    if (status === "chunking")
      return (
        20 +
        ((liveProgress.current_page || 0) / (liveProgress.total_pages || 1)) *
        30
      );
    if (status === "embedding")
      return (
        50 +
        ((liveProgress.embeddings_generated || 0) /
          (liveProgress.chunks_created || 1)) *
        30
      );
    if (status === "upserting")
      return (
        80 +
        ((liveProgress.vectors_upserted || 0) /
          (liveProgress.chunks_created || 1)) *
        18
      );

    return Math.max(uploadProgress?.percentage ?? 0, 5);
  };

  const currentPercentage = Math.round(animatedPercentage);
  const currentStatus =
    liveProgress?.status || uploadProgress?.status || "idle";
  const radius = 48;
  const circumference = 2 * Math.PI * radius;

  const renderedLogs = useMemo(
    () =>
      logs.map((log, idx) => (
        <LogItem key={idx} log={log} getLogColor={getLogColor} />
      )),
    [logs]
  );

  const { bookList, paperList } = useMemo(() => {
    const b = [];
    const p = [];
    books.forEach((item) => {
      const isPaper =
        item.total_chunks < 100 ||
        (item.title && item.title.toLowerCase().includes("paper"));
      if (isPaper) {
        p.push(item);
      } else {
        b.push(item);
      }
    });
    return { bookList: b, paperList: p };
  }, [books]);

  const filteredBooks = useMemo(() => {
    if (!Array.isArray(books)) return [];
    if (searchMode === "all") return books;
    if (searchMode === "books") return bookList;
    if (searchMode === "papers") return paperList;
    return books;
  }, [books, searchMode, bookList, paperList]);

  return (
    <div className="h-screen bg-gradient-to-br from-stone-50 via-stone-100 to-amber-50/30 text-stone-800 relative flex flex-col">
      {/* Subtle Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-emerald-200/15 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-20 w-96 h-96 bg-amber-200/15 rounded-full blur-3xl" />
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(16,185,129,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(16,185,129,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      <div className="relative z-10 flex flex-col flex-1 w-full">
        {/* Header */}
        <header className="bg-white border-b border-stone-200 shadow-sm sticky top-0 z-10 flex-shrink-0">
          <div className="px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleReset}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-all text-gray-700 disabled:opacity-50"
                  disabled={isIngesting}
                >
                  <ArrowLeft className="w-5 h-5" />
                </button>
                <div className="bg-gradient-to-br from-emerald-500 to-teal-600 p-2 rounded-lg shadow-md shadow-emerald-500/20">
                  <Upload className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    Ingest Documents
                  </h1>
                  <p className="text-sm text-gray-600">
                    Add Books or Research Papers to your knowledge base
                  </p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* LEFT SIDEBAR */}
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white rounded-xl p-6 border border-stone-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Library className="w-5 h-5 mr-2" />
                    Library
                  </h3>
                  <div className="space-y-2">
                    {/* Search Scope Toggle */}
                    <div className="flex justify-center mb-4">
                      <div className="bg-stone-100 p-1 rounded-lg flex space-x-1">
                        <button
                          onClick={() => setSearchMode("all")}
                          className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${searchMode === "all"
                            ? "bg-emerald-600 text-white shadow-md"
                            : "text-stone-600 hover:bg-stone-200"
                            }`}
                        >
                          All Sources
                        </button>
                        <button
                          onClick={() => setSearchMode("books")}
                          className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${searchMode === "books"
                            ? "bg-emerald-600 text-white shadow-md"
                            : "text-stone-600 hover:bg-stone-200"
                            }`}
                        >
                          <BookOpen className="w-3 h-3 mr-1" /> Books
                        </button>
                        <button
                          onClick={() => setSearchMode("papers")}
                          className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${searchMode === "papers"
                            ? "bg-teal-600 text-white shadow-md"
                            : "text-stone-600 hover:bg-stone-200"
                            }`}
                        >
                          <GraduationCap className="w-3 h-3 mr-1" /> Papers
                        </button>
                      </div>
                    </div>

                    {/* Scrollable library list */}
                    <div className="max-h-96 overflow-y-auto space-y-2">
                      {/* RESEARCH PAPERS SECTION */}
                      {(searchMode === "all" || searchMode === "papers") &&
                        paperList.length > 0 && (
                          <div>
                            {searchMode === "all" && (
                              <div className="text-xs font-bold text-teal-600 uppercase tracking-wider mb-2 flex items-center">
                                <GraduationCap className="w-3 h-3 mr-1" />
                                Research Papers
                              </div>
                            )}
                            <div className="space-y-1">
                              {paperList.map((book, idx) => (
                                <button
                                  key={idx}
                                  className="w-full text-left px-3 py-2 rounded-lg transition-all bg-white border border-stone-200 text-stone-700 hover:bg-stone-50 hover:border-teal-300"
                                >
                                  <div className="flex items-center space-x-2">
                                    <FileText className="w-3 h-3 flex-shrink-0 text-teal-600" />
                                    <div className="truncate flex-1">
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
                                    </div>
                                  </div>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                      {/* BOOKS SECTION */}
                      {(searchMode === "all" || searchMode === "books") &&
                        bookList.length > 0 && (
                          <div>
                            {searchMode === "all" && (
                              <div className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-2 mt-4 flex items-center">
                                <Book className="w-3 h-3 mr-1" />
                                Books
                              </div>
                            )}
                            <div className="space-y-1">
                              {bookList.map((book, idx) => (
                                <button
                                  key={idx}
                                  className="w-full text-left px-3 py-2 rounded-lg transition-all bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 hover:border-blue-300"
                                >
                                  <div className="flex items-center space-x-2">
                                    <Book className="w-3 h-3 flex-shrink-0 text-blue-600" />
                                    <div className="truncate flex-1">
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
                                    </div>
                                  </div>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                      {books.length === 0 && (
                        <div className="text-center text-gray-500 text-sm py-8 opacity-70">
                          No documents found. <br /> Upload some!
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Stats */}
                <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Stats
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 text-sm">
                        Total Docs:
                      </span>
                      <span className="font-bold text-gray-900 text-lg">
                        {filteredBooks.length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 text-sm">
                        Total Chunks:
                      </span>
                      <span className="font-bold text-gray-900 text-lg">
                        {filteredBooks.reduce(
                          (sum, b) => sum + (b.total_chunks || 0),
                          0
                        )}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* RIGHT SIDE - Upload Form */}
              <div className="lg:col-span-3 space-y-6">
                <div className="bg-white rounded-xl p-8 border border-stone-200 shadow-sm">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Upload PDFs
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Select books or research papers. The system will
                    auto-detect the document type.
                  </p>

                  <div className="space-y-6">
                    {/* File Input Zone */}
                    {!isIngesting && (
                      <>
                        <div>
                          <label className="block text-sm font-semibold text-gray-700 mb-3">
                            Select Files
                          </label>
                          <label
                            htmlFor="file-upload"
                            className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${isDragging
                              ? "border-emerald-500 bg-emerald-50 scale-[1.01] shadow-lg"
                              : "border-stone-300 bg-stone-50 hover:bg-emerald-50 hover:border-emerald-400"
                              }`}
                            onDragEnter={handleDragEnter}
                            onDragLeave={handleDragLeave}
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                          >
                            <div className="flex flex-col items-center justify-center pt-8 pb-6">
                              <Upload
                                className={`w-12 h-12 mb-2 transition-all duration-300 ${isDragging
                                  ? "text-emerald-600 scale-125"
                                  : "text-stone-400"
                                  }`}
                              />
                              <p className="text-sm font-semibold text-gray-900">
                                {isDragging
                                  ? "🚀 Drop files here"
                                  : "Click to upload or drag and drop"}
                              </p>
                              <p className="text-xs text-gray-500 mt-1">
                                Multiple PDFs supported
                              </p>
                            </div>
                            <input
                              id="file-upload"
                              type="file"
                              accept=".pdf"
                              multiple
                              onClick={(e) => (e.target.value = null)}
                              onChange={(e) =>
                                handleFilesSelected(e.target.files)
                              }
                              className="hidden"
                              disabled={isIngesting}
                            />
                          </label>

                          {/* File Queue List */}
                          {fileQueue.length > 0 && (
                            <div className="mt-4">
                              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                                Selected Files ({fileQueue.length})
                              </h4>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {fileQueue.map((item) => (
                                  <div
                                    key={item.id}
                                    className="p-3 bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-between group hover:bg-gray-100 transition-all"
                                  >
                                    <div className="flex items-center space-x-3 truncate">
                                      <FileText className="w-5 h-5 text-blue-600 flex-shrink-0" />
                                      <div className="truncate">
                                        <p className="text-sm font-medium text-gray-900 truncate">
                                          {item.file.name}
                                        </p>
                                        <p className="text-xs text-gray-600">
                                          {(
                                            item.file.size /
                                            1024 /
                                            1024
                                          ).toFixed(2)}{" "}
                                          MB
                                        </p>
                                      </div>
                                    </div>
                                    <button
                                      onClick={() => removeFile(item.id)}
                                      className="p-1.5 hover:bg-red-100 text-gray-400 hover:text-red-600 rounded-lg transition-colors"
                                    >
                                      <X className="w-4 h-4" />
                                    </button>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </>
                    )}

                    {/* Progress Section */}
                    {(isIngesting || completedCount > 0) && (
                      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200 shadow-md">
                        <div className="flex items-center justify-between mb-6">
                          <h3 className="text-xl font-bold text-gray-900 flex items-center">
                            <div
                              className={`w-2 h-2 rounded-full mr-3 ${currentPercentage === 100 && !isIngesting
                                ? "bg-green-500"
                                : "bg-green-500 animate-pulse"
                                }`}
                            ></div>
                            {isIngesting
                              ? `Processing File ${currentFileIndex + 1} of ${fileQueue.length
                              }`
                              : "Ingestion Complete"}
                          </h3>
                          <span className="text-xs font-mono bg-white px-2 py-1 rounded text-gray-700 border border-gray-300">
                            {completedCount} / {fileQueue.length} Done
                          </span>
                        </div>

                        <div className="space-y-6">
                          {/* Queue Visualizer */}
                          <div className="flex space-x-2 overflow-x-auto pb-2">
                            {fileQueue.map((item, idx) => (
                              <div
                                key={item.id}
                                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold border transition-all ${idx === currentFileIndex
                                  ? "bg-blue-600 border-blue-500 text-white animate-pulse shadow-md"
                                  : item.status === "success"
                                    ? "bg-green-100 border-green-500 text-green-700"
                                    : item.status === "error"
                                      ? "bg-red-100 border-red-500 text-red-700"
                                      : "bg-gray-100 border-gray-300 text-gray-500"
                                  }`}
                              >
                                {item.status === "success" ? (
                                  <CheckCircle className="w-5 h-5" />
                                ) : item.status === "error" ? (
                                  <AlertCircle className="w-5 h-5" />
                                ) : (
                                  idx + 1
                                )}
                              </div>
                            ))}
                          </div>

                          {/* Circular Progress (Current File) */}
                          <div className="flex items-center justify-center py-4">
                            <div className="relative w-32 h-32">
                              <svg
                                className="transform -rotate-90"
                                width="128"
                                height="128"
                              >
                                <circle
                                  cx="64"
                                  cy="64"
                                  r={radius}
                                  stroke="#e5e7eb"
                                  strokeWidth="8"
                                  fill="none"
                                />
                                <circle
                                  cx="64"
                                  cy="64"
                                  r={radius}
                                  stroke="url(#gradient)"
                                  strokeWidth="8"
                                  fill="none"
                                  strokeDasharray={circumference}
                                  strokeDashoffset={
                                    circumference -
                                    (currentPercentage / 100) * circumference
                                  }
                                  strokeLinecap="round"
                                  className="progress-circle"
                                />
                                <defs>
                                  <linearGradient
                                    id="gradient"
                                    x1="0%"
                                    y1="0%"
                                    x2="100%"
                                    y2="100%"
                                  >
                                    <stop
                                      offset="0%"
                                      stopColor="#10b981"
                                    />
                                    <stop
                                      offset="100%"
                                      stopColor="#0d9488"
                                    />
                                  </linearGradient>
                                </defs>
                              </svg>
                              <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-3xl font-bold text-gray-900">
                                  {currentPercentage}%
                                </span>
                                <span className="text-xs text-gray-600 mt-1 capitalize">
                                  {currentStatus}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Status Message */}
                          <div className="text-center">
                            <p className={`text-sm font-medium ${getStatusColor(currentStatus)}`}>
                              {liveProgress?.message || uploadProgress?.message || "Initializing..."}
                            </p>
                          </div>

                          {/* Detailed Progress Info */}
                          {liveProgress && (
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              {liveProgress.total_pages && (
                                <div className="bg-white p-3 rounded-lg border border-gray-200">
                                  <div className="text-gray-600 text-xs">Pages</div>
                                  <div className="text-gray-900 font-semibold">
                                    {liveProgress.current_page || 0} / {liveProgress.total_pages}
                                  </div>
                                </div>
                              )}
                              {liveProgress.chunks_created && (
                                <div className="bg-white p-3 rounded-lg border border-gray-200">
                                  <div className="text-gray-600 text-xs">Chunks</div>
                                  <div className="text-gray-900 font-semibold">
                                    {liveProgress.chunks_created}
                                  </div>
                                </div>
                              )}
                              {liveProgress.embeddings_generated && (
                                <div className="bg-white p-3 rounded-lg border border-gray-200">
                                  <div className="text-gray-600 text-xs">Embeddings</div>
                                  <div className="text-gray-900 font-semibold">
                                    {liveProgress.embeddings_generated}
                                  </div>
                                </div>
                              )}
                              {liveProgress.vectors_upserted && (
                                <div className="bg-white p-3 rounded-lg border border-gray-200">
                                  <div className="text-gray-600 text-xs">Vectors</div>
                                  <div className="text-gray-900 font-semibold">
                                    {liveProgress.vectors_upserted}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex space-x-3">
                      {!isIngesting && fileQueue.length > 0 && (
                        <button
                          onClick={startQueueProcessing}
                          className="flex-1 px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg hover:from-emerald-600 hover:to-teal-700 transition-all font-semibold shadow-md flex items-center justify-center space-x-2"
                        >
                          <Upload className="w-5 h-5" />
                          <span>Start Processing ({fileQueue.length} files)</span>
                        </button>
                      )}
                      {!isIngesting && fileQueue.length > 0 && (
                        <button
                          onClick={() => setFileQueue([])}
                          className="px-6 py-3 bg-stone-200 text-stone-700 rounded-lg hover:bg-stone-300 transition-all font-semibold"
                        >
                          Clear Queue
                        </button>
                      )}
                    </div>

                    {/* Logs Section */}
                    {logs.length > 0 && (
                      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
                        <div className="flex items-center justify-between p-4 border-b border-gray-200">
                          <h4 className="text-sm font-semibold text-gray-900 flex items-center">
                            <FileText className="w-4 h-4 mr-2" />
                            Processing Logs ({logs.length})
                          </h4>
                          <button
                            onClick={() => setShowLogs(!showLogs)}
                            className="text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center space-x-1"
                          >
                            {showLogs ? (
                              <>
                                <EyeOff className="w-4 h-4" />
                                <span>Hide</span>
                              </>
                            ) : (
                              <>
                                <Eye className="w-4 h-4" />
                                <span>Show</span>
                              </>
                            )}
                          </button>
                        </div>
                        {showLogs && (
                          <div className="p-4 max-h-80 overflow-y-auto bg-gray-50 font-mono text-xs space-y-1 terminal-glow">
                            {renderedLogs}
                            <div ref={logsEndRef} />
                          </div>
                        )}
                      </div>
                    )}
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
