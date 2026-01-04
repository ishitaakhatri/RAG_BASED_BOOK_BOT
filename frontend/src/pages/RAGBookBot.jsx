import React, { useState, useRef, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import "tailwindcss/tailwind.css";
import jsPDF from "jspdf";
import { useAuth } from "@clerk/clerk-react";

import {
  Search,
  BookOpen,
  Upload,
  Settings,
  MessageSquare,
  FileText,
  Code,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Loader,
  Book,
  ChevronDown,
  ChevronUp,
  Filter,
  Layers,
  Repeat,
  Plus,
  Trash2,
  Clock,
  MessageCircle,
  History,
  X,
  Copy,
  Download,
  ToggleLeft,
  ToggleRight,
  Library,
  GraduationCap,
  Edit,
  LogOut,
  Pause,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = "http://localhost:8000";

const BACKEND_DEFAULTS = {
  pass1K: 50,
  pass2K: 15,
  pass3Enabled: true,
  maxTokens: 30000,
};

export default function RAGBookBot() {
  // CLERK AUTH HOOK
  const { getToken, signOut } = useAuth();

  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState("all");
  const [showSettings, setShowSettings] = useState(false);
  const [currentLoadingStage, setCurrentLoadingStage] = useState(0);
  const messagesEndRef = useRef(null);

  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editingText, setEditingText] = useState("");

  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [showSessions, setShowSessions] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const [searchMode, setSearchMode] = useState("all");

  const [useBackendDefaults, setUseBackendDefaults] = useState(true);
  const [pass1K, setPass1K] = useState(BACKEND_DEFAULTS.pass1K);
  const [pass2K, setPass2K] = useState(BACKEND_DEFAULTS.pass2K);
  const [pass3Enabled, setPass3Enabled] = useState(
    BACKEND_DEFAULTS.pass3Enabled
  );
  const [maxTokens, setMaxTokens] = useState(BACKEND_DEFAULTS.maxTokens);

  const abortControllerRef = useRef(null);
  const navigate = useNavigate();

  const loadingStages = [
    "Processing your request…",
    "Preparing your response…",
    "Analyzing… Please wait",
    "Working on it…",
    "Almost ready…",
    "Fetching results…",
    "Please hold on, just a moment…",
  ];

  // Helper to make authenticated requests
  const authFetch = async (endpoint, options = {}) => {
    const token = await getToken();
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    };
    return fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });
  };

  useEffect(() => {
    if (!loading) {
      setCurrentLoadingStage(0);
      return;
    }
    const interval = setInterval(() => {
      setCurrentLoadingStage((prev) => (prev + 1) % loadingStages.length);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

  useEffect(() => {
    fetchBooks();
    fetchSessions();
  }, []);

  const shouldAutoScrollRef = useRef(true);

  useEffect(() => {
    if (shouldAutoScrollRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleUserScroll = (e) => {
    const el = e.target;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    shouldAutoScrollRef.current = isAtBottom;
  };

  const { bookList, paperList } = useMemo(() => {
    const b = [];
    const p = [];
    books.forEach((item) => {
      const isPaper =
        item.total_chunks < 100 ||
        (item.title && item.title.toLowerCase().includes("paper"));
      if (isPaper) p.push(item);
      else b.push(item);
    });
    return { bookList: b, paperList: p };
  }, [books]);

  const downloadChat = () => {
    if (messages.length === 0) {
      alert("No messages to download!");
      return;
    }
    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const maxWidth = pageWidth - 2 * margin;
    let yPosition = 20;

    pdf.setFillColor(139, 92, 246);
    pdf.rect(0, 0, pageWidth, 30, "F");
    pdf.setTextColor(255, 255, 255);
    pdf.setFontSize(20);
    pdf.setFont("helvetica", "bold");
    pdf.text("RAG Bot - Chat History", margin, 20);

    yPosition = 40;
    pdf.setTextColor(100, 100, 100);
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "normal");
    pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition);
    yPosition += 5;
    pdf.text(
      `Session ID: ${currentSessionId || "New Session"}`,
      margin,
      yPosition
    );
    yPosition += 10;
    pdf.setDrawColor(200, 200, 200);
    pdf.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;

    messages.forEach((msg, idx) => {
      if (yPosition > pageHeight - 50) {
        pdf.addPage();
        yPosition = 20;
      }
      if (msg.role === "user") {
        pdf.setFillColor(139, 92, 246);
        pdf.roundedRect(margin, yPosition - 5, maxWidth, 8, 2, 2, "F");
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(11);
        pdf.setFont("helvetica", "bold");
        pdf.text(`Question ${Math.floor(idx / 2) + 1}`, margin + 3, yPosition);
        yPosition += 10;
        pdf.setTextColor(0, 0, 0);
        pdf.setFontSize(10);
        pdf.setFont("helvetica", "normal");
        const userLines = pdf.splitTextToSize(msg.content, maxWidth - 6);
        userLines.forEach((line) => {
          if (yPosition > pageHeight - 20) {
            pdf.addPage();
            yPosition = 20;
          }
          pdf.text(line, margin + 3, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      } else {
        pdf.setFillColor(34, 197, 94);
        pdf.roundedRect(margin, yPosition - 5, maxWidth, 8, 2, 2, "F");
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(11);
        pdf.setFont("helvetica", "bold");
        pdf.text("Answer", margin + 3, yPosition);
        yPosition += 10;
        pdf.setTextColor(0, 0, 0);
        pdf.setFontSize(10);
        pdf.setFont("helvetica", "normal");
        let cleanContent = msg.content
          .replace(/\*\*(.*?)\*\*/g, "$1")
          .replace(/\*(.*?)\*/g, "$1")
          .replace(/`(.*?)`/g, "$1")
          .replace(/#{1,6}\s/g, "")
          .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
        const assistantLines = pdf.splitTextToSize(cleanContent, maxWidth - 6);
        assistantLines.forEach((line) => {
          if (yPosition > pageHeight - 20) {
            pdf.addPage();
            yPosition = 20;
          }
          pdf.text(line, margin + 3, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      }
    });
    pdf.save(`chat_${currentSessionId || Date.now()}.pdf`);
  };

  const fetchBooks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/books`);
      const data = await response.json();
      setBooks(data.books || []);
    } catch (error) {
      console.error("Failed to fetch books:", error);
    }
  };

  const fetchSessions = async () => {
    try {
      // Use authFetch
      const response = await authFetch(`/sessions?limit=50`);
      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      // Use authFetch
      const response = await authFetch(`/conversation/${sessionId}`);
      const data = await response.json();
      const loadedMessages = [];
      for (const turn of data.turns) {
        loadedMessages.push({ role: "user", content: turn.user_query });
        loadedMessages.push({
          role: "assistant",
          content: turn.assistant_response,
          sources: [],
          stats: turn.stats || {},
          pipeline_stages: turn.pipeline_stages || [],
          rewritten_queries: turn.rewritten_queries || [],
          answered_from_history: turn.answered_from_history || false,
          resolved_query: turn.resolved_query || null,
        });
      }
      setMessages(loadedMessages);
      setCurrentSessionId(sessionId);
      setSearchResults([]);
    } catch (error) {
      console.error("Failed to load session:", error);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCurrentSessionId(null);
    setSearchResults([]);
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this conversation?")) return;
    try {
      // Use authFetch with DELETE
      const response = await authFetch(`/conversation/${sessionId}`, {
        method: "DELETE",
      });
      if (response.ok) {
        await fetchSessions();
        if (sessionId === currentSessionId) startNewChat();
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
    }
  };

  const searchSessions = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    try {
      // Use authFetch
      const response = await authFetch(
        `/search/sessions?query=${encodeURIComponent(searchQuery)}&limit=10`
      );
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleInterruptQuery = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (currentSessionId) {
      try {
        // Use authFetch
        await authFetch(
          `/cancel-query?session_id=${encodeURIComponent(currentSessionId)}`,
          { method: "POST" }
        );
      } catch (error) {
        console.error("Error notifying backend:", error);
      }
    }
    setLoading(false);
    setQuery("");
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    abortControllerRef.current = new AbortController();

    const requestPayload = {
      query,
      session_id: currentSessionId,
      book_filter: selectedBook === "all" ? null : selectedBook,
      search_mode: searchMode,
      top_k: 5,
      ...(useBackendDefaults
        ? {}
        : {
            pass1_k: pass1K,
            pass2_k: pass2K,
            pass3_enabled: pass3Enabled,
            max_tokens: maxTokens,
          }),
    };

    try {
      const token = await getToken(); // Get token explicitly for AbortSignal support
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestPayload),
        signal: abortControllerRef.current.signal,
      });

      const data = await response.json();
      if (data.error || data.detail) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${data.error || data.detail}`,
            error: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources,
            stats: data.stats,
            confidence: data.confidence,
            pipeline_stages: data.pipeline_stages,
            rewritten_queries: data.rewritten_queries || [],
            answered_from_history: data.answered_from_history,
            resolved_query: data.resolved_query,
          },
        ]);
        if (!currentSessionId && data.session_id)
          setCurrentSessionId(data.session_id);
        await fetchSessions();
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Failed to get response: ${error.message}`,
            error: true,
          },
        ]);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleEditQuery = (index, currentText) => {
    setEditingMessageIndex(index);
    setEditingText(currentText);
  };
  const handleCancelEdit = () => {
    setEditingMessageIndex(null);
    setEditingText("");
  };

  const handleSubmitEdit = async (index) => {
    if (!editingText.trim() || loading) return;
    const updatedMessages = [...messages];
    updatedMessages[index].content = editingText;
    setMessages(updatedMessages);
    if (
      index + 1 < messages.length &&
      messages[index + 1].role === "assistant"
    ) {
      updatedMessages.splice(index + 1, 1);
      setMessages(updatedMessages);
    }
    setEditingMessageIndex(null);
    setEditingText("");
    setLoading(true);

    abortControllerRef.current = new AbortController();
    const requestPayload = {
      query: editingText,
      session_id: currentSessionId,
      book_filter: selectedBook === "all" ? null : selectedBook,
      search_mode: searchMode,
      top_k: 5,
      ...(useBackendDefaults
        ? {}
        : {
            pass1_k: pass1K,
            pass2_k: pass2K,
            pass3_enabled: pass3Enabled,
            max_tokens: maxTokens,
          }),
    };

    try {
      const token = await getToken();
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestPayload),
        signal: abortControllerRef.current.signal,
      });
      const data = await response.json();
      if (data.error || data.detail) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${data.error || data.detail}`,
            error: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources,
            stats: data.stats,
            confidence: data.confidence,
            pipeline_stages: data.pipeline_stages,
            rewritten_queries: data.rewritten_queries || [],
            answered_from_history: data.answered_from_history,
            resolved_query: data.resolved_query,
          },
        ]);
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Failed to get response: ${error.message}`,
            error: true,
          },
        ]);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  const renderDocButton = (doc, icon) => (
    <button
      key={doc.title}
      onClick={() => setSelectedBook(doc.title)}
      className={`w-full text-left px-3 py-2 rounded-lg transition-all truncate group ${
        selectedBook === doc.title
          ? "bg-purple-600 text-white"
          : "bg-white/5 text-purple-200 hover:bg-white/10"
      }`}
      title={`${doc.title} by ${doc.author}`}
    >
      <div className="flex items-center space-x-2">
        {icon}
        <span className="text-sm font-semibold truncate">{doc.title}</span>
      </div>
      <div className="text-xs opacity-75 pl-5">by {doc.author}</div>
    </button>
  );

  return (
    <div className="h-screen bg-slate-950 text-white relative overflow-hidden flex">
      {/* Animated Background Elements - Matching Landing Page */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-pink-600/20 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-3xl" />
      </div>
      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px]" />
      <div className="relative z-10 flex flex-1 w-full">
      {/* Sidebar */}
      <div
        className={`${
          showSessions ? "w-80" : "w-0"
        } transition-all duration-300 bg-black/30 backdrop-blur-lg border-r border-white/10 overflow-hidden flex flex-col`}
      >
        <div className="p-4 border-b border-white/10">
          <button
            onClick={startNewChat}
            className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all"
          >
            <Plus className="w-5 h-5" />
            <span className="font-semibold">New Chat</span>
          </button>
        </div>
        <div className="p-4 border-b border-white/10">
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && searchSessions()}
              placeholder="Search conversations..."
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-purple-300 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <Search className="absolute left-3 top-2.5 w-4 h-4 text-purple-300" />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setSearchResults([]);
                }}
                className="absolute right-3 top-2.5 text-purple-300 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          {isSearching && (
            <div className="mt-2 text-xs text-purple-300 flex items-center">
              <Loader className="w-3 h-3 animate-spin mr-2" />
              Searching...
            </div>
          )}
        </div>
        <div
          className="flex-1 overflow-y-scroll p-4 space-y-2 custom-scrollbar"
          style={{ maxHeight: "980px" }}
        >
          {searchResults.length > 0 ? (
            <>
              <div className="text-xs text-purple-300 mb-2">
                {searchResults.length} results for "{searchQuery}"
              </div>
              {searchResults.map((result, idx) => (
                <div
                  key={idx}
                  onClick={() => loadSession(result.session_id)}
                  className="bg-white/5 hover:bg-white/10 rounded-lg p-3 cursor-pointer transition-all border border-white/10 hover:border-purple-400/50"
                >
                  <div className="text-sm text-white font-medium mb-1 truncate">
                    {result.user_query}
                  </div>
                  <div className="text-xs text-purple-200 mb-2 line-clamp-2">
                    {result.assistant_response}
                  </div>
                  <div className="flex items-center justify-between text-xs text-purple-300">
                    <span className="flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatTimestamp(result.timestamp)}
                    </span>
                    <span className="text-green-400">
                      {(result.relevance_score * 100).toFixed(0)}% match
                    </span>
                  </div>
                </div>
              ))}
            </>
          ) : (
            <>
              {currentSessionId && (
                <div className="mb-2 text-xs text-purple-300 font-semibold">
                  CURRENT CHAT
                </div>
              )}
              {sessions.map((session, idx) => {
                const isCurrent = session.session_id === currentSessionId;
                return (
                  <div
                    key={idx}
                    onClick={() => loadSession(session.session_id)}
                    className={`rounded-lg p-3 cursor-pointer transition-all border ${
                      isCurrent
                        ? "bg-purple-600/30 border-purple-400"
                        : "bg-white/5 hover:bg-white/10 border-white/10 hover:border-purple-400/50"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white font-medium mb-1 truncate">
                          {session.title}
                        </div>
                        <div className="text-xs text-purple-200 truncate mb-2">
                          {session.last_message}
                        </div>
                        <div className="flex items-center space-x-3 text-xs text-purple-300">
                          <span className="flex items-center">
                            <MessageCircle className="w-3 h-3 mr-1" />
                            {session.message_count}
                          </span>
                          <span className="flex items-center">
                            <Clock className="w-3 h-3 mr-1" />
                            {formatTimestamp(session.updated_at)}
                          </span>
                        </div>
                      </div>
                      <button
                        onClick={(e) => deleteSession(session.session_id, e)}
                        className="ml-2 p-1 hover:bg-red-500/20 rounded text-red-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full min-h-0">
        <header className="bg-black/20 backdrop-blur-lg border-b border-white/10">
          <div className="px-4 sm:px-6 lg:px-8 py-4">
            <div className="relative flex items-center">
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setShowSessions(!showSessions)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-all text-white"
                >
                  <History className="w-5 h-5" />
                </button>
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                  <Library className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    RAG Knowledge Bot
                  </h1>
                  <p className="text-sm text-purple-200">
                    {currentSessionId
                      ? "Conversation with Memory"
                      : "Start New Conversation"}
                  </p>
                </div>
              </div>
              <div className="absolute right-16 top-1/2 transform -translate-y-1/2 flex items-center space-x-3 flex-nowrap">
                <button
                  onClick={downloadChat}
                  disabled={messages.length === 0}
                  className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  title="Download chat history"
                >
                  <Download className="w-4 h-4" />
                  <span className="hidden sm:inline">Download</span>
                </button>
                <button
                  onClick={() => navigate("/ingest")}
                  className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all"
                >
                  <Upload className="w-4 h-4" />
                  <span className="hidden sm:inline">Upload Doc</span>
                </button>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="flex-shrink-0 p-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all"
                >
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </header>
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full min-h-0">
          <div className="lg:col-span-1 space-y-4 overflow-hidden flex flex-col">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 flex-1 flex flex-col min-h-0">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center flex-shrink-0">
                <Library className="w-5 h-5 mr-2" />
                Library{" "}
                <span className="ml-auto text-xs font-normal text-purple-300 bg-black/30 px-2 py-1 rounded">
                  {searchMode === "all"
                    ? "All"
                    : searchMode === "books"
                    ? "Books"
                    : "Papers"}
                </span>
              </h3>
              <div className="space-y-2 flex-1 flex flex-col min-h-0">
                <div className="flex justify-center">
                  <div className="bg-black/30 p-1 rounded-lg flex space-x-1">
                    <button
                      onClick={() => setSearchMode("all")}
                      className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${
                        searchMode === "all"
                          ? "bg-purple-600 text-white shadow-lg"
                          : "text-purple-300 hover:bg-white/5"
                      }`}
                    >
                      All Sources
                    </button>
                    <button
                      onClick={() => setSearchMode("books")}
                      className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${
                        searchMode === "books"
                          ? "bg-blue-600 text-white shadow-lg"
                          : "text-purple-300 hover:bg-white/5"
                      }`}
                    >
                      <BookOpen className="w-3 h-3 mr-1" /> Books
                    </button>
                    <button
                      onClick={() => setSearchMode("papers")}
                      className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${
                        searchMode === "papers"
                          ? "bg-green-600 text-white shadow-lg"
                          : "text-purple-300 hover:bg-white/5"
                      }`}
                    >
                      <GraduationCap className="w-3 h-3 mr-1" /> Papers
                    </button>
                  </div>
                </div>
                <div className="overflow-y-auto pr-2 space-y-3 custom-scrollbar flex-1">
                  {(searchMode === "all" || searchMode === "papers") &&
                    paperList.length > 0 && (
                      <div className="animate-fade-in">
                        {searchMode === "all" && (
                          <div className="text-xs font-bold text-green-400 uppercase tracking-wider mb-1 mt-2 flex items-center">
                            <GraduationCap className="w-3 h-3 mr-1" /> Research
                            Papers
                          </div>
                        )}
                        <div className="space-y-1">
                          {paperList.map((doc, idx) =>
                            renderDocButton(
                              doc,
                              <FileText className="w-3 h-3 flex-shrink-0 text-green-300" />
                            )
                          )}
                        </div>
                      </div>
                    )}
                  {(searchMode === "all" || searchMode === "books") &&
                    bookList.length > 0 && (
                      <div className="animate-fade-in">
                        {searchMode === "all" && (
                          <div className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-1 mt-4 flex items-center">
                            <Book className="w-3 h-3 mr-1" /> Books
                          </div>
                        )}
                        <div className="space-y-1">
                          {bookList.map((doc, idx) =>
                            renderDocButton(
                              doc,
                              <Book className="w-3 h-3 flex-shrink-0 text-blue-300" />
                            )
                          )}
                        </div>
                      </div>
                    )}
                  {books.length === 0 && (
                    <div className="text-center text-purple-300 text-sm py-8 opacity-70">
                      No documents found. <br /> Upload some!
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 flex-shrink-0">
              <h3 className="text-lg font-semibold text-white mb-3">Stats</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between text-purple-200">
                  <span>Books:</span>
                  <span className="font-semibold text-white">
                    {bookList.length}
                  </span>
                </div>
                <div className="flex justify-between text-purple-200">
                  <span>Papers:</span>
                  <span className="font-semibold text-white">
                    {paperList.length}
                  </span>
                </div>
                <div className="flex justify-between text-purple-200">
                  <span>Sessions:</span>
                  <span className="font-semibold text-white">
                    {sessions.length}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="lg:col-span-3 space-y-4 flex flex-col h-full min-h-0">
            {showSettings && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-white flex items-center">
                    <Settings className="w-5 h-5 mr-2" />
                    Retrieval Settings
                  </h3>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={() => setUseBackendDefaults(!useBackendDefaults)}
                      className="flex items-center text-sm text-purple-200 hover:text-white transition-colors focus:outline-none"
                    >
                      {useBackendDefaults ? (
                        <ToggleRight className="w-8 h-8 text-purple-500 mr-2" />
                      ) : (
                        <ToggleLeft className="w-8 h-8 text-gray-400 mr-2" />
                      )}
                      <span>Use Server Defaults</span>
                    </button>
                    <div className="w-px h-6 bg-white/20"></div>
                    <button
                      onClick={() => signOut()}
                      className="flex items-center space-x-1 px-3 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-300 hover:text-red-200 rounded-lg transition-all border border-red-500/30 hover:border-red-500/50"
                      title="Logout"
                    >
                      <LogOut className="w-4 h-4" />
                      <span className="text-xs font-semibold">Logout</span>
                    </button>
                  </div>
                </div>
                <div
                  className={`grid grid-cols-2 gap-4 transition-opacity duration-300 mb-6 ${
                    useBackendDefaults ? "opacity-50" : "opacity-100"
                  }`}
                >
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">
                      Pass 1: Initial Candidates
                    </label>
                    <input
                      type="range"
                      min="30"
                      max="100"
                      aria-disabled={useBackendDefaults}
                      value={pass1K}
                      onChange={(e) => setPass1K(parseInt(e.target.value))}
                      className={`w-full accent-purple-500 ${
                        useBackendDefaults ? "pointer-events-none" : ""
                      }`}
                    />
                    <span className="text-white text-sm">{pass1K} chunks</span>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">
                      Pass 2: After Reranking
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="30"
                      disabled={useBackendDefaults}
                      value={pass2K}
                      onChange={(e) => setPass2K(parseInt(e.target.value))}
                      className={`w-full accent-purple-500 ${
                        useBackendDefaults ? "pointer-events-none" : ""
                      }`}
                    />
                    <span className="text-white text-sm">{pass2K} chunks</span>
                  </div>
                </div>
              </div>
            )}
            <div className="flex-1 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 flex flex-col min-h-0">
              <div
                className="flex-1 min-h-0 overflow-y-auto p-6 space-y-4 chat-messages scrollbar-thin"
                onScroll={handleUserScroll}
              >
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Sparkles className="w-16 h-16 text-purple-400 mb-4" />
                    <h3 className="text-2xl font-semibold text-white mb-2">
                      {currentSessionId
                        ? "Continue Your Conversation"
                        : "Research & Coding Assistant"}
                    </h3>
                    <p className="text-purple-200 max-w-md">
                      {currentSessionId
                        ? "Ask follow-up questions - I remember our conversation!"
                        : "Ask questions about your uploaded books and research papers. I can distinguish between theoretical proofs and coding implementation!"}
                    </p>
                    <p className="text-purple-300 text-sm mt-4">
                      💡 Tip: Use the toggle below to switch between Books
                      (Code) and Papers (Theory).
                    </p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <MessageBubble
                      key={idx}
                      message={msg}
                      id={`msg-${idx}`}
                      index={idx}
                      isEditing={editingMessageIndex === idx}
                      editingText={editingText}
                      onEditChange={setEditingText}
                      onEdit={handleEditQuery}
                      onCancelEdit={handleCancelEdit}
                      onSubmitEdit={handleSubmitEdit}
                      isLoading={loading}
                    />
                  ))
                )}
                {loading && (
                  <div className="flex justify-start">
                    <div className="max-w-2xl bg-white/10 backdrop-blur-lg border border-white/20 rounded-lg px-6 py-4 text-white/80 flex items-center space-x-3">
                      <span className="text-sm font-medium transition-all duration-300 ease-in-out">
                        {loadingStages[currentLoadingStage]}
                      </span>
                      <div className="flex items-center space-x-1.5">
                        <div
                          className={`w-2 h-2 rounded-full transition-all duration-300 ${
                            currentLoadingStage % 3 === 0
                              ? "bg-blue-400 scale-100"
                              : "bg-blue-400/40 scale-75"
                          }`}
                        />
                        <div
                          className={`w-2 h-2 rounded-full transition-all duration-300 ${
                            currentLoadingStage % 3 === 1
                              ? "bg-purple-400 scale-100"
                              : "bg-purple-400/40 scale-75"
                          }`}
                        />
                        <div
                          className={`w-2 h-2 rounded-full transition-all duration-300 ${
                            currentLoadingStage % 3 === 2
                              ? "bg-pink-400 scale-100"
                              : "bg-pink-400/40 scale-75"
                          }`}
                        />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              <div className="p-4 border-t border-white/20 space-y-3">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) =>
                      e.key === "Enter" && !loading && handleQuerySubmit(e)
                    }
                    placeholder={
                      currentSessionId
                        ? "Ask a follow-up question..."
                        : "Ask a question..."
                    }
                    className="flex-1 px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-purple-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    disabled={loading}
                  />
                  {!loading ? (
                    <button
                      onClick={handleQuerySubmit}
                      disabled={!query.trim()}
                      className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      <Search className="w-5 h-5" />
                    </button>
                  ) : (
                    <button
                      onClick={handleInterruptQuery}
                      className="p-3 bg-yellow-500 hover:bg-yellow-600 text-white rounded-none transition-all flex items-center justify-center shadow-lg hover:shadow-xl"
                      title="Pause the query processing"
                    >
                      <Pause className="w-5 h-5" />
                    </button>
                  )}
                </div>
                {currentSessionId && (
                  <div className="mt-2 text-xs text-purple-300 flex items-center">
                    <MessageCircle className="w-3 h-3 mr-1" />
                    Session active - I remember our conversation
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  id,
  index,
  isEditing,
  editingText,
  onEditChange,
  onEdit,
  onCancelEdit,
  onSubmitEdit,
  isLoading,
}) {
  const [showSources, setShowSources] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);
  const [showRewrittenQueries, setShowRewrittenQueries] = useState(false);

  if (message.role === "user") {
    if (isEditing) {
      return (
        <div id={id} className="flex justify-end">
          <div className="max-w-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg px-4 py-3 w-full">
            <div className="flex gap-2">
              <textarea
                value={editingText}
                onChange={(e) => onEditChange(e.target.value)}
                className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-200 focus:outline-none focus:ring-2 focus:ring-white resize-none"
                rows="3"
                disabled={isLoading}
                autoFocus
              />
            </div>
            <div className="flex justify-end space-x-2 mt-3">
              <button
                onClick={onCancelEdit}
                disabled={isLoading}
                className="px-3 py-1.5 bg-red-500/50 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => onSubmitEdit(index)}
                disabled={isLoading}
                className="px-3 py-1.5 bg-green-500 hover:bg-green-600 text-white rounded text-sm font-medium transition-colors disabled:opacity-50"
              >
                {isLoading ? "Submitting..." : "Submit"}
              </button>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div id={id} className="flex justify-end group">
        <div className="max-w-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg px-4 py-3 relative">
          <p className="whitespace-pre-wrap">{message.content}</p>
          <button
            onClick={() => onEdit(index, message.content)}
            className="absolute top-2 right-2 p-2 bg-white/20 hover:bg-white/30 text-white rounded opacity-0 group-hover:opacity-100 transition-all duration-200"
            title="Edit this query"
          >
            <Edit className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div id={id} className="flex justify-start">
      <div className="max-w-3xl bg-white/10 backdrop-blur-lg border border-white/20 rounded-lg px-4 py-3 text-white w-full">
        {message.error ? (
          <div className="flex items-start space-x-2 text-red-300">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{message.content}</p>
          </div>
        ) : (
          <>
            {message.answered_from_history && (
              <div className="mb-3 bg-blue-500/20 border border-blue-400/50 rounded-lg p-3">
                <div className="flex items-center space-x-2 text-blue-200">
                  <History className="w-4 h-4" />
                  <span className="text-sm font-semibold">
                    Answered from conversation memory
                  </span>
                </div>
                {message.resolved_query &&
                  message.resolved_query !== message.content && (
                    <div className="mt-2 text-xs text-blue-300">
                      Resolved query: "{message.resolved_query}"
                    </div>
                  )}
              </div>
            )}

            {message.rewritten_queries &&
              message.rewritten_queries.length > 0 && (
                <div className="mb-3">
                  <button
                    onClick={() =>
                      setShowRewrittenQueries(!showRewrittenQueries)
                    }
                    className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-colors mb-2 w-full justify-between bg-white/5 p-3 rounded-lg border border-white/10 hover:border-purple-400/50"
                  >
                    <div className="flex items-center space-x-2">
                      <Repeat className="w-4 h-4" />
                      <span className="font-semibold">
                        Query Expansion ({message.rewritten_queries.length}{" "}
                        variations)
                      </span>
                    </div>
                    {showRewrittenQueries ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                  {showRewrittenQueries && (
                    <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg p-4 border border-blue-400/30">
                      <div className="space-y-2">
                        {message.rewritten_queries.map((query, idx) => (
                          <div
                            key={idx}
                            className="flex items-start space-x-2 bg-white/5 rounded p-2 border border-white/10"
                          >
                            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gradient-to-br from-blue-400 to-purple-400 flex items-center justify-center text-xs font-bold text-white">
                              {idx + 1}
                            </div>
                            <p className="flex-1 text-sm text-gray-200 leading-relaxed">
                              {query}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

            <div className="bg-white/5 rounded-lg p-4 mb-3 border border-white/10">
              <div className="flex items-center space-x-2 mb-2">
                <MessageSquare className="w-5 h-5 text-purple-300" />
                <h4 className="font-semibold text-purple-100">Answer</h4>
              </div>
              <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/50 max-w-none text-gray-100">
                <ReactMarkdown
                  components={{
                    code: ({ node, inline, className, children, ...props }) => {
                      const [copied, setCopied] = React.useState(false);
                      const handleCopy = () => {
                        const codeText = String(children).replace(/\n$/, "");
                        navigator.clipboard.writeText(codeText);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      };
                      if (inline)
                        return (
                          <code
                            className="bg-white/10 rounded px-1 py-0.5"
                            {...props}
                          >
                            {children}
                          </code>
                        );
                      return (
                        <div className="relative group my-4">
                          <button
                            onClick={handleCopy}
                            className="absolute top-2 right-2 p-2 bg-purple-600/80 hover:bg-purple-600 text-white rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 z-10 flex items-center space-x-1.5 shadow-lg"
                          >
                            {copied ? (
                              <>
                                <CheckCircle className="w-4 h-4" />
                                <span className="text-xs">Copied!</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-4 h-4" />
                                <span className="text-xs">Copy</span>
                              </>
                            )}
                          </button>
                          <code
                            className="block bg-gradient-to-br from-gray-900 to-gray-800 p-4 pt-6 rounded-lg overflow-x-auto border border-white/10"
                            {...props}
                          >
                            {children}
                          </code>
                        </div>
                      );
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>

            {message.pipeline_stages && message.pipeline_stages.length > 0 && (
              <div className="mb-3">
                <button
                  onClick={() => setShowPipeline(!showPipeline)}
                  className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-colors mb-2 w-full justify-between bg-white/5 p-3 rounded-lg border border-white/10 hover:border-purple-400/50"
                >
                  <div className="flex items-center space-x-2">
                    <Layers className="w-4 h-4" />
                    <span className="font-semibold">
                      Retrieval Pipeline ({message.stats?.final || 0} final
                      chunks)
                    </span>
                  </div>
                  {showPipeline ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>
                {showPipeline && (
                  <EnhancedPipelineDisplay
                    stages={message.pipeline_stages}
                    stats={message.stats}
                  />
                )}
              </div>
            )}

            {message.sources && message.sources.length > 0 && (
              <div className="mt-3">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-all duration-300 w-full justify-between bg-gradient-to-r from-white/5 to-white/10 p-4 rounded-xl border border-white/10 hover:border-purple-400/50 hover:shadow-lg hover:shadow-purple-500/20 group"
                >
                  <div className="flex items-center space-x-2">
                    <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg group-hover:scale-110 transition-transform duration-300">
                      <Book className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-semibold">
                      {message.sources.length} Source
                      {message.sources.length > 1 ? "s" : ""}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs px-2 py-1 bg-purple-500/20 rounded-full border border-purple-400/30">
                      References
                    </span>
                    {showSources ? (
                      <ChevronUp className="w-4 h-4 group-hover:transform group-hover:-translate-y-1 transition-transform" />
                    ) : (
                      <ChevronDown className="w-4 h-4 group-hover:transform group-hover:translate-y-1 transition-transform" />
                    )}
                  </div>
                </button>
                {showSources && (
                  <div className="mt-3 space-y-3">
                    {message.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="group bg-gradient-to-br from-purple-900/20 via-pink-900/10 to-purple-900/20 rounded-xl p-4 border border-purple-400/20 hover:border-purple-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-purple-500/20 backdrop-blur-sm relative overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="relative flex items-start space-x-4">
                          <div className="bg-gradient-to-br from-purple-500 via-purple-600 to-pink-500 rounded-xl p-3 flex-shrink-0 shadow-lg shadow-purple-500/30 group-hover:scale-105 group-hover:rotate-3 transition-all duration-300">
                            {source.book_title.includes("Paper") ||
                            source.total_pages < 50 ? (
                              <FileText className="w-5 h-5 text-white" />
                            ) : (
                              <Book className="w-5 h-5 text-white" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-base font-bold text-white mb-2 leading-tight group-hover:text-purple-100 transition-colors">
                              {source.book_title}
                            </div>
                            {source.author && (
                              <div className="text-xs text-purple-300/80 mb-3 italic flex items-center space-x-1">
                                <span className="w-1 h-1 bg-purple-400 rounded-full" />
                                <span>by {source.author}</span>
                              </div>
                            )}
                            <div className="inline-flex items-center bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-400/30 rounded-lg px-3 py-2 text-xs text-purple-200">
                              <div className="w-1.5 h-1.5 bg-purple-400 rounded-full mr-2 animate-pulse" />
                              <span className="font-medium">
                                {source.chapter}
                              </span>
                            </div>
                          </div>
                          {source.relevance && (
                            <div className="flex-shrink-0">
                              <div className="relative">
                                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg blur opacity-50 group-hover:opacity-75 transition-opacity" />
                                <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-bold px-4 py-2 rounded-lg border border-purple-400/30 shadow-lg">
                                  {source.relevance.toFixed(0)}%
                                </div>
                              </div>
                              <div className="text-[10px] text-purple-300 text-center mt-1">
                                relevance
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function EnhancedPipelineDisplay({ stages, stats }) {
  const [expandedStage, setExpandedStage] = useState(null);
  const [showingChunks, setShowingChunks] = useState({});

  const getStageColor = (index) =>
    [
      "from-blue-500 to-blue-600",
      "from-purple-500 to-purple-600",
      "from-pink-500 to-pink-600",
      "from-green-500 to-green-600",
      "from-yellow-500 to-yellow-600",
      "from-indigo-500 to-indigo-600",
    ][index % 6];
  const getChangeIndicator = (currentCount, previousCount) => {
    if (previousCount === null) return null;
    const diff = currentCount - previousCount;
    if (diff > 0)
      return <span className="text-green-300 text-xs ml-2">(+{diff})</span>;
    if (diff < 0)
      return <span className="text-orange-300 text-xs ml-2">({diff})</span>;
    return null;
  };

  return (
    <div className="mt-3 space-y-3 bg-black/20 p-4 rounded-lg">
      <div className="flex items-center justify-between text-sm bg-white/5 p-3 rounded border border-white/10">
        <div className="flex items-center space-x-4">
          <span className="text-white font-semibold">{stats?.pass1 || 0}</span>
          <span className="text-purple-300">→</span>
          <span className="text-white font-semibold">{stats?.pass2 || 0}</span>
          <span className="text-purple-300">→</span>
          <span className="text-white font-semibold">{stats?.pass3 || 0}</span>
          <span className="text-purple-300">→</span>
          <span className="text-green-300 font-semibold">
            {stats?.final || 0}
          </span>
        </div>
        {stats?.tokens && (
          <span className="text-purple-200 text-xs">{stats.tokens} tokens</span>
        )}
      </div>
      {stages.map((stage, index) => {
        const previousCount = index > 0 ? stages[index - 1].chunk_count : null;
        const isExpanded = expandedStage === index;
        const hasChunks = stage.chunks && stage.chunks.length > 0;
        return (
          <div key={index} className="relative">
            {index < stages.length - 1 && (
              <div className="absolute left-6 top-full h-3 w-0.5 bg-purple-400/30" />
            )}
            <div className="bg-white/5 rounded-lg border border-white/10 overflow-hidden">
              <div
                className={`bg-gradient-to-r ${getStageColor(
                  index
                )} p-3 flex items-center justify-between cursor-pointer`}
                onClick={() => setExpandedStage(isExpanded ? null : index)}
              >
                <div className="flex items-center space-x-3">
                  <div className="bg-white/20 rounded-full p-1.5">
                    <Filter className="w-3 h-3 text-white" />
                  </div>
                  <div>
                    <div className="text-white font-semibold text-sm">
                      {stage.stage_name}
                    </div>
                    <div className="text-white/80 text-xs">
                      {stage.chunk_count} chunks total
                      {getChangeIndicator(stage.chunk_count, previousCount)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {hasChunks && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowingChunks((prev) => ({
                          ...prev,
                          [index]: !prev[index],
                        }));
                      }}
                      className="px-2 py-1 bg-white/20 hover:bg-white/30 rounded text-xs text-white"
                    >
                      {showingChunks[index] ? "Hide" : "View"}
                    </button>
                  )}
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-white" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-white" />
                  )}
                </div>
              </div>
              {isExpanded && (
                <div className="p-3 bg-black/20 text-xs text-purple-200">
                  {index === 0 &&
                    "Broad semantic search using vector similarity with query expansion."}
                  {index === 1 &&
                    "Precision ranking with cross-encoder to select most relevant."}
                  {index === 2 &&
                    "Intelligent expansion following related concepts."}
                  {index === 3 && "Cluster-based expansion to find semantically related chunks."}
                  {index === 4 && "Final context assembly and compression for LLM input."}
                  {!stage.stage_name.includes("Pass") && "Pipeline stage processing."}
                </div>
              )}
              {showingChunks[index] && hasChunks && (
                <div className="p-3 bg-black/30 max-h-64 overflow-y-auto space-y-2">
                  {stage.chunks.slice(0, 5).map((chunk, i) => (
                    <div key={i} className="bg-white/5 rounded p-2 text-xs">
                      <div className="font-semibold text-white mb-1 flex items-center justify-between">
                        <span>{chunk.book_title}</span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] ${
                            chunk.type === "code"
                              ? "bg-green-500/30 text-green-200"
                              : "bg-blue-500/30 text-blue-200"
                          }`}
                        >
                          {chunk.type}
                        </span>
                      </div>
                      <div className="text-purple-300 text-[11px] mb-1">
                        {chunk.chapter} • {chunk.relevance?.toFixed(0)}%
                        relevant
                      </div>
                      <div className="text-gray-300 text-[10px] bg-black/30 p-1 rounded">
                        {chunk.content_preview?.substring(0, 100)}...
                      </div>
                    </div>
                  ))}
                  {stage.chunks.length > 5 && (
                    <div className="text-center text-purple-300 text-xs">
                      + {stage.chunks.length - 5} more chunks
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

<style>
  {`
input[type="range"]::-webkit-slider-runnable-track {
  height: 6px;
  background: linear-gradient(to right, #8b5cf6, #ec4899);
  border-radius: 3px;
}
input[type="range"]::-webkit-slider-thumb {
  width: 18px;
  height: 18px;
  background: #fff;
  border: 2px solid #8b5cf6;
  border-radius: 50%;
  margin-top: -6px;
  cursor: pointer;
  box-shadow: 0 0 2px #8b5cf6;
}
input[type="range"]:disabled::-webkit-slider-thumb {
  background: #e5e7eb;
  border-color: #a78bfa;
}
input[type="range"]:focus::-webkit-slider-thumb {
  outline: 2px solid #8b5cf6;
}
input[type="range"]::-moz-range-track {
  height: 6px;
  background: linear-gradient(to right, #8b5cf6, #ec4899);
  border-radius: 3px;
}
input[type="range"]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  background: #fff;
  border: 2px solid #8b5cf6;
  border-radius: 50%;
  cursor: pointer;
}
input[type="range"]:disabled::-moz-range-thumb {
  background: #e5e7eb;
  border-color: #a78bfa;
}
input[type="range"]::-ms-fill-lower {
  background: #8b5cf6;
}
input[type="range"]::-ms-fill-upper {
  background: #ec4899;
}
input[type="range"] {
  background: transparent;
}
`}
</style>;