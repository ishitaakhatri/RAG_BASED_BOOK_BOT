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

const API_BASE_URL = "/api";

const BACKEND_DEFAULTS = {
  pass1K: 50,
  pass2K: 15,
  pass3Enabled: true,
  maxTokens: 30000,
};

export default function RAGBookBot() {
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

    pdf.setFillColor(59, 130, 246);
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
        pdf.setFillColor(59, 130, 246);
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
        pdf.setFillColor(16, 185, 129);
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
      const response = await authFetch(`/sessions?limit=50`);
      const data = await response.json();
      const processedSessions = (data.sessions || []).map((session) => ({
        ...session,
        title:
          session.title ||
          session.first_message?.substring(0, 50) + "..." ||
          `Conversation ${session.session_id}`,
        last_message:
          session.last_message ||
          session.last_assistant_response?.substring(0, 80) + "..." ||
          "No messages yet",
      }));
      setSessions(processedSessions);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
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
          ? "bg-blue-600 text-white shadow-md"
          : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 hover:border-blue-300"
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
    <div className="h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 text-gray-900 relative overflow-hidden flex">
      {/* Subtle Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-20 w-96 h-96 bg-indigo-200/20 rounded-full blur-3xl" />
      </div>
      
      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(99,102,241,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.03)_1px,transparent_1px)] bg-[size:64px_64px]" />
      
      <div className="relative z-10 flex flex-1 w-full">
        {/* Sidebar */}
        <div
          className={`${
            showSessions ? "w-80" : "w-0"
          } transition-all duration-300 bg-white border-r border-gray-200 overflow-hidden flex flex-col shadow-lg`}
        >
          <div className="p-4 border-b border-gray-200">
            <button
              onClick={startNewChat}
              className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all shadow-md"
            >
              <Plus className="w-5 h-5" />
              <span className="font-semibold">New Chat</span>
            </button>
          </div>
          <div className="p-4 border-b border-gray-200">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && searchSessions()}
                placeholder="Search conversations..."
                className="w-full pl-10 pr-4 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
              {searchQuery && (
                <button
                  onClick={() => {
                    setSearchQuery("");
                    setSearchResults([]);
                  }}
                  className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-700"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            {isSearching && (
              <div className="mt-2 text-xs text-blue-600 flex items-center">
                <Loader className="w-3 h-3 animate-spin mr-2" />
                Searching...
              </div>
            )}
          </div>
          <div
            className="flex-1 overflow-y-scroll p-4 space-y-2"
            style={{ maxHeight: "980px" }}
          >
            {searchResults.length > 0 ? (
              <>
                <div className="text-xs text-gray-600 mb-2 font-medium">
                  {searchResults.length} results for "{searchQuery}"
                </div>
                {searchResults.map((result, idx) => (
                  <div
                    key={idx}
                    onClick={() => loadSession(result.session_id)}
                    className="bg-gray-50 hover:bg-blue-50 rounded-lg p-3 cursor-pointer transition-all border border-gray-200 hover:border-blue-300"
                  >
                    <div className="text-sm text-gray-900 font-medium mb-1 truncate">
                      {result.user_query}
                    </div>
                    <div className="text-xs text-gray-600 mb-2 line-clamp-2">
                      {result.assistant_response}
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span className="flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {formatTimestamp(result.timestamp)}
                      </span>
                      <span className="text-green-600 font-medium">
                        {(result.relevance_score * 100).toFixed(0)}% match
                      </span>
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <>
                {currentSessionId && (
                  <div className="mb-2 text-xs text-gray-600 font-semibold uppercase tracking-wide">
                    Current Chat
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
                          ? "bg-blue-50 border-blue-300 shadow-sm"
                          : "bg-gray-50 hover:bg-blue-50 border-gray-200 hover:border-blue-300"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-gray-900 font-medium mb-1 truncate">
                            {session.title}
                          </div>
                          <div className="text-xs text-gray-600 truncate mb-2">
                            {session.last_message}
                          </div>
                          <div className="flex items-center space-x-3 text-xs text-gray-500">
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
                          className="ml-2 p-1 hover:bg-red-100 rounded text-red-500 hover:text-red-700 transition-colors"
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
          <header className="bg-white border-b border-gray-200 shadow-sm">
            <div className="px-4 sm:px-6 lg:px-8 py-4">
              <div className="relative flex items-center">
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setShowSessions(!showSessions)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-all text-gray-700"
                  >
                    <History className="w-5 h-5" />
                  </button>
                  <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-lg shadow-md">
                    <Library className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">
                      RAG Knowledge Bot
                    </h1>
                    <p className="text-sm text-gray-600">
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
                    className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
                    title="Download chat history"
                  >
                    <Download className="w-4 h-4" />
                    <span className="hidden sm:inline">Download</span>
                  </button>
                  <button
                    onClick={() => navigate("/ingest")}
                    className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all shadow-md"
                  >
                    <Upload className="w-4 h-4" />
                    <span className="hidden sm:inline">Upload Doc</span>
                  </button>
                  <button
                    onClick={() => setShowSettings(!showSettings)}
                    className="flex-shrink-0 p-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all"
                  >
                    <Settings className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </header>
          
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full min-h-0 p-6">
            <div className="lg:col-span-1 space-y-4 overflow-hidden flex flex-col">
              <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex-1 flex flex-col min-h-0">
                <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center flex-shrink-0">
                  <Library className="w-5 h-5 mr-2" />
                  Library{" "}
                  <span className="ml-auto text-xs font-normal text-gray-600 bg-gray-100 px-2 py-1 rounded">
                    {searchMode === "all"
                      ? "All"
                      : searchMode === "books"
                      ? "Books"
                      : "Papers"}
                  </span>
                </h3>
                <div className="space-y-2 flex-1 flex flex-col min-h-0">
                  <div className="flex justify-center">
                    <div className="bg-gray-100 p-1 rounded-lg flex space-x-1">
                      <button
                        onClick={() => setSearchMode("all")}
                        className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${
                          searchMode === "all"
                            ? "bg-blue-600 text-white shadow-md"
                            : "text-gray-600 hover:bg-gray-200"
                        }`}
                      >
                        All Sources
                      </button>
                      <button
                        onClick={() => setSearchMode("books")}
                        className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${
                          searchMode === "books"
                            ? "bg-blue-600 text-white shadow-md"
                            : "text-gray-600 hover:bg-gray-200"
                        }`}
                      >
                        <BookOpen className="w-3 h-3 mr-1" /> Books
                      </button>
                      <button
                        onClick={() => setSearchMode("papers")}
                        className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center space-x-1 ${
                          searchMode === "papers"
                            ? "bg-green-600 text-white shadow-md"
                            : "text-gray-600 hover:bg-gray-200"
                        }`}
                      >
                        <GraduationCap className="w-3 h-3 mr-1" /> Papers
                      </button>
                    </div>
                  </div>
                  <div className="overflow-y-auto pr-2 space-y-3 flex-1">
                    {(searchMode === "all" || searchMode === "papers") &&
                      paperList.length > 0 && (
                        <div className="animate-fade-in">
                          {searchMode === "all" && (
                            <div className="text-xs font-bold text-green-600 uppercase tracking-wider mb-1 mt-2 flex items-center">
                              <GraduationCap className="w-3 h-3 mr-1" />{" "}
                              Research Papers
                            </div>
                          )}
                          <div className="space-y-1">
                            {paperList.map((doc, idx) =>
                              renderDocButton(
                                doc,
                                <FileText className="w-3 h-3 flex-shrink-0 text-green-600" />
                              )
                            )}
                          </div>
                        </div>
                      )}
                    {(searchMode === "all" || searchMode === "books") &&
                      bookList.length > 0 && (
                        <div className="animate-fade-in">
                          {searchMode === "all" && (
                            <div className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-1 mt-4 flex items-center">
                              <Book className="w-3 h-3 mr-1" /> Books
                            </div>
                          )}
                          <div className="space-y-1">
                            {bookList.map((doc, idx) =>
                              renderDocButton(
                                doc,
                                <Book className="w-3 h-3 flex-shrink-0 text-blue-600" />
                              )
                            )}
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
              <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm flex-shrink-0">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Stats</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between text-gray-600">
                    <span>Books:</span>
                    <span className="font-semibold text-gray-900">
                      {bookList.length}
                    </span>
                  </div>
                  <div className="flex justify-between text-gray-600">
                    <span>Papers:</span>
                    <span className="font-semibold text-gray-900">
                      {paperList.length}
                    </span>
                  </div>
                  <div className="flex justify-between text-gray-600">
                    <span>Sessions:</span>
                    <span className="font-semibold text-gray-900">
                      {sessions.length}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="lg:col-span-3 space-y-4 flex flex-col h-full min-h-0">
              {showSettings && (
                <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                      <Settings className="w-5 h-5 mr-2" />
                      Retrieval Settings
                    </h3>
                    <div className="flex items-center space-x-3">
                      <button
                        onClick={() =>
                          setUseBackendDefaults(!useBackendDefaults)
                        }
                        className="flex items-center text-sm text-gray-600 hover:text-gray-900 transition-colors focus:outline-none"
                      >
                        {useBackendDefaults ? (
                          <ToggleRight className="w-8 h-8 text-blue-600 mr-2" />
                        ) : (
                          <ToggleLeft className="w-8 h-8 text-gray-400 mr-2" />
                        )}
                        <span>Use Server Defaults</span>
                      </button>
                      <div className="w-px h-6 bg-gray-300"></div>
                      <button
                        onClick={() => signOut()}
                        className="flex items-center space-x-1 px-3 py-2 bg-red-50 hover:bg-red-100 text-red-600 hover:text-red-700 rounded-lg transition-all border border-red-200"
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
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Pass 1: Initial Candidates
                      </label>
                      <input
                        type="range"
                        min="30"
                        max="100"
                        aria-disabled={useBackendDefaults}
                        value={pass1K}
                        onChange={(e) => setPass1K(parseInt(e.target.value))}
                        className={`w-full accent-blue-600 ${
                          useBackendDefaults ? "pointer-events-none" : ""
                        }`}
                      />
                      <span className="text-gray-900 text-sm">
                        {pass1K} chunks
                      </span>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Pass 2: After Reranking
                      </label>
                      <input
                        type="range"
                        min="10"
                        max="30"
                        disabled={useBackendDefaults}
                        value={pass2K}
                        onChange={(e) => setPass2K(parseInt(e.target.value))}
                        className={`w-full accent-blue-600 ${
                          useBackendDefaults ? "pointer-events-none" : ""
                        }`}
                      />
                      <span className="text-gray-900 text-sm">
                        {pass2K} chunks
                      </span>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="flex-1 bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col min-h-0">
                <div
                  className="flex-1 min-h-0 overflow-y-auto p-6 space-y-4"
                  onScroll={handleUserScroll}
                >
                  {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                      <Sparkles className="w-16 h-16 text-blue-500 mb-4" />
                      <h3 className="text-2xl font-semibold text-gray-900 mb-2">
                        {currentSessionId
                          ? "Continue Your Conversation"
                          : "Research & Coding Assistant"}
                      </h3>
                      <p className="text-gray-600 max-w-md">
                        {currentSessionId
                          ? "Ask follow-up questions - I remember our conversation!"
                          : "Ask questions about your uploaded books and research papers. I can distinguish between theoretical proofs and coding implementation!"}
                      </p>
                      <p className="text-gray-500 text-sm mt-4">
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
                      <div className="max-w-2xl bg-blue-50 border border-blue-200 rounded-2xl p-4 shadow-sm">
                        <div className="flex items-center space-x-3">
                          <Loader className="w-5 h-5 text-blue-600 animate-spin" />
                          <span className="text-sm text-gray-700 font-medium">
                            {loadingStages[currentLoadingStage]}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-gray-200 bg-gray-50">
                  <form onSubmit={handleQuerySubmit} className="flex space-x-3">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      className="flex-1 px-4 py-3 bg-white border border-gray-300 rounded-xl text-gray-900 placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                      disabled={loading}
                    />
                    {loading ? (
                      <button
                        type="button"
                        onClick={handleInterruptQuery}
                        className="px-6 py-3 bg-red-600 text-white rounded-xl hover:bg-red-700 transition-all font-semibold shadow-md flex items-center space-x-2"
                      >
                        <Pause className="w-5 h-5" />
                        <span>Stop</span>
                      </button>
                    ) : (
                      <button
                        type="submit"
                        disabled={!query.trim()}
                        className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-md flex items-center space-x-2"
                      >
                        <MessageSquare className="w-5 h-5" />
                        <span>Send</span>
                      </button>
                    )}
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// MessageBubble Component (keeping original logic, updating styles)
const MessageBubble = ({
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
}) => {
  const [showSources, setShowSources] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  if (message.role === "user") {
    return (
      <div className="flex justify-end" id={id}>
        <div className="max-w-2xl">
          {isEditing ? (
            <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4 shadow-sm">
              <textarea
                value={editingText}
                onChange={(e) => onEditChange(e.target.value)}
                className="w-full p-3 bg-white border border-gray-300 rounded-lg text-gray-900 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 mb-3"
                rows={4}
              />
              <div className="flex space-x-2">
                <button
                  onClick={() => onSubmitEdit(index)}
                  disabled={isLoading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium transition-all"
                >
                  Resend
                </button>
                <button
                  onClick={onCancelEdit}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 text-sm font-medium transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="group">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl px-5 py-3 shadow-md">
                <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                  {message.content}
                </p>
              </div>
              <button
                onClick={() => onEdit(index, message.content)}
                className="mt-2 text-xs text-gray-500 hover:text-blue-600 transition-colors opacity-0 group-hover:opacity-100 flex items-center space-x-1"
              >
                <Edit className="w-3 h-3" />
                <span>Edit</span>
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start" id={id}>
      <div className="max-w-3xl w-full">
        <div className="bg-gray-50 border border-gray-200 rounded-2xl p-5 shadow-sm">
          {message.answered_from_history && (
            <div className="mb-3 flex items-center space-x-2 text-xs text-blue-600 bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200 w-fit">
              <History className="w-3.5 h-3.5" />
              <span className="font-medium">Answered from conversation history</span>
            </div>
          )}
          
          {message.resolved_query && message.resolved_query !== message.content && (
            <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-xs font-semibold text-blue-700 mb-1 flex items-center">
                <Repeat className="w-3.5 h-3.5 mr-1" />
                Resolved Query
              </div>
              <p className="text-sm text-gray-700 italic">{message.resolved_query}</p>
            </div>
          )}

          <div className="prose prose-sm max-w-none">
            <ReactMarkdown className="text-gray-800 leading-relaxed">
              {message.content}
            </ReactMarkdown>
          </div>

          <div className="flex items-center space-x-3 mt-4 pt-3 border-t border-gray-200">
            <button
              onClick={() => copyToClipboard(message.content, id)}
              className="text-xs text-gray-500 hover:text-blue-600 transition-colors flex items-center space-x-1"
            >
              {copiedId === id ? (
                <>
                  <CheckCircle className="w-3.5 h-3.5" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>

            {message.sources && message.sources.length > 0 && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-xs text-gray-500 hover:text-blue-600 transition-colors flex items-center space-x-1"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>{message.sources.length} Sources</span>
                {showSources ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            )}

            {message.stats && (
              <button
                onClick={() => setShowStats(!showStats)}
                className="text-xs text-gray-500 hover:text-blue-600 transition-colors flex items-center space-x-1"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Stats</span>
                {showStats ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            )}
          </div>

          {showSources && message.sources && message.sources.length > 0 && (
            <div className="mt-4 space-y-2">
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
                Sources
              </h4>
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-white border border-gray-200 rounded-lg text-xs"
                >
                  <div className="font-semibold text-gray-900 mb-1">
                    {source.title} - Page {source.page}
                  </div>
                  <div className="text-gray-600 leading-relaxed">
                    {source.content.substring(0, 200)}...
                  </div>
                  {source.score && (
                    <div className="mt-2 text-gray-500">
                      Relevance: {(source.score * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {showStats && message.stats && (
            <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg">
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-3">
                Performance Stats
              </h4>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Time:</span>
                  <span className="font-medium text-gray-900">
                    {message.stats.total_time?.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Retrieval:</span>
                  <span className="font-medium text-gray-900">
                    {message.stats.retrieval_time?.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Generation:</span>
                  <span className="font-medium text-gray-900">
                    {message.stats.generation_time?.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Tokens:</span>
                  <span className="font-medium text-gray-900">
                    {message.stats.tokens_used}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
