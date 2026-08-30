import React, { useState, useRef, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import "tailwindcss/tailwind.css";
import jsPDF from "jspdf";
import { useAuth, UserButton } from "@clerk/clerk-react";

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
  Pause,
  CheckSquare,
  Square,
  Zap,
  HelpCircle,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

const BACKEND_DEFAULTS = {
  pass1K: 50,
  pass2K: 15,
  pass3Enabled: true,
  maxTokens: 30000,
};

// --- ERROR BOUNDARY COMPONENT ---
// This prevents the "White Screen of Death" by catching render errors
class SafeErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("React Render Error Caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
          <div className="font-bold flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Component Error
          </div>
          <p className="mt-1 opacity-75">
            This message could not be displayed. Check console for details.
          </p>
          <pre className="mt-2 text-xs bg-red-100 p-2 rounded overflow-auto max-w-full">
            {this.state.error?.toString()}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function RAGBookBot() {
  const { getToken } = useAuth();

  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBooks, setSelectedBooks] = useState([]);  // Changed to array for multi-select
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
  const [libraryFilter, setLibraryFilter] = useState("");

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

  const { bookList, paperList, totalChunks } = useMemo(() => {
    const b = [];
    const p = [];
    let chunks = 0;

    books.forEach((item) => {
      chunks += item.total_chunks || 0;
      const isPaper =
        item.total_chunks < 100 ||
        (item.title && item.title.toLowerCase().includes("paper"));
      if (isPaper) p.push(item);
      else b.push(item);
    });
    return { bookList: b, paperList: p, totalChunks: chunks };
  }, [books]);

  // Filter for visual display only (doesn't affect retrieval)
  const filterDoc = (doc) => {
    if (!libraryFilter) return true;
    const term = libraryFilter.toLowerCase();
    return doc.title?.toLowerCase().includes(term) || doc.author?.toLowerCase().includes(term);
  };

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
          sources: [], // Sources not loaded from history in this simplified version to avoid crashes
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
      book_filter: selectedBooks.length === 0 ? null : selectedBooks,
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
      console.log("DEBUG: API Response Received", data); // Debug Log

      if (data.error || data.detail) {
        const errorMsg =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || data.error);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${errorMsg}`,
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
        console.error("DEBUG: Fetch Error", error);
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
      book_filter: selectedBooks.length === 0 ? null : selectedBooks,
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
        const errorMsg =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || data.error);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${errorMsg}`,
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
    if (!timestamp) return "";
    try {
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
    } catch (e) {
      return "";
    }
  };

  const renderDocButton = (doc, icon) => {
    const isSelected = selectedBooks.includes(doc.title);
    const isPaper = doc.total_chunks < 100 || (doc.title && doc.title.toLowerCase().includes("paper"));
    const toggleSelection = () => {
      if (isSelected) {
        setSelectedBooks(selectedBooks.filter(t => t !== doc.title));
      } else {
        setSelectedBooks([...selectedBooks, doc.title]);
      }
    };
    return (
      <button
        key={doc.title}
        onClick={toggleSelection}
        className={`w-full text-left p-3 rounded-xl transition-all border shadow-sm ${isSelected
            ? "bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-300 shadow-emerald-100"
            : "bg-white border-stone-200 hover:border-stone-300 hover:shadow-md"
          }`}
        title={`${doc.title} by ${doc.author}`}
      >
        <div className="flex items-start gap-3">
          {/* Checkbox */}
          <div className={`mt-0.5 flex-shrink-0 ${isSelected ? "text-emerald-600" : "text-stone-400"}`}>
            {isSelected ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4" />}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={`flex-shrink-0 ${isPaper ? "text-teal-500" : "text-emerald-500"}`}>
                {isPaper ? <FileText className="w-3.5 h-3.5" /> : <Book className="w-3.5 h-3.5" />}
              </span>
              <span className={`text-sm font-medium truncate ${isSelected ? "text-emerald-900" : "text-stone-800"}`}>
                {doc.title}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs text-stone-500">
              <span className="truncate">by {doc.author}</span>
              <span className={`flex-shrink-0 ${isSelected ? "text-emerald-600" : "text-stone-400"}`}>
                {doc.total_chunks || 0} chunks
              </span>
            </div>
          </div>
        </div>
      </button>
    );
  };

  return (
    <div className="h-screen bg-gradient-to-br from-stone-50 via-stone-100 to-amber-50/30 text-stone-800 relative overflow-hidden flex">
      {/* Subtle Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-emerald-200/15 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-20 w-96 h-96 bg-amber-200/15 rounded-full blur-3xl" />
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(16,185,129,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(16,185,129,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      <div className="relative z-10 flex flex-1 w-full">
        {/* Sidebar */}
        <div
          className={`${showSessions ? "w-80" : "w-0"
            } transition-all duration-300 bg-white border-r border-stone-200 overflow-hidden flex flex-col shadow-lg`}
        >
          <div className="p-4 border-b border-stone-200">
            <button
              onClick={startNewChat}
              className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg hover:from-emerald-600 hover:to-teal-700 transition-all shadow-md shadow-emerald-500/20"
            >
              <Plus className="w-5 h-5" />
              <span className="font-semibold">New Chat</span>
            </button>
          </div>
          <div className="p-4 border-b border-stone-200">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && searchSessions()}
                placeholder="Search conversations..."
                className="w-full pl-10 pr-4 py-2 bg-stone-50 border border-stone-300 rounded-lg text-stone-800 placeholder-stone-400 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
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
              <div className="mt-2 text-xs text-emerald-600 flex items-center">
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
                    className="bg-stone-50 hover:bg-emerald-50 rounded-lg p-3 cursor-pointer transition-all border border-stone-200 hover:border-emerald-300"
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
                      className={`rounded-lg p-3 cursor-pointer transition-all border ${isCurrent
                        ? "bg-emerald-50 border-emerald-300 shadow-sm"
                        : "bg-stone-50 hover:bg-emerald-50 border-stone-200 hover:border-emerald-300"
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
          <header className="bg-white border-b border-stone-200 shadow-sm">
            <div className="px-4 sm:px-6 lg:px-8 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setShowSessions(!showSessions)}
                    className={`p-2 rounded-lg transition-all ${showSessions ? "bg-emerald-100 text-emerald-700" : "hover:bg-gray-100 text-gray-700"}`}
                    title="Toggle conversation history"
                  >
                    <History className="w-5 h-5" />
                  </button>
                  <div className="bg-gradient-to-br from-emerald-500 to-teal-600 p-2 rounded-lg shadow-md shadow-emerald-500/20">
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
                <div className="flex items-center space-x-3">
                  <button
                    onClick={downloadChat}
                    disabled={messages.length === 0}
                    className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-stone-600 text-white rounded-lg hover:bg-stone-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
                    title="Download chat as PDF"
                  >
                    <Download className="w-4 h-4" />
                    <span className="hidden sm:inline">Download</span>
                  </button>
                  <button
                    onClick={() => navigate("/ingest")}
                    className="flex-shrink-0 flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg hover:from-emerald-600 hover:to-teal-700 transition-all shadow-md"
                  >
                    <Upload className="w-4 h-4" />
                    <span className="hidden sm:inline">Upload Doc</span>
                  </button>
                  <button
                    onClick={() => setShowSettings(!showSettings)}
                    className={`flex-shrink-0 p-2 rounded-lg transition-all ${showSettings ? "bg-emerald-100 text-emerald-700" : "bg-gray-100 text-gray-700 hover:bg-gray-200"}`}
                    title="Toggle settings"
                  >
                    <Settings className="w-5 h-5" />
                  </button>
                  <UserButton
                    afterSignOutUrl="/"
                    appearance={{
                      elements: {
                        avatarBox: "w-9 h-9",
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full min-h-0 p-6">
            {/* Streamlined Library Panel */}
            <div className="lg:col-span-1 overflow-hidden flex flex-col">
              <div className="bg-white rounded-xl p-4 border border-stone-200 shadow-sm flex-1 flex flex-col min-h-0">
                {/* Header */}
                <div className="flex items-center justify-between mb-3 flex-shrink-0">
                  <h3 className="text-base font-semibold text-gray-900 flex items-center">
                    <Filter className="w-4 h-4 mr-2 text-emerald-600" />
                    Sources
                    {selectedBooks.length > 0 && (
                      <span className="ml-2 text-xs bg-emerald-500 text-white px-1.5 py-0.5 rounded-full font-medium">
                        {selectedBooks.length}
                      </span>
                    )}
                  </h3>
                  {selectedBooks.length > 0 && (
                    <button
                      onClick={() => setSelectedBooks([])}
                      className="text-xs text-stone-500 hover:text-stone-700"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {/* Selected Filter Chips */}
                {selectedBooks.length > 0 && (
                  <div className="mb-3 flex-shrink-0 flex flex-wrap gap-1">
                    {selectedBooks.map(title => (
                      <div key={title} className="flex items-center bg-emerald-100 text-emerald-800 rounded px-2 py-0.5 text-xs">
                        <span className="truncate max-w-[100px]">{title}</span>
                        <button
                          onClick={() => setSelectedBooks(selectedBooks.filter(t => t !== title))}
                          className="ml-1 text-emerald-600 hover:text-emerald-800"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Compact Mode Toggle */}
                <div className="flex mb-3 flex-shrink-0">
                  <div className="bg-stone-100 p-0.5 rounded-lg flex w-full">
                    <button
                      onClick={() => setSearchMode("all")}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${searchMode === "all" ? "bg-white text-stone-900 shadow-sm" : "text-stone-500"
                        }`}
                    >
                      All
                    </button>
                    <button
                      onClick={() => setSearchMode("books")}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all flex items-center justify-center gap-1 ${searchMode === "books" ? "bg-white text-stone-900 shadow-sm" : "text-stone-500"
                        }`}
                    >
                      <Book className="w-3 h-3" /> Books
                    </button>
                    <button
                      onClick={() => setSearchMode("papers")}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all flex items-center justify-center gap-1 ${searchMode === "papers" ? "bg-white text-stone-900 shadow-sm" : "text-stone-500"
                        }`}
                    >
                      <FileText className="w-3 h-3" /> Papers
                    </button>
                  </div>
                </div>

                {/* Search - For navigation only */}
                <div className="relative flex-shrink-0 mb-2">
                  <input
                    type="text"
                    value={libraryFilter}
                    onChange={(e) => setLibraryFilter(e.target.value)}
                    placeholder="Find in list..."
                    className="w-full pl-8 pr-7 py-1.5 bg-stone-50 border border-stone-200 rounded-md text-sm placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  />
                  <Search className="absolute left-2.5 top-2 w-3.5 h-3.5 text-stone-400" />
                  {libraryFilter && (
                    <button onClick={() => setLibraryFilter("")} className="absolute right-2 top-2 text-stone-400 hover:text-stone-600">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                {/* Document List */}
                <div className="overflow-y-auto flex-1 space-y-1 pr-1">
                  {(searchMode === "all" || searchMode === "papers") && paperList.filter(filterDoc).length > 0 && (
                    <>
                      {searchMode === "all" && (
                        <div className="text-[10px] font-semibold text-teal-600 uppercase tracking-wider py-1 sticky top-0 bg-white">
                          Papers ({paperList.filter(filterDoc).length})
                        </div>
                      )}
                      {paperList.filter(filterDoc).map((doc) => renderDocButton(doc, null))}
                    </>
                  )}
                  {(searchMode === "all" || searchMode === "books") && bookList.filter(filterDoc).length > 0 && (
                    <>
                      {searchMode === "all" && (
                        <div className="text-[10px] font-semibold text-emerald-600 uppercase tracking-wider py-1 mt-2 sticky top-0 bg-white">
                          Books ({bookList.filter(filterDoc).length})
                        </div>
                      )}
                      {bookList.filter(filterDoc).map((doc) => renderDocButton(doc, null))}
                    </>
                  )}
                  {books.length === 0 && (
                    <div className="text-center text-stone-400 text-sm py-6">No documents yet</div>
                  )}
                  {books.length > 0 && bookList.filter(filterDoc).length === 0 && paperList.filter(filterDoc).length === 0 && (
                    <div className="text-center text-stone-400 text-xs py-4">No matches</div>
                  )}
                </div>

                {/* Compact Footer */}
                <div className="flex-shrink-0 pt-2 mt-2 border-t border-stone-100 text-xs text-stone-500 text-center">
                  {bookList.length} books · {paperList.length} papers
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
                          <ToggleRight className="w-8 h-8 text-emerald-600 mr-2" />
                        ) : (
                          <ToggleLeft className="w-8 h-8 text-gray-400 mr-2" />
                        )}
                        <span>Use Server Defaults</span>
                      </button>
                      <button
                        onClick={() => {
                          setPass1K(BACKEND_DEFAULTS.pass1K);
                          setPass2K(BACKEND_DEFAULTS.pass2K);
                          setPass3Enabled(BACKEND_DEFAULTS.pass3Enabled);
                          setMaxTokens(BACKEND_DEFAULTS.maxTokens);
                        }}
                        className="text-xs text-stone-500 hover:text-stone-700 px-3 py-1 border border-stone-200 rounded-md hover:bg-stone-50 transition-all"
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                  <div
                    className={`grid grid-cols-2 gap-4 transition-opacity duration-300 mb-6 ${useBackendDefaults ? "opacity-50" : "opacity-100"
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
                        className={`w-full accent-emerald-600 ${useBackendDefaults ? "pointer-events-none" : ""
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
                        className={`w-full accent-emerald-600 ${useBackendDefaults ? "pointer-events-none" : ""
                          }`}
                      />
                      <span className="text-gray-900 text-sm">
                        {pass2K} chunks
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="relative flex-1 bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col min-h-0 overflow-hidden">
                <div
                  className="flex-1 min-h-0 overflow-y-auto p-6 space-y-4"
                  onScroll={handleUserScroll}
                >
                  {messages.length === 0 ? (
                    <div className="relative flex items-center justify-center h-full rounded-xl overflow-hidden">
                      {/* Background Graphic - Professional styling */}
                      <div className="absolute inset-0 bg-[url('/image.jpg')] bg-no-repeat bg-center bg-contain opacity-[0.18] pointer-events-none" />
                      {/* Subtle gradient overlay */}
                      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-white/40 pointer-events-none" />
                    </div>
                  ) : (
                    messages.map((msg, idx) => (
                      <SafeErrorBoundary key={idx}>
                        <MessageBubble
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
                      </SafeErrorBoundary>
                    ))
                  )}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="max-w-2xl bg-emerald-50 border border-emerald-200 rounded-2xl p-4 shadow-sm">
                        <div className="flex items-center space-x-3">
                          <Loader className="w-5 h-5 text-emerald-600 animate-spin" />
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
                <div className="p-4 border-t border-stone-200 bg-stone-50">
                  <form onSubmit={handleQuerySubmit} className="flex space-x-3">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      className="flex-1 px-4 py-3 bg-white border border-stone-300 rounded-xl text-stone-800 placeholder-stone-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
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
                        className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-xl hover:from-emerald-600 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-md flex items-center space-x-2"
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

// MessageBubble Component with Defensive Rendering & Error Boundary
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
  const [showPipeline, setShowPipeline] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const copyToClipboard = (text, id) => {
    if (!text) return; // Guard against empty text
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
                className="w-full p-3 bg-white border border-stone-300 rounded-lg text-stone-800 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 mb-3"
                rows={4}
              />
              <div className="flex space-x-2">
                <button
                  onClick={() => onSubmitEdit(index)}
                  disabled={isLoading}
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 text-sm font-medium transition-all"
                >
                  Resend
                </button>
                <button
                  onClick={onCancelEdit}
                  className="px-4 py-2 bg-stone-200 text-stone-700 rounded-lg hover:bg-stone-300 text-sm font-medium transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="group">
              <div className="bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-2xl px-5 py-3 shadow-md">
                <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                  {message.content || ""}
                </p>
              </div>
              <button
                onClick={() => onEdit(index, message.content)}
                className="mt-2 text-xs text-stone-500 hover:text-emerald-600 transition-colors opacity-0 group-hover:opacity-100 flex items-center space-x-1"
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

  // Helper to ensure pipeline_stages is an array
  const validPipelineStages = Array.isArray(message.pipeline_stages)
    ? message.pipeline_stages
    : [];
  // Helper to ensure sources is an array of objects
  const validSources = Array.isArray(message.sources)
    ? message.sources.filter((s) => s && typeof s === "object")
    : [];

  return (
    <div className="flex justify-start" id={id}>
      <div className="max-w-3xl w-full">
        <div className="bg-stone-50 border border-stone-200 rounded-2xl p-5 shadow-sm animate-slide">
          {message.answered_from_history && (
            <div className="mb-3 flex items-center space-x-2 text-xs text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200 w-fit">
              <History className="w-3.5 h-3.5" />
              <span className="font-medium">
                Answered from conversation history
              </span>
            </div>
          )}

          {message.resolved_query &&
            message.resolved_query !== message.content && (
              <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="text-xs font-semibold text-blue-700 mb-1 flex items-center">
                  <Repeat className="w-3.5 h-3.5 mr-1" />
                  Resolved Query
                </div>
                <p className="text-sm text-gray-700 italic">
                  {message.resolved_query}
                </p>
              </div>
            )}

          <div className="prose prose-sm max-w-none text-gray-800 leading-relaxed">
            {/* Fix: Fallback to empty string if content is null/undefined to prevent crash */}
            <ReactMarkdown>
              {typeof message.content === "string"
                ? message.content
                : String(message.content || "")}
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

            {validSources.length > 0 && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-xs text-gray-500 hover:text-blue-600 transition-colors flex items-center space-x-1"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>{validSources.length} Sources</span>
                {showSources ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            )}

            {validPipelineStages.length > 0 && (
              <button
                onClick={() => setShowPipeline(!showPipeline)}
                className="text-xs text-gray-500 hover:text-blue-600 transition-colors flex items-center space-x-1"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Pipeline</span>
                {showPipeline ? (
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

          {/* PIPELINE DISPLAY */}
          {/* Fix: Ensure pipeline_stages is actually an array before rendering */}
          {showPipeline && validPipelineStages.length > 0 && (
            <SafeErrorBoundary>
              <EnhancedPipelineDisplay
                stages={validPipelineStages}
                stats={message.stats}
              />
            </SafeErrorBoundary>
          )}

          {showSources && validSources.length > 0 && (
            <div className="mt-4 space-y-2">
              <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
                Sources
              </h4>
              {validSources.map((source, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-white border border-gray-200 rounded-lg text-xs"
                >
                  <div className="font-semibold text-gray-900 mb-1">
                    {source.title || "Unknown Title"} - Page{" "}
                    {source.page || "?"}
                  </div>
                  <div className="text-gray-600 leading-relaxed">
                    {/* Fix: Defensive check for source.content */}
                    {(source.content || "").substring(0, 200)}...
                  </div>
                  {(source.relevance || source.score) && (
                    <div className="mt-2 text-gray-500">
                      Relevance:{" "}
                      {((source.relevance || source.score) * 100).toFixed(1)}%
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

  return (
    <div className="mt-3 space-y-3 bg-gray-50 p-4 rounded-lg border border-gray-200">
      <div className="flex items-center justify-between text-sm bg-white p-3 rounded border border-gray-200 shadow-sm">
        <div className="flex items-center space-x-4">
          <span className="text-gray-900 font-semibold">
            {stats?.pass1 || 0}
          </span>
          <span className="text-gray-400">→</span>
          <span className="text-gray-900 font-semibold">
            {stats?.pass2 || 0}
          </span>
          <span className="text-gray-400">→</span>
          <span className="text-green-600 font-semibold">
            {stats?.final || 0}
          </span>
        </div>
        {stats?.tokens && (
          <span className="text-gray-500 text-xs">{stats.tokens} tokens</span>
        )}
      </div>
      {stages.map((stage, index) => {
        const isExpanded = expandedStage === index;
        const hasChunks = stage.chunks && stage.chunks.length > 0;
        return (
          <div key={index} className="relative">
            {index < stages.length - 1 && (
              <div className="absolute left-6 top-full h-3 w-0.5 bg-gray-300" />
            )}
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
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
                  <div className="text-white font-semibold text-sm">
                    {stage.stage_name} ({stage.chunk_count})
                  </div>
                </div>
                {hasChunks && (
                  <ChevronDown
                    className={`w-4 h-4 text-white transition-transform ${isExpanded ? "rotate-180" : ""
                      }`}
                  />
                )}
              </div>
              {isExpanded && hasChunks && (
                <div className="p-3 bg-gray-100 max-h-64 overflow-y-auto space-y-2 border-t border-gray-200">
                  {stage.chunks.slice(0, 5).map((chunk, i) => (
                    <div
                      key={i}
                      className="bg-white border border-gray-200 rounded p-2 text-xs shadow-sm"
                    >
                      <div className="font-semibold text-gray-900 flex justify-between">
                        <span>{chunk.book_title || "Unknown Book"}</span>
                        <span className="bg-blue-100 text-blue-700 px-1.5 rounded">
                          {(chunk.relevance || 0).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-gray-600 mt-1">
                        {(chunk.content_preview || "").substring(0, 100)}...
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
