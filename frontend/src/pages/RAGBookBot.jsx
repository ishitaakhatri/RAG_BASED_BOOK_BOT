//RAGBookBot.jsx
import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "tailwindcss/tailwind.css";


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
} from "lucide-react";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = "http://localhost:8000";

export default function RAGBookBot() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState("all");
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef(null);

  // SESSION MANAGEMENT
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [showSessions, setShowSessions] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Pipeline settings
  const [pass1K, setPass1K] = useState(50);
  const [pass2K, setPass2K] = useState(15);
  const [pass3Enabled, setPass3Enabled] = useState(true);
  const [maxTokens, setMaxTokens] = useState(2500);
  const navigate = useNavigate();

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


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
      const response = await fetch(`${API_BASE_URL}/sessions?limit=50`);
      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/conversation/${sessionId}`);
      const data = await response.json();

      const loadedMessages = [];
      for (const turn of data.turns) {
        loadedMessages.push({
          role: "user",
          content: turn.user_query,
        });
        loadedMessages.push({
          role: "assistant",
          content: turn.assistant_response,
          sources: turn.sources_used
            ? turn.sources_used.map((id) => ({ chunk_id: id }))
            : [],
          stats: {},
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

    if (!confirm("Are you sure you want to delete this conversation?")) {
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/conversation/${sessionId}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        await fetchSessions();

        if (sessionId === currentSessionId) {
          startNewChat();
        }
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
      const response = await fetch(
        `${API_BASE_URL}/search/sessions?query=${encodeURIComponent(
          searchQuery
        )}&limit=10`
      );
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          session_id: currentSessionId,
          book_filter: selectedBook === "all" ? null : selectedBook,
          top_k: 5,
          pass1_k: pass1K,
          pass2_k: pass2K,
          pass3_enabled: pass3Enabled,
          max_tokens: maxTokens,
        }),
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

        if (!currentSessionId && data.session_id) {
          setCurrentSessionId(data.session_id);
        }

        await fetchSessions();
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Failed to get response: ${error.message}`,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
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

  // Show Chat View
  return (
    <div className="h-screen  bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* SESSIONS SIDEBAR */}
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

        {/* Search Bar */}
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

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {" "}
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

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col h-full min-h-0">
        {/* Header */}
        <header className="bg-black/20 backdrop-blur-lg border-b border-white/10">
          <div className="px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setShowSessions(!showSessions)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-all text-white"
                >
                  <History className="w-5 h-5" />
                </button>
                <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                  <BookOpen className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    RAG Book Bot
                  </h1>
                  <p className="text-sm text-purple-200">
                    {currentSessionId
                      ? "Conversation with Memory"
                      : "Start New Conversation"}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => navigate("/ingest")}
                  className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all"
                >
                  <Upload className="w-4 h-4" />
                  <span>Upload Book</span>
                </button>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all"
                >
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </header>{" "}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full min-h-0">
          {" "}
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4 overflow-hidden">
            {/* Books List */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
                <Book className="w-5 h-5 mr-2" />
                Available Books
              </h3>
              <div className="space-y-2">
                <button
                  onClick={() => setSelectedBook("all")}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-all ${
                    selectedBook === "all"
                      ? "bg-purple-600 text-white"
                      : "bg-white/5 text-purple-200 hover:bg-white/10"
                  }`}
                >
                  All Books
                </button>
                {books.map((book, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedBook(book.title)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-all truncate ${
                      selectedBook === book.title
                        ? "bg-purple-600 text-white"
                        : "bg-white/5 text-purple-200 hover:bg-white/10"
                    }`}
                    title={`${book.title} by ${book.author}`}
                  >
                    <div className="text-sm font-semibold">{book.title}</div>
                    <div className="text-xs opacity-75">by {book.author}</div>
                    {book.total_chunks > 0 && (
                      <div className="text-xs opacity-60 mt-1">
                        {book.total_chunks} chunks
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-3">Stats</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between text-purple-200">
                  <span>Total Books:</span>
                  <span className="font-semibold text-white">
                    {books.length}
                  </span>
                </div>
                <div className="flex justify-between text-purple-200">
                  <span>Sessions:</span>
                  <span className="font-semibold text-white">
                    {sessions.length}
                  </span>
                </div>
                <div className="flex justify-between text-purple-200">
                  <span>Current Queries:</span>
                  <span className="font-semibold text-white">
                    {messages.filter((m) => m.role === "user").length}
                  </span>
                </div>
              </div>
            </div>
          </div>
          {/* Main Chat Area */}
          <div className="lg:col-span-3 space-y-4 flex flex-col h-full min-h-0">
            {/* Settings Panel */}
            {showSettings && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  5-Pass Retrieval Settings
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">
                      Pass 1: Initial Candidates
                    </label>
                    <input
                      type="range"
                      min="30"
                      max="100"
                      value={pass1K}
                      onChange={(e) => setPass1K(parseInt(e.target.value))}
                      className="w-full"
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
                      value={pass2K}
                      onChange={(e) => setPass2K(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="text-white text-sm">{pass2K} chunks</span>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">
                      Max Context Tokens
                    </label>
                    <input
                      type="range"
                      min="1500"
                      max="4000"
                      step="100"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="text-white text-sm">
                      {maxTokens} tokens
                    </span>
                  </div>
                  <div>
                    <label className="flex items-center space-x-2 text-purple-200 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={pass3Enabled}
                        onChange={(e) => setPass3Enabled(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span>Enable Multi-Hop (Pass 3)</span>
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Chat Messages */}
            <div className="flex-1 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 flex flex-col min-h-0">
              <div
                className="flex-1 min-h-0 overflow-y-auto p-6 space-y-4 chat-messages scrollbar-thin"
                onScroll={handleUserScroll}
              >
                {" "}
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Sparkles className="w-16 h-16 text-purple-400 mb-4" />
                    <h3 className="text-2xl font-semibold text-white mb-2">
                      {currentSessionId
                        ? "Continue Your Conversation"
                        : "Welcome to RAG Book Bot"}
                    </h3>
                    <p className="text-purple-200 max-w-md">
                      {currentSessionId
                        ? "Ask follow-up questions - I remember our conversation!"
                        : "Ask questions about your ingested books and get AI-powered answers with memory!"}
                    </p>
                    <p className="text-purple-300 text-sm mt-4">
                      💡 Tip: Your conversations are saved automatically
                    </p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <MessageBubble key={idx} message={msg} id={`msg-${idx}`} />
                  ))
                )}
                {loading && (
                  <div className="flex items-center space-x-2 text-purple-200">
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Thinking with context from our conversation...</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 border-t border-white/20">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) =>
                      e.key === "Enter" && handleQuerySubmit(e)
                    }
                    placeholder={
                      currentSessionId
                        ? "Ask a follow-up question..."
                        : "Ask a question about your books..."
                    }
                    className="flex-1 px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-purple-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    disabled={loading}
                  />
                  <button
                    onClick={handleQuerySubmit}
                    disabled={loading || !query.trim()}
                    className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    <Search className="w-5 h-5" />
                  </button>
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
  );
}

function MessageBubble({ message, id }) {
  const [showSources, setShowSources] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);
  const [showRewrittenQueries, setShowRewrittenQueries] = useState(false);

  if (message.role === "user") {
    return (
      <div id={id} className="flex justify-end">
        <div className="max-w-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg px-4 py-3">
          <p className="whitespace-pre-wrap">{message.content}</p>
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
            {/* Memory Indicator */}
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

            {/* Rewritten Queries Section */}
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

            {/* Answer Section */}
            <div className="bg-white/5 rounded-lg p-4 mb-3 border border-white/10">
              <div className="flex items-center space-x-2 mb-2">
                <MessageSquare className="w-5 h-5 text-purple-300" />
                <h4 className="font-semibold text-purple-100">Answer</h4>
              </div>
              <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/50 max-w-none text-gray-100">
                <ReactMarkdown
                  components={{
                    code: ({ node, inline, className, children, ...props }) => {
                      return (
                        <code
                          className={`${className} ${
                            inline
                              ? "bg-white/10 rounded px-1 py-0.5"
                              : "block bg-black/30 p-4 rounded-lg overflow-x-auto"
                          }`}
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>

            {/* Pipeline Visualization */}
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

            {/* Sources Section */}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-colors w-full justify-between bg-white/5 p-3 rounded-lg border border-white/10 hover:border-purple-400/50"
                >
                  <div className="flex items-center space-x-2">
                    <Book className="w-4 h-4" />
                    <span className="font-semibold">
                      {message.sources.length} Source
                      {message.sources.length > 1 ? "s" : ""}
                    </span>
                  </div>
                  {showSources ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                {showSources && (
                  <div className="mt-2 space-y-2">
                    {message.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="bg-white/5 rounded-lg p-3 border border-white/10"
                      >
                        <div className="flex items-start space-x-3">
                          <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded p-2 flex-shrink-0">
                            <Book className="w-4 h-4 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-semibold text-white mb-1">
                              {source.book_title}
                            </div>
                            {source.author && (
                              <div className="text-xs text-purple-300 mb-2">
                                by {source.author}
                              </div>
                            )}
                            <div className="text-xs text-purple-200 space-y-1">
                              <div className="flex items-center space-x-2">
                                <FileText className="w-3 h-3" />
                                <span>{source.chapter}</span>
                              </div>
                              {source.page && <div>Page {source.page}</div>}
                              {source.relevance && (
                                <div className="flex items-center space-x-2 mt-2">
                                  <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-gradient-to-r from-green-400 to-green-500 rounded-full"
                                      style={{ width: `${source.relevance}%` }}
                                    />
                                  </div>
                                  <span className="text-green-300 text-xs font-mono">
                                    {source.relevance.toFixed(0)}%
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                          <span
                            className={`px-2 py-1 rounded text-xs flex-shrink-0 ${
                              source.type === "code"
                                ? "bg-green-500/30 text-green-200"
                                : "bg-blue-500/30 text-blue-200"
                            }`}
                          >
                            {source.type}
                          </span>
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

  const getStageColor = (index) => {
    const colors = [
      "from-blue-500 to-blue-600",
      "from-purple-500 to-purple-600",
      "from-pink-500 to-pink-600",
      "from-green-500 to-green-600",
    ];
    return colors[index % colors.length];
  };

  const getChangeIndicator = (currentCount, previousCount) => {
    if (previousCount === null) return null;

    const diff = currentCount - previousCount;
    if (diff > 0) {
      return <span className="text-green-300 text-xs ml-2">(+{diff})</span>;
    } else if (diff < 0) {
      return <span className="text-orange-300 text-xs ml-2">({diff})</span>;
    }
    return null;
  };

  return (
    <div className="mt-3 space-y-3 bg-black/20 p-4 rounded-lg">
      {/* Flow Summary */}
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

      {/* Stages */}
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
                  {index === 3 && "Final optimization with deduplication."}
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
