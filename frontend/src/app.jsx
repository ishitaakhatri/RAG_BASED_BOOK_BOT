// frontend/src/App.jsx
import React, { useState, useRef, useEffect } from 'react';
import { Search, BookOpen, Upload, Settings, MessageSquare, FileText, Code, Sparkles, AlertCircle, CheckCircle, Loader, Book, ChevronDown, ChevronUp } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export default function RAGBookBot() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState('all');
  const [topK, setTopK] = useState(5);
  const [showSettings, setShowSettings] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const messagesEndRef = useRef(null);

  // Pipeline settings
  const [pass1K, setPass1K] = useState(50);
  const [pass2K, setPass2K] = useState(15);
  const [pass3Enabled, setPass3Enabled] = useState(true);
  const [maxTokens, setMaxTokens] = useState(2500);

  useEffect(() => {
    fetchBooks();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchBooks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/books`);
      const data = await response.json();
      setBooks(data.books || []);
    } catch (error) {
      console.error('Failed to fetch books:', error);
    }
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setQuery('');

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          book_filter: selectedBook === 'all' ? null : selectedBook,
          top_k: topK,
          pass1_k: pass1K,
          pass2_k: pass2K,
          pass3_enabled: pass3Enabled,
          max_tokens: maxTokens
        })
      });

      const data = await response.json();

      if (data.error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${data.error}`,
          error: true
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          stats: data.stats,
          confidence: data.confidence
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Failed to get response: ${error.message}`,
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('book_title', uploadFile.name.replace('.pdf', ''));
    formData.append('author', 'Unknown');

    setUploadProgress({ status: 'uploading', message: 'Uploading PDF...' });

    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setUploadProgress({
          status: 'success',
          message: `Successfully ingested ${data.result.total_chunks} chunks!`,
          result: data.result
        });
        fetchBooks();
        setTimeout(() => {
          setShowUpload(false);
          setUploadFile(null);
          setUploadProgress(null);
        }, 3000);
      } else {
        setUploadProgress({
          status: 'error',
          message: data.error || 'Upload failed'
        });
      }
    } catch (error) {
      setUploadProgress({
        status: 'error',
        message: `Upload failed: ${error.message}`
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">RAG Book Bot</h1>
                <p className="text-sm text-purple-200">AI-Powered Book Assistant</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowUpload(!showUpload)}
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
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            {/* Books List */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
                <Book className="w-5 h-5 mr-2" />
                Available Books
              </h3>
              <div className="space-y-2">
                <button
                  onClick={() => setSelectedBook('all')}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-all ${
                    selectedBook === 'all'
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/5 text-purple-200 hover:bg-white/10'
                  }`}
                >
                  All Books
                </button>
                {books.map((book, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedBook(book)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-all truncate ${
                      selectedBook === book
                        ? 'bg-purple-600 text-white'
                        : 'bg-white/5 text-purple-200 hover:bg-white/10'
                    }`}
                    title={book}
                  >
                    {book}
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
                  <span className="font-semibold text-white">{books.length}</span>
                </div>
                <div className="flex justify-between text-purple-200">
                  <span>Queries:</span>
                  <span className="font-semibold text-white">{messages.filter(m => m.role === 'user').length}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="lg:col-span-3 space-y-4">
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
                    <span className="text-white text-sm">{maxTokens} tokens</span>
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

            {/* Upload Panel */}
            {showUpload && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Upload className="w-5 h-5 mr-2" />
                  Upload New Book
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">
                      Select PDF File
                    </label>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
                    />
                  </div>
                  {uploadProgress && (
                    <div className={`p-4 rounded-lg ${
                      uploadProgress.status === 'success' ? 'bg-green-500/20 border border-green-500/50' :
                      uploadProgress.status === 'error' ? 'bg-red-500/20 border border-red-500/50' :
                      'bg-blue-500/20 border border-blue-500/50'
                    }`}>
                      <div className="flex items-center space-x-2 text-white">
                        {uploadProgress.status === 'uploading' && <Loader className="w-5 h-5 animate-spin" />}
                        {uploadProgress.status === 'success' && <CheckCircle className="w-5 h-5" />}
                        {uploadProgress.status === 'error' && <AlertCircle className="w-5 h-5" />}
                        <span>{uploadProgress.message}</span>
                      </div>
                    </div>
                  )}
                  <button
                    onClick={handleUploadSubmit}
                    disabled={!uploadFile || uploadProgress?.status === 'uploading'}
                    className="w-full px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {uploadProgress?.status === 'uploading' ? 'Uploading...' : 'Upload & Ingest'}
                  </button>
                </div>
              </div>
            )}

            {/* Chat Messages */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 h-[600px] flex flex-col">
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Sparkles className="w-16 h-16 text-purple-400 mb-4" />
                    <h3 className="text-2xl font-semibold text-white mb-2">
                      Welcome to RAG Book Bot
                    </h3>
                    <p className="text-purple-200 max-w-md">
                      Ask questions about your ingested books and get AI-powered answers with citations!
                    </p>
                  </div>
                ) : (
                  messages.map((msg, idx) => (
                    <MessageBubble key={idx} message={msg} />
                  ))
                )}
                {loading && (
                  <div className="flex items-center space-x-2 text-purple-200">
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Thinking...</span>
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
                    onKeyPress={(e) => e.key === 'Enter' && handleQuerySubmit(e)}
                    placeholder="Ask a question about your books..."
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
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg px-4 py-3">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-3xl bg-white/10 backdrop-blur-lg border border-white/20 rounded-lg px-4 py-3 text-white">
        {message.error ? (
          <div className="flex items-start space-x-2 text-red-300">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{message.content}</p>
          </div>
        ) : (
          <>
            <p className="whitespace-pre-wrap mb-3">{message.content}</p>
            
            {message.confidence && (
              <div className="text-sm text-purple-200 mb-2">
                Confidence: {(message.confidence * 100).toFixed(1)}%
              </div>
            )}

            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/20">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-colors"
                >
                  {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  <span>{message.sources.length} Sources</span>
                </button>
                
                {showSources && (
                  <div className="mt-2 space-y-2">
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="bg-white/5 rounded p-2 text-sm">
                        <div className="flex items-center space-x-2 text-purple-200">
                          <FileText className="w-4 h-4" />
                          <span>{source.chapter}</span>
                          {source.relevance && (
                            <span className="ml-auto text-xs">
                              {source.relevance.toFixed(1)}% relevant
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {message.stats && (
              <div className="mt-3 pt-3 border-t border-white/20 text-xs text-purple-300 space-y-1">
                <div>Pipeline: {message.stats.pass1} → {message.stats.pass2} → {message.stats.final} chunks</div>
                <div>Tokens: {message.stats.tokens}</div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}