import React, { useState, useRef, useEffect } from 'react';
import { 
  Search, BookOpen, Upload, Settings, MessageSquare, FileText, Code, 
  Sparkles, AlertCircle, CheckCircle, Loader, Book, ChevronDown, 
  ChevronUp, Eye, Filter, Layers, Tag, X, Menu, BarChart3, Zap,
  BookMarked, TrendingUp, Clock, ChevronRight
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = 'http://localhost:8000';

export default function RAGBookBot() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [books, setBooks] = useState([]);
  const [selectedBook, setSelectedBook] = useState('all');
  const [bookSearch, setBookSearch] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);

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

  const handleQuerySubmit = async () => {
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
          pass1_k: pass1K,
          pass2_k: pass2K,
          pass3_enabled: pass3Enabled,
          max_tokens: maxTokens
        })
      });

      const data = await response.json();

      if (data.error || data.detail) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${data.error || data.detail}`,
          error: true
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          stats: data.stats,
          confidence: data.confidence,
          pipeline_stages: data.pipeline_stages
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

  const handleUploadSubmit = async () => {
    if (!uploadFile) return;

    const formData = new FormData();
    formData.append('file', uploadFile);

    setUploadProgress({ status: 'uploading', message: 'Uploading and processing PDF...' });

    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setUploadProgress({
          status: 'success',
          message: `✅ Successfully ingested "${data.result.title}" by ${data.result.author}!`,
          result: data.result
        });
        
        await fetchBooks();
        
        setTimeout(() => {
          setShowUpload(false);
          setUploadFile(null);
          setUploadProgress(null);
        }, 2000);
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };

  const filteredBooks = books.filter(book => 
    book.title.toLowerCase().includes(bookSearch.toLowerCase()) ||
    book.author.toLowerCase().includes(bookSearch.toLowerCase())
  );

  const totalQueries = messages.filter(m => m.role === 'user').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
      <header className="sticky top-0 z-50 bg-slate-900/95 backdrop-blur-xl border-b border-slate-800/50 shadow-2xl">
        <div className="max-w-[1800px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 text-blue-400 transition-colors"
              >
                <Menu className="w-5 h-5" />
              </button>
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl blur-md opacity-75"></div>
                  <div className="relative bg-gradient-to-br from-blue-600 to-cyan-600 p-2.5 rounded-xl">
                    <BookOpen className="w-6 h-6 text-white" />
                  </div>
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    RAG Book Bot
                  </h1>
                  <p className="text-xs text-slate-400">AI-Powered Knowledge Assistant</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="hidden sm:flex items-center space-x-4 px-4 py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center space-x-2">
                  <BookMarked className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-slate-300">{books.length}</span>
                  <span className="text-xs text-slate-500">books</span>
                </div>
                <div className="w-px h-4 bg-slate-700"></div>
                <div className="flex items-center space-x-2">
                  <MessageSquare className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-semibold text-slate-300">{totalQueries}</span>
                  <span className="text-xs text-slate-500">queries</span>
                </div>
              </div>
              
              <button
                onClick={() => setShowUpload(!showUpload)}
                className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-lg transition-all shadow-lg shadow-blue-500/25"
              >
                <Upload className="w-4 h-4" />
                <span className="hidden sm:inline">Upload</span>
              </button>
              
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-blue-400 rounded-lg transition-all border border-slate-700/50"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[1800px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className={`lg:col-span-1 space-y-4 ${sidebarOpen ? 'block' : 'hidden lg:block'}`}>
            <div className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 shadow-2xl overflow-hidden">
              <div className="p-4 border-b border-slate-800/50 bg-gradient-to-r from-slate-800/50 to-slate-900/50">
                <h3 className="text-lg font-semibold text-slate-200 mb-3 flex items-center">
                  <Book className="w-5 h-5 mr-2 text-blue-400" />
                  Library ({books.length})
                </h3>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <input
                    type="text"
                    placeholder="Search books..."
                    value={bookSearch}
                    onChange={(e) => setBookSearch(e.target.value)}
                    className="w-full pl-10 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-slate-300 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-sm"
                  />
                </div>
              </div>
              
              <div className="max-h-[600px] overflow-y-auto p-3 space-y-2">
                <button
                  onClick={() => setSelectedBook('all')}
                  className={`w-full text-left px-4 py-3 rounded-xl transition-all ${
                    selectedBook === 'all'
                      ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/25'
                      : 'bg-slate-800/30 text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">All Books</span>
                    <ChevronRight className={`w-4 h-4 transition-transform ${selectedBook === 'all' ? 'rotate-90' : ''}`} />
                  </div>
                </button>
                
                {filteredBooks.map((book, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedBook(book.title)}
                    className={`w-full text-left px-4 py-3 rounded-xl transition-all group ${
                      selectedBook === book.title
                        ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg shadow-blue-500/25'
                        : 'bg-slate-800/30 text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                    }`}
                    title={`${book.title} by ${book.author}`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className={`text-sm font-semibold truncate ${
                          selectedBook === book.title ? 'text-white' : 'text-slate-200 group-hover:text-slate-100'
                        }`}>
                          {book.title}
                        </div>
                        <div className={`text-xs truncate mt-0.5 ${
                          selectedBook === book.title ? 'text-blue-100' : 'text-slate-500 group-hover:text-slate-400'
                        }`}>
                          by {book.author}
                        </div>
                        {book.total_chunks > 0 && (
                          <div className={`text-xs mt-1 flex items-center gap-1 ${
                            selectedBook === book.title ? 'text-blue-200' : 'text-slate-600'
                          }`}>
                            <Layers className="w-3 h-3" />
                            {book.total_chunks} chunks
                          </div>
                        )}
                      </div>
                      <ChevronRight className={`w-4 h-4 flex-shrink-0 transition-transform ${
                        selectedBook === book.title ? 'rotate-90' : 'opacity-0 group-hover:opacity-50'
                      }`} />
                    </div>
                  </button>
                ))}
                
                {filteredBooks.length === 0 && (
                  <div className="text-center py-8 text-slate-500">
                    <Book className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No books found</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-3 space-y-4">
            {showSettings && (
              <div className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 shadow-2xl overflow-hidden">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-slate-200 flex items-center">
                      <Settings className="w-5 h-5 mr-2 text-blue-400" />
                      Pipeline Configuration
                    </h3>
                    <button
                      onClick={() => setShowSettings(false)}
                      className="p-1 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center justify-between">
                          <span>Pass 1: Broad Search</span>
                          <span className="text-blue-400 font-semibold">{pass1K}</span>
                        </label>
                        <input
                          type="range"
                          min="30"
                          max="100"
                          value={pass1K}
                          onChange={(e) => setPass1K(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-700/50 rounded-lg appearance-none cursor-pointer"
                        />
                        <p className="text-xs text-slate-500 mt-1">Initial candidates from vector search</p>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center justify-between">
                          <span>Pass 2: Precision Rerank</span>
                          <span className="text-cyan-400 font-semibold">{pass2K}</span>
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="30"
                          value={pass2K}
                          onChange={(e) => setPass2K(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-700/50 rounded-lg appearance-none cursor-pointer"
                        />
                        <p className="text-xs text-slate-500 mt-1">Top results after cross-encoder</p>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center justify-between">
                          <span>Max Context Tokens</span>
                          <span className="text-purple-400 font-semibold">{maxTokens}</span>
                        </label>
                        <input
                          type="range"
                          min="1500"
                          max="4000"
                          step="100"
                          value={maxTokens}
                          onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-700/50 rounded-lg appearance-none cursor-pointer"
                        />
                        <p className="text-xs text-slate-500 mt-1">Maximum tokens for LLM context</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center">
                      <label className="flex items-center space-x-3 cursor-pointer group">
                        <div className="relative">
                          <input
                            type="checkbox"
                            checked={pass3Enabled}
                            onChange={(e) => setPass3Enabled(e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-slate-700 rounded-full peer peer-checked:bg-gradient-to-r peer-checked:from-blue-600 peer-checked:to-cyan-600 transition-colors"></div>
                          <div className="absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform peer-checked:translate-x-5"></div>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-slate-300 group-hover:text-slate-200">Enable Multi-Hop</span>
                          <p className="text-xs text-slate-500">Cross-chapter retrieval (Pass 3)</p>
                        </div>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {showUpload && (
              <div className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 shadow-2xl overflow-hidden">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-slate-200 flex items-center">
                      <Upload className="w-5 h-5 mr-2 text-blue-400" />
                      Upload Book
                    </h3>
                    <button
                      onClick={() => setShowUpload(false)}
                      className="p-1 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-3">
                        Select PDF (Format: "Title - Author.pdf")
                      </label>
                      <div className="relative">
                        <input
                          type="file"
                          accept=".pdf"
                          onChange={(e) => setUploadFile(e.target.files[0])}
                          className="hidden"
                          id="file-upload"
                        />
                        <label
                          htmlFor="file-upload"
                          className="flex items-center justify-center w-full px-6 py-4 border-2 border-dashed border-slate-700 rounded-xl hover:border-blue-500/50 transition-colors cursor-pointer bg-slate-800/30 hover:bg-slate-800/50"
                        >
                          <div className="text-center">
                            <Upload className="w-8 h-8 mx-auto mb-2 text-slate-500" />
                            <p className="text-sm text-slate-400">
                              {uploadFile ? uploadFile.name : 'Click to select PDF file'}
                            </p>
                          </div>
                        </label>
                      </div>
                    </div>
                    
                    {uploadProgress && (
                      <div className={`p-4 rounded-xl ${
                        uploadProgress.status === 'success' ? 'bg-green-500/10 border-2 border-green-500/30' :
                        uploadProgress.status === 'error' ? 'bg-red-500/10 border-2 border-red-500/30' :
                        'bg-blue-500/10 border-2 border-blue-500/30'
                      }`}>
                        <div className="flex items-center space-x-3">
                          {uploadProgress.status === 'uploading' && <Loader className="w-5 h-5 animate-spin text-blue-400" />}
                          {uploadProgress.status === 'success' && <CheckCircle className="w-5 h-5 text-green-400" />}
                          {uploadProgress.status === 'error' && <AlertCircle className="w-5 h-5 text-red-400" />}
                          <span className="text-sm text-slate-200">{uploadProgress.message}</span>
                        </div>
                      </div>
                    )}
                    
                    <button
                      onClick={handleUploadSubmit}
                      disabled={!uploadFile || uploadProgress?.status === 'uploading'}
                      className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/25 font-medium"
                    >
                      {uploadProgress?.status === 'uploading' ? 'Processing...' : 'Upload & Process'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 shadow-2xl overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 250px)' }}>
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.length === 0 ? (
                  <EmptyState />
                ) : (
                  messages.map((msg, idx) => (
                    <MessageBubble key={idx} message={msg} />
                  ))
                )}
                {loading && (
                  <div className="flex items-center space-x-3 text-slate-400 bg-slate-800/30 rounded-xl p-4">
                    <Loader className="w-5 h-5 animate-spin text-blue-400" />
                    <span className="text-sm">Analyzing your question...</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="p-4 border-t border-slate-800/50 bg-slate-900/50">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask anything about your books..."
                    className="flex-1 px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
                    disabled={loading}
                  />
                  <button
                    onClick={handleQuerySubmit}
                    disabled={loading || !query.trim()}
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/25 font-medium flex items-center space-x-2"
                  >
                    <Search className="w-5 h-5" />
                    <span className="hidden sm:inline">Search</span>
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

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="relative mb-6">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-3xl blur-3xl opacity-20"></div>
        <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 p-8 rounded-3xl border border-slate-700/50">
          <Sparkles className="w-16 h-16 text-blue-400 mx-auto" />
        </div>
      </div>
      <h3 className="text-2xl font-bold text-slate-200 mb-3">
        Welcome to Your Knowledge Base
      </h3>
      <p className="text-slate-400 max-w-md mb-6">
        Ask questions about your books and get AI-powered answers with complete source attribution and pipeline transparency.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl">
        <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
          <Zap className="w-6 h-6 text-blue-400 mb-2" />
          <p className="text-xs text-slate-400">5-Pass retrieval for precision</p>
        </div>
        <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
          <TrendingUp className="w-6 h-6 text-cyan-400 mb-2" />
          <p className="text-xs text-slate-400">Smart cross-chapter linking</p>
        </div>
        <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
          <BookMarked className="w-6 h-6 text-purple-400 mb-2" />
          <p className="text-xs text-slate-400">Full source attribution</p>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false);
  const [showPipeline, setShowPipeline] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-2xl px-5 py-3 shadow-lg shadow-blue-500/25">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-4xl bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl overflow-hidden shadow-xl w-full">
        {message.error ? (
          <div className="p-5 flex items-start space-x-3 text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{message.content}</p>
          </div>
        ) : (
          <>
            <div className="p-6">
              <div className="flex items-center space-x-2 mb-4">
                <div className="p-1.5 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg">
                  <MessageSquare className="w-4 h-4 text-white" />
                </div>
                <h4 className="font-semibold text-slate-200">Answer</h4>
                {message.confidence && (
                  <span className="ml-auto text-xs text-slate-500">
                    {(message.confidence * 100).toFixed(0)}% confidence
                  </span>
                )}
              </div>
              <div className="prose prose-invert prose-sm max-w-none text-slate-300 prose-headings:text-slate-200 prose-a:text-blue-400 prose-code:text-cyan-400 prose-code:bg-slate-900/50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-slate-900/80 prose-pre:border prose-pre:border-slate-700/50">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
            </div>

            {message.pipeline_stages && message.pipeline_stages.length > 0 && (
              <div className="border-t border-slate-700/50">
                <button
                  onClick={() => setShowPipeline(!showPipeline)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Layers className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-slate-300">
                      Retrieval Pipeline
                    </span>
                    <span className="text-sm text-slate-500">
                      {message.stats?.final || 0} final chunks
                    </span>
                  </div>
                  {showPipeline ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </button>
                
                {showPipeline && (
                  <div className="px-6 pb-6">
                    <PipelineVisualization stages={message.pipeline_stages} stats={message.stats} />
                  </div>
                )}
              </div>
            )}

            {message.sources && message.sources.length > 0 && (
              <div className="border-t border-slate-700/50">
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Book className="w-5 h-5 text-cyan-400" />
                    <span className="font-medium text-slate-300">
                      {message.sources.length} Source{message.sources.length > 1 ? 's' : ''}
                    </span>
                  </div>
                  {showSources ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </button>
                
                {showSources && (
                  <div className="px-6 pb-6 space-y-3">
                    {message.sources.map((source, idx) => (
                      <SourceCard key={idx} source={source} index={idx + 1} />
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

function SourceCard({ source, index }) {
  return (
    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50 hover:border-slate-600/50 transition-colors">
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center text-white font-bold text-sm">
            {index}
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-slate-200 mb-1">{source.book_title}</div>
          {source.author && (
            <div className="text-sm text-slate-400 mb-2">by {source.author}</div>
          )}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center space-x-1 text-slate-500">
              <FileText className="w-3 h-3" />
              <span>{source.chapter}</span>
            </div>
            {source.page && (
              <div className="flex items-center space-x-1 text-slate-500">
                <span>•</span>
                <span>Page {source.page}</span>
              </div>
            )}
            <span className={`px-2 py-1 rounded-md text-xs font-medium ${
              source.type === 'code' 
                ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
                : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
            }`}>
              {source.type}
            </span>
          </div>
          {source.relevance && (
            <div className="mt-3 flex items-center space-x-2">
              <div className="flex-1 h-2 bg-slate-700/50 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full transition-all duration-500"
                  style={{ width: `${source.relevance}%` }}
                />
              </div>
              <span className="text-xs font-mono text-slate-400 w-12 text-right">
                {source.relevance.toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PipelineVisualization({ stages, stats }) {
  return (
    <div className="space-y-4 mt-4">
      <div className="flex items-center justify-between bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center space-x-3 text-sm">
          <span className="text-slate-400">Flow:</span>
          <span className="font-mono text-blue-400 font-semibold">{stats?.pass1 || 0}</span>
          <ChevronRight className="w-4 h-4 text-slate-600" />
          <span className="font-mono text-cyan-400 font-semibold">{stats?.pass2 || 0}</span>
          <ChevronRight className="w-4 h-4 text-slate-600" />
          <span className="font-mono text-purple-400 font-semibold">{stats?.pass3 || 0}</span>
          <ChevronRight className="w-4 h-4 text-slate-600" />
          <span className="font-mono text-green-400 font-semibold">{stats?.final || 0}</span>
        </div>
        {stats?.tokens && (
          <div className="text-xs text-slate-500">
            <span className="font-mono">{stats.tokens}</span> tokens
          </div>
        )}
      </div>

      <div className="space-y-3">
        {stages.map((stage, index) => {
          const colors = [
            'from-blue-500 to-blue-600',
            'from-cyan-500 to-cyan-600',
            'from-purple-500 to-purple-600',
            'from-green-500 to-green-600'
          ];
          
          return (
            <div key={index} className="bg-slate-900/50 rounded-xl border border-slate-700/50 overflow-hidden">
              <div className={`bg-gradient-to-r ${colors[index % colors.length]} px-4 py-3`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 bg-white/20 rounded-lg flex items-center justify-center text-white text-xs font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-medium text-white text-sm">{stage.stage_name}</div>
                      <div className="text-xs text-white/80">{stage.chunk_count} chunks</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}