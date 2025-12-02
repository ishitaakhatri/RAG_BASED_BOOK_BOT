// frontend/src/App.jsx - ENHANCED TO SHOW BOOK TITLES WITH CHUNKS
import React, { useState, useRef, useEffect } from 'react';
import { 
  Search, BookOpen, Upload, Settings, MessageSquare, FileText, Code, 
  Sparkles, AlertCircle, CheckCircle, Loader, Book, ChevronDown, 
  ChevronUp, Eye, Filter, Layers, Tag
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

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

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    const formData = new FormData();
    formData.append('file', uploadFile);
    // Book title and author will be auto-extracted from filename

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
          message: `Successfully ingested "${data.result.title}" by ${data.result.author}! Created ${data.result.total_chunks} chunks.`,
          result: data.result
        });
        fetchBooks();
        setTimeout(() => {
          setShowUpload(false);
          setUploadFile(null);
          setUploadProgress(null);
        }, 4000);
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
                <p className="text-sm text-purple-200">AI-Powered Book Assistant with 5-Pass Retrieval</p>
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
                    onClick={() => setSelectedBook(book.title)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-all truncate ${
                      selectedBook === book.title
                        ? 'bg-purple-600 text-white'
                        : 'bg-white/5 text-purple-200 hover:bg-white/10'
                    }`}
                    title={`${book.title} by ${book.author}`}
                  >
                    <div className="text-sm font-semibold">{book.title}</div>
                    <div className="text-xs opacity-75">by {book.author}</div>
                    {book.total_chunks > 0 && (
                      <div className="text-xs opacity-60 mt-1">{book.total_chunks} chunks</div>
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
                      Select PDF File (Format: "Book Title - Author Name.pdf")
                    </label>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setUploadFile(e.target.files[0])}
                      className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
                    />
                    {uploadFile && (
                      <p className="text-xs text-purple-300 mt-2">
                        📄 Selected: {uploadFile.name}
                      </p>
                    )}
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
                      Ask questions about your ingested books and get AI-powered answers with full pipeline visibility!
                    </p>
                    <p className="text-purple-300 text-sm mt-4">
                      💡 Tip: Upload PDFs in format "Book Title - Author Name.pdf" for auto-extraction
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
  const [showPipeline, setShowPipeline] = useState(false);

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
      <div className="max-w-3xl bg-white/10 backdrop-blur-lg border border-white/20 rounded-lg px-4 py-3 text-white w-full">
        {message.error ? (
          <div className="flex items-start space-x-2 text-red-300">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{message.content}</p>
          </div>
        ) : (
          <>
            {/* Answer Section */}
            <div className="bg-white/5 rounded-lg p-4 mb-3 border border-white/10">
              <div className="flex items-center space-x-2 mb-2">
                <MessageSquare className="w-5 h-5 text-purple-300" />
                <h4 className="font-semibold text-purple-100">Answer</h4>
              </div>
              <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/50 max-w-none text-gray-100">
                <ReactMarkdown
                  components={{
                    // Optional: Override specific elements if needed
                    code: ({node, inline, className, children, ...props}) => {
                      return (
                        <code
                          className={`${className} ${inline ? 'bg-white/10 rounded px-1 py-0.5' : 'block bg-black/30 p-4 rounded-lg overflow-x-auto'}`}
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>

            {/* NEW: Enhanced Pipeline Visualization */}
            {message.pipeline_stages && message.pipeline_stages.length > 0 && (
              <div className="mb-3">
                <button
                  onClick={() => setShowPipeline(!showPipeline)}
                  className="flex items-center space-x-2 text-sm text-purple-300 hover:text-purple-100 transition-colors mb-2 w-full justify-between bg-white/5 p-3 rounded-lg border border-white/10 hover:border-purple-400/50"
                >
                  <div className="flex items-center space-x-2">
                    <Layers className="w-4 h-4" />
                    <span className="font-semibold">
                      Retrieval Pipeline ({message.stats?.final || 0} final chunks)
                    </span>
                  </div>
                  {showPipeline ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
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
                    <span className="font-semibold">{message.sources.length} Source{message.sources.length > 1 ? 's' : ''}</span>
                  </div>
                  {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                
                {showSources && (
                  <div className="mt-2 space-y-2">
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="bg-white/5 rounded-lg p-3 border border-white/10">
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
                              {source.page && (
                                <div>Page {source.page}</div>
                              )}
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
                          <span className={`px-2 py-1 rounded text-xs flex-shrink-0 ${
                            source.type === 'code' 
                              ? 'bg-green-500/30 text-green-200' 
                              : 'bg-blue-500/30 text-blue-200'
                          }`}>
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

// ADD this new component for the enhanced pipeline display
function EnhancedPipelineDisplay({ stages, stats }) {
  const [expandedStage, setExpandedStage] = useState(null);
  const [showingChunks, setShowingChunks] = useState({});

  const getStageColor = (index) => {
    const colors = [
      'from-blue-500 to-blue-600',
      'from-purple-500 to-purple-600',
      'from-pink-500 to-pink-600',
      'from-green-500 to-green-600'
    ];
    return colors[index % colors.length];
  };

  const getChangeIndicator = (currentCount, previousCount) => {
    if (previousCount === null) return null;
    
    const diff = currentCount - previousCount;
    if (diff > 0) {
      return (
        <span className="text-green-300 text-xs ml-2">
          (+{diff})
        </span>
      );
    } else if (diff < 0) {
      return (
        <span className="text-orange-300 text-xs ml-2">
          ({diff})
        </span>
      );
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
          <span className="text-green-300 font-semibold">{stats?.final || 0}</span>
        </div>
        {stats?.tokens && (
          <span className="text-purple-200 text-xs">
            {stats.tokens} tokens
          </span>
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
                className={`bg-gradient-to-r ${getStageColor(index)} p-3 flex items-center justify-between cursor-pointer`}
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
                        setShowingChunks(prev => ({ ...prev, [index]: !prev[index] }));
                      }}
                      className="px-2 py-1 bg-white/20 hover:bg-white/30 rounded text-xs text-white"
                    >
                      {showingChunks[index] ? 'Hide' : 'View'}
                    </button>
                  )}
                  {isExpanded ? <ChevronUp className="w-4 h-4 text-white" /> : <ChevronDown className="w-4 h-4 text-white" />}
                </div>
              </div>

              {isExpanded && (
                <div className="p-3 bg-black/20 text-xs text-purple-200">
                  {index === 0 && "Broad semantic search using vector similarity across all chunks."}
                  {index === 1 && "Precision ranking with cross-encoder to select most relevant."}
                  {index === 2 && "Intelligent expansion following related concepts."}
                  {index === 3 && "Final optimization with deduplication."}
                </div>
              )}

              {showingChunks[index] && hasChunks && (
                <div className="p-3 bg-black/30 max-h-64 overflow-y-auto space-y-2">
                  {stage.chunks.slice(0, 5).map((chunk, i) => (
                    <div key={i} className="bg-white/5 rounded p-2 text-xs">
                      <div className="font-semibold text-white mb-1 flex items-center justify-between">
                        <span>{chunk.book_title}</span>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                          chunk.type === 'code' ? 'bg-green-500/30 text-green-200' : 'bg-blue-500/30 text-blue-200'
                        }`}>
                          {chunk.type}
                        </span>
                      </div>
                      <div className="text-purple-300 text-[11px] mb-1">
                        {chunk.chapter} • {chunk.relevance?.toFixed(0)}% relevant
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