import React, { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import "tailwindcss/tailwind.css";

import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";

import { Library, LogOut, BookOpen, Sparkles, Lock, Mail } from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

// Custom Authentication Hook
const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in (check localStorage/sessionStorage)
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const login = async (email, password) => {
    try {
      // Replace with your actual API call
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
        localStorage.setItem("user", JSON.stringify(userData));
        return { success: true };
      }
      return { success: false, error: "Invalid credentials" };
    } catch (error) {
      console.error("Login failed:", error);
      return { success: false, error: "Login failed" };
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("user");
  };

  return { user, login, logout, loading };
};

// Protected Route Component
const ProtectedRoute = ({ children, user }) => {
  if (!user) {
    return <Navigate to="/" replace />;
  }
  return children;
};

// Enhanced Landing/Auth Page
const LandingPage = ({ onLogin }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({ email: "", password: "", name: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    if (!formData.email || !formData.password) {
      setError("Please fill in all required fields");
      setLoading(false);
      return;
    }

    const result = await onLogin(formData.email, formData.password);
    
    if (!result.success) {
      setError(result.error || "Authentication failed");
    }
    
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-pink-600/20 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-3xl" />
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px]" />

      <div className="relative z-10 flex items-center justify-center min-h-screen px-4 py-12">
        <div className="w-full max-w-6xl">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            
            {/* LEFT PANEL - Enhanced Branding */}
            <div className="flex flex-col justify-center space-y-8">
              {/* Logo Section */}
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl blur-xl opacity-50" />
                  <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center shadow-2xl">
                    <Library className="w-10 h-10 text-white" />
                  </div>
                </div>
                <div>
                  <h1 className="text-5xl font-extrabold bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 text-transparent bg-clip-text">
                    RAG Bot
                  </h1>
                  <p className="text-slate-400 text-sm mt-1">Knowledge at your fingertips</p>
                </div>
              </div>

              {/* Tagline */}
              <div className="space-y-4">
                <h2 className="text-4xl font-bold text-white leading-tight">
                  Your personal AI librarian for
                  <span className="block bg-gradient-to-r from-purple-400 to-pink-400 text-transparent bg-clip-text">
                    research & discovery
                  </span>
                </h2>
                <p className="text-slate-400 text-lg leading-relaxed max-w-md">
                  Transform your documents into interactive knowledge bases. Ask questions,
                  get precise answers, and unlock insights from your research papers and books.
                </p>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 gap-4 max-w-md">
                <FeatureCard 
                  icon={<Sparkles className="w-5 h-5" />}
                  title="AI-Powered Search"
                  description="Semantic understanding of your queries"
                />
                <FeatureCard 
                  icon={<Lock className="w-5 h-5" />}
                  title="Private & Secure"
                  description="Your documents stay confidential"
                />
              </div>
            </div>

            {/* RIGHT PANEL - Modern Auth Form */}
            <div className="w-full max-w-md mx-auto">
              <div className="relative">
                {/* Glow effect behind form */}
                <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-3xl blur-2xl" />
                
                <div className="relative bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl p-8">
                  {/* Form Header */}
                  <div className="text-center mb-8">
                    <h3 className="text-3xl font-bold text-white mb-2">
                      {isSignUp ? "Create Account" : "Welcome Back"}
                    </h3>
                    <p className="text-slate-400 text-sm">
                      {isSignUp 
                        ? "Start your journey with RAG Knowledge Bot" 
                        : "Sign in to access your knowledge library"}
                    </p>
                  </div>

                  {/* Error Message */}
                  {error && (
                    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                      {error}
                    </div>
                  )}

                  {/* Form */}
                  <form onSubmit={handleSubmit} className="space-y-5">
                    {isSignUp && (
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-300 block">
                          Full Name
                        </label>
                        <input
                          type="text"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          className="w-full px-4 py-3 bg-slate-950 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                          placeholder="John Doe"
                        />
                      </div>
                    )}

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-300 block">
                        Email Address
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-500" />
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                          className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                          placeholder="you@example.com"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-300 block">
                        Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-500" />
                        <input
                          type="password"
                          value={formData.password}
                          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                          className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                          placeholder="••••••••"
                        />
                      </div>
                    </div>

                    {!isSignUp && (
                      <div className="flex items-center justify-between text-sm">
                        <label className="flex items-center text-slate-400 cursor-pointer">
                          <input type="checkbox" className="mr-2 rounded" />
                          Remember me
                        </label>
                        <button type="button" className="text-purple-400 hover:text-purple-300 transition">
                          Forgot password?
                        </button>
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full py-3.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-xl font-semibold text-white shadow-lg shadow-purple-500/30 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? "Processing..." : isSignUp ? "Create Account" : "Sign In"}
                    </button>
                  </form>

                  {/* Toggle Sign Up/In */}
                  <div className="mt-6 text-center">
                    <p className="text-slate-400 text-sm">
                      {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
                      <button
                        onClick={() => {
                          setIsSignUp(!isSignUp);
                          setError("");
                        }}
                        className="text-purple-400 hover:text-purple-300 font-semibold transition"
                      >
                        {isSignUp ? "Sign In" : "Sign Up"}
                      </button>
                    </p>
                  </div>

                  {/* Divider */}
                  <div className="relative my-6">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-white/10"></div>
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-slate-900 px-2 text-slate-500">Or continue with</span>
                    </div>
                  </div>

                  {/* Social Login Options */}
                  <div className="grid grid-cols-2 gap-3">
                    <button className="py-2.5 px-4 bg-slate-950 border border-white/10 rounded-xl text-sm font-medium text-white hover:bg-slate-800 transition flex items-center justify-center space-x-2">
                      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                      </svg>
                      <span>Google</span>
                    </button>
                    <button className="py-2.5 px-4 bg-slate-950 border border-white/10 rounded-xl text-sm font-medium text-white hover:bg-slate-800 transition flex items-center justify-center space-x-2">
                      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                      </svg>
                      <span>GitHub</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Feature Card Component
const FeatureCard = ({ icon, title, description }) => (
  <div className="flex items-start space-x-3 p-4 rounded-xl bg-slate-900/50 border border-white/5 backdrop-blur-sm hover:border-purple-500/30 transition-all">
    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-purple-600/20 to-pink-600/20 flex items-center justify-center text-purple-400">
      {icon}
    </div>
    <div>
      <h4 className="font-semibold text-white text-sm mb-1">{title}</h4>
      <p className="text-slate-400 text-xs leading-relaxed">{description}</p>
    </div>
  </div>
);

// Main App Component
export default function App() {
  const [books, setBooks] = useState([]);
  const { user, login, logout, loading } = useAuth();

  const fetchBooks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/books`);
      const data = await response.json();
      setBooks(data.books || []);
    } catch (error) {
      console.error("Failed to fetch books:", error);
    }
  };

  useEffect(() => {
    if (user) {
      fetchBooks();
    }
  }, [user]);

  const handleUploadSuccess = () => {
    fetchBooks();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  if (!user) {
    return <LandingPage onLogin={login} />;
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Top Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center">
              <Library className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 text-transparent bg-clip-text">
              RAG Bot
            </span>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3 px-4 py-2 bg-slate-800/50 rounded-xl border border-white/10">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center text-sm font-semibold">
                {user.name?.charAt(0).toUpperCase() || "U"}
              </div>
              <span className="text-sm text-slate-300">{user.name || user.email}</span>
            </div>
            <button
              onClick={logout}
              className="p-2.5 hover:bg-slate-800 rounded-xl transition-colors group"
              title="Logout"
            >
              <LogOut className="w-5 h-5 text-slate-400 group-hover:text-red-400 transition-colors" />
            </button>
          </div>
        </div>
      </nav>

      <div className="pt-20">
        <Routes>
          <Route
            path="/"
            element={
              <ProtectedRoute user={user}>
                <RAGBookBot />
              </ProtectedRoute>
            }
          />
          <Route
            path="/ingest"
            element={
              <ProtectedRoute user={user}>
                <IngestionPage books={books} onUploadSuccess={handleUploadSuccess} />
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </div>
  );
}
