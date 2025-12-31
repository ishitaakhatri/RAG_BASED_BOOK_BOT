import React, { useState } from "react";
import { Lock, Eye, EyeOff } from "lucide-react";

export default function AdminLogin({ onLoginSuccess }) {
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Simple password (change this to your desired password)
  const ADMIN_PASSWORD = "Rag";

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    // Simulate a small delay
    setTimeout(() => {
      if (password === ADMIN_PASSWORD) {
        // Store auth token in localStorage
        localStorage.setItem("adminAuth", "true");
        onLoginSuccess();
      } else {
        setError("Invalid password. Please try again.");
        setPassword("");
      }
      setLoading(false);
    }, 500);
  };

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center overflow-hidden relative">
      {/* Animated background blobs */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      
      <div className="w-full max-w-md relative z-10">
        {/* Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-8 shadow-2xl hover:shadow-purple-500/20 transition-all duration-300 hover:border-white/30 animate-fade-in">
          {/* Header */}
          <div className="flex justify-center mb-8">
            <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-4 rounded-xl shadow-lg shadow-purple-500/50 hover:shadow-purple-500/80 transition-all duration-300 transform hover:scale-110">
              <Lock className="w-8 h-8 text-white animate-bounce" style={{ animationDelay: '0.2s' }} />
            </div>
          </div>

          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-300 to-pink-300 text-center mb-2 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            Welcome Admin
          </h1>
          <p className="text-purple-200 text-center mb-8 animate-fade-in" style={{ animationDelay: '0.3s' }}>
            Enter your password to access the RAG Knowledge Bot
          </p>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6 animate-fade-in" style={{ animationDelay: '0.4s' }}>
            {/* Password Input */}
            <div className="relative">
              <label className="block text-sm font-semibold text-purple-300 mb-3 flex items-center space-x-2">
                <span>🔐</span>
                <span>Admin Password</span>
              </label>
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg blur opacity-0 group-hover:opacity-20 transition-all duration-300"></div>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    setError("");
                  }}
                  placeholder="Enter your password..."
                  className="relative w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-purple-300/50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:bg-white/10 focus:border-purple-400 transition-all duration-300"
                  disabled={loading}
                  autoFocus
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-purple-300 hover:text-white hover:scale-125 transition-all duration-200"
                  disabled={loading}
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/20 border border-red-400/50 rounded-lg p-4 flex items-start space-x-3 animate-shake backdrop-blur-sm">
                <div className="w-5 h-5 rounded-full bg-red-500/50 flex items-center justify-center flex-shrink-0 mt-0.5 animate-pulse">
                  <span className="text-red-100 text-xs font-bold">!</span>
                </div>
                <p className="text-red-200 text-sm font-medium">{error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !password.trim()}
              className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 hover:shadow-lg hover:shadow-purple-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center space-x-2 transform hover:scale-105 active:scale-95"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Verifying...</span>
                </>
              ) : (
                <>
                  <Lock className="w-5 h-5" />
                  <span>Unlock Bot</span>
                </>
              )}
            </button>
          </form>

          {/* Footer Info */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <p className="text-center text-xs text-purple-300 animate-fade-in" style={{ animationDelay: '0.5s' }}>
              🔒 This is a secure admin area. <br />
              Only authorized personnel have access.
            </p>
          </div>
        </div>

        {/* Background decoration */}
        <div className="mt-8 text-center animate-fade-in" style={{ animationDelay: '0.6s' }}>
          <div className="inline-block px-4 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 hover:border-white/20 transition-all duration-300 backdrop-blur-sm">
            <p className="text-xs text-purple-300">
              ✨ RAG Knowledge Bot • Admin Portal
            </p>
          </div>
        </div>
      </div>
      
      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          10%, 30%, 50%, 70%, 90% { transform: translateX(-2px); }
          20%, 40%, 60%, 80% { transform: translateX(2px); }
        }
        
        .animate-fade-in {
          animation: fade-in 0.6s ease-out forwards;
          opacity: 0;
        }
        
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }
      `}</style>
    </div>
  );
}