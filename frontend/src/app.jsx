import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import "tailwindcss/tailwind.css";

import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";

import {
  SignedIn,
  SignedOut,
  useSignIn,
  useSignUp,
  UserButton,
} from "@clerk/clerk-react";

import { Library, BookOpen, Sparkles, Lock, Mail, Eye, EyeOff } from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

// Custom Landing/Auth Page with Clerk Backend
const LandingPage = () => {
  const { signIn, setActive: setActiveSignIn, isLoaded: signInLoaded } = useSignIn();
  const { signUp, setActive: setActiveSignUp, isLoaded: signUpLoaded } = useSignUp();
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({ email: "", password: "", name: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleEmailPasswordSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isSignUp) {
        if (!signUpLoaded) return;

        const result = await signUp.create({
          emailAddress: formData.email,
          password: formData.password,
          firstName: formData.name.split(" ")[0] || "",
          lastName: formData.name.split(" ").slice(1).join(" ") || "",
        });

        await signUp.prepareEmailAddressVerification({ strategy: "email_code" });

        if (result.status === "complete") {
          await setActiveSignUp({ session: result.createdSessionId });
        }
      } else {
        if (!signInLoaded) return;

        const result = await signIn.create({
          identifier: formData.email,
          password: formData.password,
        });

        if (result.status === "complete") {
          await setActiveSignIn({ session: result.createdSessionId });
        }
      }
    } catch (err) {
      console.error("Auth error:", err);
      setError(err.errors?.[0]?.message || "Authentication failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSocialLogin = async (provider) => {
    try {
      if (!signInLoaded) return;

      await signIn.authenticateWithRedirect({
        strategy: provider === "google" ? "oauth_google" : "oauth_github",
        redirectUrl: "/sso-callback",
        redirectUrlComplete: "/",
      });
    } catch (err) {
      console.error("Social login error:", err);
      setError("Social login failed. Please try again.");
    }
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

            {/* RIGHT PANEL - Custom Auth Form */}
            <div className="w-full max-w-md mx-auto">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-3xl blur-2xl" />
                
                <div className="relative bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl p-8">
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

                  {error && (
                    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                      {error}
                    </div>
                  )}

                  <form onSubmit={handleEmailPasswordSubmit} className="space-y-5">
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
                          required={isSignUp}
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
                          required
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
                          type={showPassword ? "text" : "password"}
                          value={formData.password}
                          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                          className="w-full pl-11 pr-12 py-3 bg-slate-950 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                          placeholder="••••••••"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
                          tabIndex={-1}
                        >
                          {showPassword ? (
                            <EyeOff className="w-5 h-5" />
                          ) : (
                            <Eye className="w-5 h-5" />
                          )}
                        </button>
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

                  <div className="mt-6 text-center">
                    <p className="text-slate-400 text-sm">
                      {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
                      <button
                        onClick={() => {
                          setIsSignUp(!isSignUp);
                          setError("");
                          setFormData({ email: "", password: "", name: "" });
                          setShowPassword(false);
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

                  {/* Google Login Only */}
                  <button
                    type="button"
                    onClick={() => handleSocialLogin("google")}
                    disabled={loading}
                    className="w-full py-2.5 px-4 bg-slate-950 border border-white/10 rounded-xl text-sm font-medium text-white hover:bg-slate-800 transition flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24">
                      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                    </svg>
                    <span>Continue with Google</span>
                  </button>
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

// Main App Component - UNCHANGED INTERNAL STRUCTURE
export default function App() {
  const [books, setBooks] = useState([]);

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
    fetchBooks();
  }, []);

  const handleUploadSuccess = () => {
    fetchBooks();
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white overflow-hidden">
      {/* ================= SIGNED OUT - CUSTOM UI ================= */}
      <SignedOut>
        <LandingPage />
      </SignedOut>

      {/* ================= SIGNED IN - ORIGINAL UNCHANGED ================= */}
      <SignedIn>
        {/* Original User Button in top right */}
        <nav className="fixed top-4 right-6 z-50">
          <UserButton
            afterSignOutUrl="/"
            appearance={{
              elements: {
                userButtonAvatarBox: "w-9 h-9",
              },
            }}
          />
        </nav>

        {/* Original Routes - Completely Unchanged */}
        <Routes>
          <Route path="/" element={<RAGBookBot />} />
          <Route
            path="/ingest"
            element={
              <IngestionPage
                books={books}
                onUploadSuccess={handleUploadSuccess}
              />
            }
          />
        </Routes>
      </SignedIn>
    </div>
  );
}
