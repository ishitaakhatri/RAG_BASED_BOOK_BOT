import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import "tailwindcss/tailwind.css";

import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";

import { SignedIn, SignedOut, useSignIn, UserButton, SignIn } from "@clerk/clerk-react";

import {
  Library,
  Sparkles,
  Lock,
  Mail,
  Eye,
  EyeOff,
  KeyRound,
  ArrowLeft,
  BookOpen,
  Zap,
} from "lucide-react";

const API_BASE_URL = "/api";

// Custom Landing/Auth Page with Clerk Backend
const LandingPage = () => {
  const { signIn, setActive, isLoaded } = useSignIn();

  // States
  const [resetStep, setResetStep] = useState(null);
  const [showClerkLogin, setShowClerkLogin] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    code: "",
    newPassword: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isLoaded) return;
    setError("");
    setLoading(true);

    try {
      if (resetStep === "email") {
        await signIn.create({
          strategy: "reset_password_email_code",
          identifier: formData.email,
        });
        setResetStep("code");
        setError("");
      } else if (resetStep === "code") {
        const result = await signIn.attemptFirstFactor({
          strategy: "reset_password_email_code",
          code: formData.code,
          password: formData.newPassword,
        });

        if (result.status === "complete") {
          await setActive({ session: result.createdSessionId });
        } else {
          setError("Verification failed. Please check your code.");
        }
      } else {
        const result = await signIn.create({
          identifier: formData.email,
          password: formData.password,
        });

        if (result.status === "complete") {
          await setActive({ session: result.createdSessionId });
        } else {
          console.log("Login incomplete, switching to Clerk UI. Status:", result.status);
          setShowClerkLogin(true);
        }
      }
    } catch (err) {
      console.error("Auth error:", err);

      const errorCode = err.errors?.[0]?.code;

      if (errorCode === "form_password_pwned") {
        setShowClerkLogin(true);
        return;
      }

      setError(
        err.errors?.[0]?.message || "Authentication failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSocialLogin = async (provider) => {
    try {
      if (!isLoaded) return;

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

  const getTitle = () => {
    if (resetStep === "email") return "Reset Password";
    if (resetStep === "code") return "New Password";
    return "Welcome Back";
  };

  const getSubtitle = () => {
    if (resetStep === "email")
      return "Enter your email to receive a reset code";
    if (resetStep === "code") return "Enter the code sent to your email";
    return "Sign in to access your knowledge library";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-stone-100 to-amber-50/30 text-stone-800 relative overflow-y-auto">
      {/* Subtle Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-emerald-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-20 w-96 h-96 bg-amber-200/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/3 w-64 h-64 bg-teal-200/15 rounded-full blur-3xl" />
      </div>

      {/* Subtle Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(16,185,129,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(16,185,129,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      <div className="relative z-10 flex items-center justify-center min-h-screen px-6 py-16">
        <div className="w-full max-w-6xl">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* LEFT PANEL - Branding */}
            <div className="flex flex-col justify-center space-y-10">
              {/* Logo Section */}
              <div className="flex items-center space-x-5">
                <div className="relative">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/25">
                    <Library className="w-8 h-8 text-white" />
                  </div>
                </div>
                <div>
                  <h1 className="text-5xl font-bold text-stone-800">
                    RAG Bot
                  </h1>
                  <p className="text-stone-500 text-sm mt-1">
                    Knowledge at your fingertips
                  </p>
                </div>
              </div>

              {/* Tagline */}
              <div className="space-y-5">
                <h2 className="text-4xl font-bold text-stone-800 leading-tight">
                  Your personal AI librarian for
                  <span className="block text-emerald-600">
                    research & discovery
                  </span>
                </h2>
                <p className="text-stone-600 text-lg leading-relaxed max-w-lg">
                  Transform your documents into interactive knowledge bases. Ask
                  questions, get precise answers, and unlock insights from your
                  research papers and books.
                </p>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 gap-4 max-w-lg">
                <FeatureCard
                  icon={<Sparkles className="w-5 h-5" />}
                  title="AI-Powered Search"
                  description="Semantic understanding of your queries"
                  accentColor="emerald"
                />
                <FeatureCard
                  icon={<BookOpen className="w-5 h-5" />}
                  title="Multi-Source Knowledge"
                  description="Books, papers, and documents in one place"
                  accentColor="teal"
                />
                <FeatureCard
                  icon={<Lock className="w-5 h-5" />}
                  title="Private & Secure"
                  description="Your documents stay confidential"
                  accentColor="amber"
                />
              </div>
            </div>

            {/* RIGHT PANEL - Auth Form */}
            <div className="w-full max-w-md mx-auto">
              <div className="relative">
                {showClerkLogin ? (
                  // CLERK UI FALLBACK
                  <div className="flex flex-col justify-center items-center bg-white rounded-2xl shadow-xl shadow-stone-200/50 border border-stone-200 p-8 min-h-[500px] relative">
                    <SignIn
                      appearance={{
                        elements: {
                          rootBox: "w-full",
                          card: "bg-transparent shadow-none w-full",
                          headerTitle: "text-stone-800",
                          headerSubtitle: "text-stone-600",
                          socialButtonsBlockButton: "text-stone-700 border-stone-300 hover:bg-stone-50",
                          formFieldLabel: "text-stone-700",
                          formFieldInput: "bg-white border-stone-300 text-stone-800",
                          footerActionText: "text-stone-600",
                          footerActionLink: "text-emerald-600 hover:text-emerald-700",
                          dividerLine: "bg-stone-200",
                          dividerText: "text-stone-500"
                        }
                      }}
                    />
                    <button
                      onClick={() => setShowClerkLogin(false)}
                      className="absolute top-4 right-4 text-stone-400 hover:text-stone-700 transition-colors"
                      title="Return to custom login"
                    >
                      <ArrowLeft className="w-5 h-5" />
                    </button>
                  </div>
                ) : (
                  // CUSTOM UI
                  <div className="relative bg-white rounded-2xl shadow-xl shadow-stone-200/50 border border-stone-200 p-10">
                    <div className="text-center mb-8 relative">
                      {resetStep && (
                        <button
                          onClick={() => {
                            setResetStep(null);
                            setError("");
                            setFormData({
                              email: "",
                              password: "",
                              code: "",
                              newPassword: "",
                            });
                          }}
                          className="absolute left-0 top-1 text-stone-400 hover:text-stone-700 transition"
                          title="Back to Login"
                        >
                          <ArrowLeft className="w-5 h-5" />
                        </button>
                      )}
                      <h3 className="text-3xl font-bold text-stone-800 mb-2">
                        {getTitle()}
                      </h3>
                      <p className="text-stone-500 text-sm">{getSubtitle()}</p>
                    </div>

                    {error && (
                      <div className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm">
                        {error}
                      </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                      {/* EMAIL INPUT */}
                      {(resetStep === null || resetStep === "email") && (
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-stone-700 block">
                            Email Address
                          </label>
                          <div className="relative">
                            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-stone-400" />
                            <input
                              type="email"
                              value={formData.email}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  email: e.target.value,
                                })
                              }
                              className="w-full pl-11 pr-4 py-3 bg-stone-50 border border-stone-300 rounded-xl text-stone-800 placeholder-stone-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 transition-all"
                              placeholder="you@example.com"
                              required
                            />
                          </div>
                        </div>
                      )}

                      {/* PASSWORD INPUT */}
                      {resetStep === null && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <label className="text-sm font-medium text-stone-700 block">
                              Password
                            </label>
                            <button
                              type="button"
                              onClick={() => {
                                setResetStep("email");
                                setError("");
                              }}
                              className="text-sm text-emerald-600 hover:text-emerald-700 transition"
                            >
                              Forgot password?
                            </button>
                          </div>
                          <div className="relative">
                            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-stone-400" />
                            <input
                              type={showPassword ? "text" : "password"}
                              value={formData.password}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  password: e.target.value,
                                })
                              }
                              className="w-full pl-11 pr-12 py-3 bg-stone-50 border border-stone-300 rounded-xl text-stone-800 placeholder-stone-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 transition-all"
                              placeholder="••••••••"
                              required
                            />
                            <button
                              type="button"
                              onClick={() => setShowPassword(!showPassword)}
                              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-stone-400 hover:text-stone-600 transition-colors focus:outline-none"
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
                      )}

                      {/* RESET CODE INPUT */}
                      {resetStep === "code" && (
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-stone-700 block">
                            Reset Code
                          </label>
                          <div className="relative">
                            <KeyRound className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-stone-400" />
                            <input
                              type="text"
                              value={formData.code}
                              onChange={(e) =>
                                setFormData({ ...formData, code: e.target.value })
                              }
                              className="w-full pl-11 pr-4 py-3 bg-stone-50 border border-stone-300 rounded-xl text-stone-800 placeholder-stone-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 transition-all"
                              placeholder="Enter code from email"
                              required
                            />
                          </div>
                        </div>
                      )}

                      {/* NEW PASSWORD INPUT */}
                      {resetStep === "code" && (
                        <div className="space-y-2">
                          <label className="text-sm font-medium text-stone-700 block">
                            New Password
                          </label>
                          <div className="relative">
                            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-stone-400" />
                            <input
                              type={showPassword ? "text" : "password"}
                              value={formData.newPassword}
                              onChange={(e) =>
                                setFormData({
                                  ...formData,
                                  newPassword: e.target.value,
                                })
                              }
                              className="w-full pl-11 pr-12 py-3 bg-stone-50 border border-stone-300 rounded-xl text-stone-800 placeholder-stone-400 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 transition-all"
                              placeholder="New secure password"
                              required
                            />
                            <button
                              type="button"
                              onClick={() => setShowPassword(!showPassword)}
                              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-stone-400 hover:text-stone-600 transition-colors focus:outline-none"
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
                      )}

                      <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-3.5 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 rounded-xl font-semibold text-white shadow-lg shadow-emerald-500/25 hover:shadow-xl hover:shadow-emerald-500/30 transition-all transform hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loading
                          ? "Processing..."
                          : resetStep === "email"
                            ? "Send Reset Code"
                            : resetStep === "code"
                              ? "Reset & Sign In"
                              : "Sign In"}
                      </button>
                    </form>

                    {/* Divider & Social Login */}
                    {resetStep === null && (
                      <>
                        <div className="relative my-6">
                          <div className="absolute inset-0 flex items-center">
                            <div className="w-full border-t border-stone-200"></div>
                          </div>
                          <div className="relative flex justify-center text-xs uppercase">
                            <span className="bg-white px-2 text-stone-500">
                              Or continue with
                            </span>
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => handleSocialLogin("google")}
                          disabled={loading}
                          className="w-full py-2.5 px-4 bg-white border border-stone-300 rounded-xl text-sm font-medium text-stone-700 hover:bg-stone-50 hover:border-stone-400 transition flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                        >
                          <svg className="w-5 h-5" viewBox="0 0 24 24">
                            <path
                              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                              fill="#4285F4"
                            />
                            <path
                              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                              fill="#34A853"
                            />
                            <path
                              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                              fill="#FBBC05"
                            />
                            <path
                              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                              fill="#EA4335"
                            />
                          </svg>
                          <span>Continue with Google</span>
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Feature Card Component
const FeatureCard = ({ icon, title, description, accentColor = "emerald" }) => {
  const colorClasses = {
    emerald: "from-emerald-100 to-teal-100 text-emerald-600",
    teal: "from-teal-100 to-cyan-100 text-teal-600",
    amber: "from-amber-100 to-orange-100 text-amber-600",
  };

  return (
    <div className="flex items-start space-x-4 p-5 rounded-xl bg-white border border-stone-200 shadow-sm hover:shadow-md hover:border-stone-300 transition-all group">
      <div className={`flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br ${colorClasses[accentColor]} flex items-center justify-center group-hover:scale-110 transition-transform`}>
        {icon}
      </div>
      <div>
        <h4 className="font-semibold text-stone-800 text-sm mb-1">{title}</h4>
        <p className="text-stone-500 text-xs leading-relaxed">{description}</p>
      </div>
    </div>
  );
};

// Main App Component
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
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-stone-100 to-amber-50/30 text-stone-800 relative">
      {/* Subtle Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-emerald-200/15 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-20 w-96 h-96 bg-amber-200/15 rounded-full blur-3xl" />
      </div>

      {/* Subtle Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(16,185,129,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(16,185,129,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      <div className="relative z-10">
        {/* SIGNED OUT */}
        <SignedOut>
          <LandingPage />
        </SignedOut>

        {/* SIGNED IN */}
        <SignedIn>
          <nav className="fixed top-6 right-8 z-50">
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  userButtonAvatarBox: "w-10 h-10 shadow-lg ring-2 ring-white",
                },
              }}
            />
          </nav>

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
    </div>
  );
}
