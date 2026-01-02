import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import "tailwindcss/tailwind.css";
import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";
import { SignedIn, SignedOut, SignIn, UserButton } from "@clerk/clerk-react";
import { LayoutDashboard, Library } from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [books, setBooks] = useState([]);

  // Fetch books (Public endpoint, no auth needed)
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
    <div className="min-h-screen bg-slate-900 text-white">
      {/* 1. If User is Signed Out -> Show Clerk Login */}
      <SignedOut>
        <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
           <div className="mb-8 text-center animate-fade-in">
              <div className="flex justify-center mb-4">
                <div className="p-4 bg-white/10 rounded-full border border-white/20 shadow-xl shadow-purple-500/20">
                  <Library className="w-12 h-12 text-purple-400" />
                </div>
              </div>
              <h1 className="text-4xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                RAG Knowledge Bot
              </h1>
              <p className="text-slate-400">Please sign in to access your secure knowledge base</p>
           </div>
           {/* Clerk Login Component */}
           <SignIn /> 
        </div>
      </SignedOut>

      {/* 2. If User is Signed In -> Show App */}
      <SignedIn>
        {/* Top Navigation Bar */}
        <nav className="absolute top-0 right-0 p-4 z-50 flex items-center gap-4">
          <div className="bg-slate-800/80 backdrop-blur-md px-4 py-2 rounded-full border border-slate-700 flex items-center gap-2 shadow-lg">
             <LayoutDashboard className="w-4 h-4 text-purple-400"/>
             <span className="text-sm font-medium text-purple-100">User Dashboard</span>
          </div>
          <div className="bg-white rounded-full p-1">
             <UserButton afterSignOutUrl="/" />
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<RAGBookBot />} />
          <Route
            path="/ingest"
            element={
              <IngestionPage books={books} onUploadSuccess={handleUploadSuccess} />
            }
          />
        </Routes>
      </SignedIn>
    </div>
  );
}