//app.jsx
import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import "tailwindcss/tailwind.css";
import AdminLogin from "./pages/AdminLogin";
import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [books, setBooks] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(
    localStorage.getItem("adminAuth") === "true"
  );

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

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("adminAuth");
    setIsAuthenticated(false);
  };

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <AdminLogin onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <Routes>
      <Route path="/" element={<RAGBookBot onLogout={handleLogout} />} />
      <Route
        path="/ingest"
        element={
          <IngestionPage books={books} onUploadSuccess={handleUploadSuccess} />
        }
      />
    </Routes>
  );
}