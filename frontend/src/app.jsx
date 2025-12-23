//app.jsx
import React, { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import "tailwindcss/tailwind.css";
import RAGBookBot from "./pages/RAGBookBot";
import IngestionPage from "./pages/IngestionPage";

const API_BASE_URL = "http://localhost:8000";

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
    <Routes>
      <Route path="/" element={<RAGBookBot />} />
      <Route
        path="/ingest"
        element={
          <IngestionPage books={books} onUploadSuccess={handleUploadSuccess} />
        }
      />
    </Routes>
  );
}
