"""
Configuration for Marker PDF conversion with GPU/CPU support
Located at: backend/rag_based_book_bot/document_ingestion/marker_config.py
"""
import os
import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class MarkerConfig:
    """Configuration for Marker PDF to Markdown conversion"""
    
    # Device configuration
    device: Literal["cuda", "cpu", "auto"] = "auto"
    
    # Marker-specific settings
    batch_multiplier: int = 2  # For GPU processing
    max_pages: int = None  # None = process all pages
    langs: list = None  # None = auto-detect language
    
    # Output settings
    extract_images: bool = False
    paginate_output: bool = True
    
    def __post_init__(self):
        """Auto-configure device based on availability"""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.device = "cpu"
                print("⚠️  CUDA not available. Using CPU mode.")
        
        # Adjust batch multiplier for CPU
        if self.device == "cpu":
            self.batch_multiplier = 1
            print("   CPU mode: batch_multiplier set to 1")
    
    def to_marker_kwargs(self) -> dict:
        """Convert to kwargs for marker.convert_single_pdf()"""
        return {
            "max_pages": self.max_pages,
            "langs": self.langs,
            "batch_multiplier": self.batch_multiplier,
            "extract_images": self.extract_images,
        }


def get_default_config(use_gpu: bool = True) -> MarkerConfig:
    """
    Get default configuration
    
    Args:
        use_gpu: If True, use GPU if available. If False, force CPU.
    """
    return MarkerConfig(
        device="auto" if use_gpu else "cpu",
        batch_multiplier=2,
        max_pages=None,
        langs=None,
        extract_images=False,
        paginate_output=True
    )


def get_cpu_config() -> MarkerConfig:
    """Force CPU configuration"""
    return MarkerConfig(
        device="cpu",
        batch_multiplier=1,
        max_pages=None,
        langs=None,
        extract_images=False,
        paginate_output=True
    )


def get_gpu_config() -> MarkerConfig:
    """Force GPU configuration (will fail if CUDA not available)"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Cannot use GPU configuration.")
    
    return MarkerConfig(
        device="cuda",
        batch_multiplier=2,
        max_pages=None,
        langs=None,
        extract_images=False,
        paginate_output=True
    )