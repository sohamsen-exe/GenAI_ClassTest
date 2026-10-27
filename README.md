# Pretrained Model Bias Detector

A local testing tool built with Streamlit and PyTorch/HuggingFace to investigate how pretrained language models generate text conditioned on different demographic identifiers. 

## Features
* **Frontend:** Interactive Streamlit UI with model selectors, adjustable sequence lengths, and prompt templating.
* **Backend:** Leverages Hugging Face `pipeline` for dynamic text generation (GPT-2, DistilGPT-2) and a secondary RoBERTa-based pipeline for sentiment classification.
* **Visual Diagnostics:** Automatically graphs sentiment divergence across matched pairs to visualize disparities.

## Requirements
* Python 3.8+
* A machine with at least 4GB of RAM (GPU optional but recommended for faster inference).

## Installation & Run Instructions

1. **Clone/Download the repository files** (`app.py`, `backend.py`, `requirements.txt`).
2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate