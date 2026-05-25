# AI Trainer Chatbot

An AI-powered interview preparation chatbot designed to help students and freshers prepare for Machine Learning, Deep Learning, NLP, Statistics, and AI-related interviews using semantic search and sentence-transformer embeddings.

The project uses a modern semantic intent-classification architecture built with FastAPI, Streamlit, and Sentence Transformers.

---

# Features

- AI Interview Coach for ML & AI topics
- Semantic intent classification using embeddings
- Real-time chatbot interface
- Mock interview question generation
- Covers Machine Learning, Deep Learning, NLP, Statistics, and MLOps
- Resume and career guidance support
- FastAPI backend API
- Streamlit frontend interface
- SentenceTransformer-based semantic search
- Cosine similarity intent matching
- Health monitoring endpoint
- Modular and scalable project structure

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend API framework |
| Streamlit | Frontend user interface |
| Sentence Transformers | Semantic embeddings |
| NumPy | Vector operations |
| Pydantic | Request validation |
| Uvicorn | ASGI server |
| Pickle | Semantic index storage |

---

# Project Structure

```txt
AI-Trainer-Chatbot/
│
├── app.py
├── main.py
├── predict.py
├── preprocessing.py
├── train.py
│
├── dataset/
│   └── intents.json
│
├── model/
│   └── semantic_index.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How It Works

## Training Phase

1. Load intents dataset
2. Extract training patterns
3. Convert patterns into embeddings using Sentence Transformers
4. Store embeddings in `semantic_index.pkl`

---

## Inference Phase

1. User enters a query
2. Query is converted into an embedding
3. Cosine similarity is computed against stored embeddings
4. Best matching intent is selected
5. Response is returned to the user

---

# Backend API

The backend is built using FastAPI.

## Available Endpoints

| Endpoint | Description |
|---|---|
| `/chat` | Main chatbot API |
| `/health` | Health check endpoint |
| `/docs` | Swagger API documentation |

---

# Frontend

The frontend is built using Streamlit and includes:

- Interactive chat interface
- Session-based chat history
- Quick topic suggestions
- Backend health status
- Real-time responses

---

# Topics Covered

The chatbot supports interview preparation for:

- Bias-Variance Tradeoff
- Precision, Recall, and F1 Score
- Regularization Techniques
- Cross Validation
- CNN, RNN, and Transformers
- BERT and GPT
- Feature Engineering
- Gradient Descent
- Deep Learning Concepts
- Mock Interview Questions
- Resume and Career Guidance

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Trainer-Chatbot.git
cd AI-Trainer-Chatbot
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run The Project

## Step 1 — Build Semantic Index

```bash
python train.py
```

---

## Step 2 — Start FastAPI Backend

```bash
uvicorn main:app --reload
```

Backend runs on:

```txt
http://localhost:8000
```

---

## Step 3 — Start Streamlit Frontend

```bash
streamlit run app.py
```

Frontend runs on:

```txt
http://localhost:8501
```

---

# Example Questions

```txt
Explain Bias Variance Tradeoff
What is Precision vs Recall?
Ask me a mock interview question
Explain Transformers
Difference between CNN and RNN
How does backpropagation work?
```

---

# Future Improvements

- Voice-based chatbot
- LLM integration
- Resume analyzer
- AI-generated interview scoring
- Multi-language support
- Chat history database
- Cloud deployment support

---

# Author

Developed as an AI/ML interview preparation assistant project focused on semantic NLP systems and modern chatbot architecture.

---

# Acknowledgements

- Sentence Transformers
- FastAPI
- Streamlit
- Hugging Face
- Open Source AI Community