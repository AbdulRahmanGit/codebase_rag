# Codebase RAG System

A Streamlit-based web application that implements a Retrieval-Augmented Generation (RAG) system for analyzing and querying code repositories. The system uses Pinecone for vector storage and Google's Gemini for generating responses.

## Features

- **Repository Analysis**: Clone and analyze GitHub repositories
- **Code Understanding**: Process multiple programming languages and file types
- **Intelligent Querying**: Ask questions about the codebase
- **Vector Storage**: Efficient storage and retrieval using Pinecone
- **Interactive Chat**: User-friendly interface for code exploration
- **Export Capabilities**: Save Q&A sessions in markdown format

## Supported File Types

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`)
- Go (`.go`)
- Rust (`.rs`)
- Swift (`.swift`)
- Vue (`.vue`)
- Jupyter Notebooks (`.ipynb`)
- Markdown (`.md`)

## Prerequisites

- Python 3.8 or higher
- Pinecone API key
- Google Gemini API key
- Git (for repository cloning)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Configuration

1. **API Keys**:
   - Enter your Pinecone API key
   - Enter your Google Gemini API key
   - Keys are stored securely in session state

2. **Settings**:
   - Adjust chunk size (500-2000 characters)
   - Set chunk overlap (100-500 characters)
   - Enable debug mode for detailed information

## Usage

1. **Repository Analysis**:
   - Enter a GitHub repository URL
   - Click "Analyze Repository"
   - Wait for the analysis to complete

2. **Asking Questions**:
   - Type your question in the chat box
   - Click "Submit Question"
   - View the generated response

3. **History and Export**:
   - View past questions and answers
   - Export conversations to markdown
   - Download Q&A sessions

## Technical Details

- **Embeddings**: Uses `all-mpnet-base-v2` model for text embeddings
- **Vector Storage**: Pinecone serverless index with 768 dimensions
- **Query Processing**: Top 10 most relevant chunks retrieved
- **Response Generation**: Gemini 2.0 Flash model for answers

## Security

- API keys are stored in session state only
- No persistent storage of sensitive information
- Local processing of repository content

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 