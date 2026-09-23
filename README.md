# 🏠 Real Estate AI Assistant

An AI-powered **Real Estate Assistant** that allows users to extract and interact with information from real estate web pages. The application uses **Retrieval-Augmented Generation (RAG)** with a **ChromaDB Vector Database** to retrieve relevant information from provided URLs and generate context-aware responses using an LLM.

## 🚀 Features

* 🔗 Accepts up to **3 real estate website URLs**
* 📄 Extracts and processes content from the provided web pages
* 🧩 Splits and converts the content into embeddings
* 🗄️ Stores document embeddings in a **ChromaDB Vector Database**
* 🔍 Performs semantic similarity search to retrieve relevant information
* 🤖 Uses an **LLM** to generate answers based on retrieved context
* 💬 Provides an interactive **Streamlit chat interface**
* ⚡ Enables users to ask natural-language questions about the provided properties

## 🏗️ Project Architecture

The application follows a **Retrieval-Augmented Generation (RAG)** architecture:

```text
                 ┌─────────────────────┐
                 │   Real Estate URLs  │
                 │       (1 - 3)       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Web Data Loading  │
                 │    & Processing     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Text Chunking &   │
                 │     Embeddings      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      ChromaDB       │
                 │    Vector Store     │
                 └──────────┬──────────┘
                            │
                     User Question
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Semantic Similarity │
                 │      Search         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Relevant Context  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │        LLM          │
                 │  Response Generation│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Streamlit Chat    │
                 │      Response       │
                 └─────────────────────┘
```

## 📁 Project Structure

```text
real-estate-assistant/
│
├── resources/
│   └── vectordb/
│       └── chroma.sqlite3
│
├── main.py
├── rag.py
└── README.md
```

### `main.py`

Contains the **Streamlit user interface**.

The UI provides:

* Three input fields for real estate URLs
* An option to process and store the website content
* A chat interface for asking questions
* Display of LLM-generated responses

### `rag.py`

Contains the core **RAG pipeline**, including:

* Vector database initialization
* URL/document loading
* Text processing and chunking
* Embedding generation
* Storing documents in ChromaDB
* Retrieving relevant documents
* Passing retrieved context to the LLM
* Generating responses based on the retrieved information

### `resources/vectordb/chroma.sqlite3`

The **ChromaDB persistent storage file** containing the processed document information and vector database data.

## 🔄 How It Works

### 1. Provide Real Estate URLs

The user enters up to three real estate web URLs through the Streamlit interface.

```text
URL 1 → Property Website
URL 2 → Property Website
URL 3 → Property Website
```

### 2. Extract Website Content

The application loads the content from the provided URLs and processes the extracted text.

### 3. Create Embeddings

The extracted content is divided into smaller chunks and converted into numerical vector representations called **embeddings**.

### 4. Store in ChromaDB

The generated embeddings and corresponding document chunks are stored in the **ChromaDB vector database**.

```text
Web Content
     ↓
Text Chunks
     ↓
Embeddings
     ↓
ChromaDB
```

### 5. Ask Questions

The user can ask questions through the Streamlit chat interface, such as:

```text
What is the price of the property?

How many bedrooms does the property have?

What amenities are available?

Where is the property located?

What is the total area of the property?
```

### 6. Retrieve Relevant Context

When a question is submitted, the application performs a **semantic similarity search** against the vector database and retrieves the most relevant chunks of information.

### 7. Generate Response Using LLM

The retrieved information is passed as context to the LLM along with the user's question.

The LLM uses this context to generate a relevant and context-aware answer.

```text
User Question
      ↓
Vector Search
      ↓
Relevant Documents
      ↓
Retrieved Context
      ↓
LLM
      ↓
Generated Answer
```

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – Interactive web interface
* **ChromaDB** – Vector database
* **RAG (Retrieval-Augmented Generation)** – Information retrieval and generation architecture
* **Embeddings** – Semantic representation of web content
* **LLM** – Natural-language response generation

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd real-estate-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key
```

> Never commit API keys or other sensitive credentials to GitHub.

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run main.py
```

The application will open in your browser.

## 💬 Example Workflow

```text
1. Open the application
        ↓
2. Enter real estate URLs
        ↓
3. Process the URLs
        ↓
4. Website content is stored in ChromaDB
        ↓
5. Ask questions in the chat interface
        ↓
6. Relevant information is retrieved
        ↓
7. LLM generates the response
```

## 🎯 Use Cases

The Real Estate AI Assistant can be used to:

* Compare information across multiple property listings
* Quickly retrieve property details
* Answer questions about property amenities
* Find pricing and location information
* Summarize information from property websites
* Interact conversationally with real estate listings

## 🔮 Future Enhancements

* Add support for more than three URLs
* Enable comparison of multiple properties
* Add property recommendation functionality
* Display source URLs alongside responses
* Add conversation memory
* Add metadata filtering for properties
* Implement document update and deletion functionality
* Deploy the application using Streamlit Cloud or another cloud platform

## 📌 Key Concept

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be used to build a domain-specific AI assistant. Instead of relying solely on the LLM's pre-trained knowledge, the application retrieves relevant information from user-provided real estate websites and uses that information as context to generate responses.

---

### 👩‍💻 Author

**Lyneshia Correa**

If you found this project useful, consider ⭐ starring the repository!
