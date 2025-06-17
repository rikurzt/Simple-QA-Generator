# Simple QA Generator

An automated Q&A pair generation tool based on large language models that can convert various document formats into high-quality question-answer pairs (QA Pairs), designed for fine-tuning with HuggingFace SFTTrainer. 
Compared to the original project, the TaskingAI installation requirement has been removed.

## 🌟 Key Features

- **Multi-format Support**: Supports TXT, PDF, DOCX, CSV, HTML, Markdown, and other document formats
- **Intelligent Analysis**: Uses advanced text splitting techniques to divide documents into appropriately sized text chunks
- **Two-stage Generation**: Employs a two-stage processing workflow to ensure high-quality structured Q&A pairs
- **Independent Model Configuration**: Allows separate configuration of models for QA generation and JSON conversion, optimizing cost and effectiveness
- **Custom Prompts**: Fully customizable prompt system to adapt to different domains and requirements
- **Multiple Output Formats**: Supports standard JSON and SFTTrainer formats to meet different use cases
- **API Configuration**: Flexible OpenAI API configuration supporting various models and parameter adjustments
- **User-friendly**: Intuitive web interface based on Streamlit with simple operation
- **Real-time Preview**: Instantly view generated Q&A pairs to ensure quality meets expectations

## 🚀 Quick Start

### System Requirements

- Python 3.11+
- Valid OpenAI API key or compatible API service

### Dependencies

- streamlit==1.22.0
- requests==2.31.0
- openai
- langchain==0.3.25
- PyMuPDF==1.22.5
- pandas==2.1.1
- langchain_community==0.3.25

### Installation Steps

1. **Clone the Project**
```bash
git clone https://github.com/your-repo/AutoQAG.git
cd AutoQAG
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the Application**
```bash
streamlit run Code/SQA.py
```

4. **Open Browser**
Visit `http://localhost:8501` to start using

## 🔧 Usage Guide

### 1. API Configuration

When first using, please configure API settings in the sidebar:

- **API Key**: Enter your OpenAI API key
- **Base URL**: Default is OpenAI official, can be customized to other compatible services
- **QA Generation Model**: Select the language model for generating Q&A pairs
- **JSON Conversion Model**: Optionally select a different model for JSON conversion (can be left empty to use QA generation model)
- **Temperature Settings**: Adjust generation randomness (QA generation and JSON conversion can be set separately)
- **Max Token Count**: Control response length

### 2. Prompt Customization

Customize prompts according to your needs:

- **QA Generation Prompt**: Controls how to generate Q&A pairs from text
- **JSON Conversion Prompt**: Specifies output format and structure
- Can save and reset to default values

### 3. File Upload and Processing

- Supports single or batch file upload
- Automatic text splitting and preprocessing
- Real-time display of processing progress

### 4. QA Pair Generation

- Two-stage processing ensures quality:
  - **Stage 1**: Generate raw Q&A content from text chunks
  - **Stage 2**: Convert to structured JSON format
- Real-time preview of generation results
- Support multiple download formats: standard JSON format and SFTTrainer format

### Processing Workflow

1. **Document Loading**: Use LangChain's document loaders to handle multiple formats
2. **Text Splitting**: Intelligently split into appropriately sized text chunks
3. **QA Generation**: Generate Q&A content based on custom prompts
4. **Format Conversion**: Convert raw responses to structured JSON
5. **Result Output**: Provide preview and download functionality

### Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Text | .txt | Plain text files |
| PDF | .pdf | Portable Document Format |
| Word | .docx | Microsoft Word documents |
| CSV | .csv | Comma-separated values files |
| HTML | .html, .htm | HyperText Markup Language |
| Markdown | .md | Markdown language files |

## ⚙️ Configuration Options

### API Parameters

- **Temperature**: Controls generation creativity (0.0-2.0)
- **Max Tokens**: Maximum response length (100-32000)
- **Model Name**: Language model to use

### Text Processing

- **Chunk Size**: Text chunk size (default 2000 characters)
- **Chunk Overlap**: Text chunk overlap (default 500 characters)

## 📊 Output Formats

### Standard JSON Format

Generated Q&A pairs are output in standard JSON format with the following structure:

```json
{
  "qa_pairs": [
    {
      "question": "Question content",
      "answer": "Answer content",
      "source_chunk": "Original text paragraph"
    }
  ],
  "total_count": 1,
  "generated_timestamp": "2024-01-01 12:00:00"
}
```

### SFTTrainer Format

SFTTrainer format output suitable for model fine-tuning:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Question content"},
      {"role": "assistant", "content": "Answer content"}
    ]
  }
]
```


