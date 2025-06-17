# 簡易問答對生成器 (Simple QA Generator)

一個基於大型語言模型的自動化問答對生成工具，能夠將各種格式的文檔轉換為高品質的問答對（QA Pairs），專為HuggingFace SFTTrainer進行微調而誕生。
相比於原專案，移除了TaskingAI的安裝需求

## 🌟 主要特色

- **多格式支援**：支援 TXT、PDF、DOCX、CSV、HTML、Markdown 等多種文件格式
- **智能分析**：使用先進的文本分割技術，將文檔切分為適當大小的文本塊
- **雙階段生成**：採用兩階段處理流程，確保生成高質量的結構化問答對
- **獨立模型配置**：可分別設定QA生成和JSON轉換使用的模型，優化成本和效果
- **自定義提示**：完全可自定義的提示詞系統，適應不同領域和需求
- **多種輸出格式**：支援標準JSON和SFTTrainer格式，滿足不同使用場景
- **API 配置**：靈活的 OpenAI API 配置，支援多種模型和參數調整
- **用戶友好**：基於 Streamlit 的直觀網頁界面，操作簡單
- **實時預覽**：即時查看生成的問答對，確保質量符合預期

## 🚀 快速開始

### 系統需求

- Python 3.11+
- 有效的 OpenAI API 金鑰或相容的 API 服務

### 依賴套件

- streamlit==1.22.0
- requests==2.31.0
- openai
- langchain==0.3.25
- PyMuPDF==1.22.5
- pandas==2.1.1
- langchain_community==0.3.25

### 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/your-repo/AutoQAG.git
cd AutoQAG
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **啟動應用程式**
```bash
streamlit run Code/SQA.py
```

4. **開啟瀏覽器**
訪問 `http://localhost:8501` 開始使用

## 🔧 使用指南

### 1. API 配置

首次使用時，請在側邊欄配置 API 設定：

- **API 金鑰**：輸入您的 OpenAI API 金鑰
- **Base URL**：預設為 OpenAI 官方，可自定義為其他相容服務
- **QA生成模型**：選擇用於生成問答對的語言模型
- **JSON轉換模型**：可選擇不同的模型進行JSON轉換（可留空使用QA生成模型）
- **溫度設定**：調整生成的隨機性（QA生成和JSON轉換可分別設定）
- **最大 Token 數**：控制回應長度

### 2. 提示詞自定義

根據您的需求自定義提示詞：

- **QA 生成提示**：控制如何從文本生成問答對
- **JSON 轉換提示**：指定輸出格式和結構
- 可保存和重設為預設值

### 3. 文件上傳與處理

- 支援單個或批量文件上傳
- 自動文本分割和預處理
- 即時顯示處理進度

### 4. QA 對生成

- 兩階段處理確保質量：
  - **階段一**：從文本塊生成原始問答內容
  - **階段二**：轉換為結構化 JSON 格式
- 實時預覽生成結果
- 支援多種格式下載：標準JSON格式和SFTTrainer格式


### 處理流程

1. **文檔載入**：使用 LangChain 的文檔載入器處理多種格式
2. **文本分割**：智能分割為適當大小的文本塊
3. **QA 生成**：基於自定義提示生成問答內容
4. **格式轉換**：將原始回應轉換為結構化 JSON
5. **結果輸出**：提供預覽和下載功能

### 支援的文件格式

| 格式 | 擴展名 | 說明 |
|------|--------|------|
| 文本 | .txt | 純文本文件 |
| PDF | .pdf | 可攜式文檔格式 |
| Word | .docx | Microsoft Word 文檔 |
| CSV | .csv | 逗號分隔值文件 |
| HTML | .html, .htm | 網頁標記語言 |
| Markdown | .md | 標記語言文件 |

## ⚙️ 配置選項

### API 參數

- **Temperature**：控制生成的創造性（0.0-2.0）
- **Max Tokens**：回應的最大長度（100-32000）
- **Model Name**：使用的語言模型

### 文本處理

- **Chunk Size**：文本塊大小（預設 2000 字符）
- **Chunk Overlap**：文本塊重疊（預設 500 字符）

## 📊 輸出格式

### 標準 JSON 格式

生成的問答對以標準 JSON 格式輸出，結構如下：

```json
{
  "qa_pairs": [
    {
      "question": "問題內容",
      "answer": "答案內容",
      "source_chunk": "原始文本段落"
    }
  ],
  "total_count": 1,
  "generated_timestamp": "2024-01-01 12:00:00"
}
```

### SFTTrainer 格式

適用於模型微調的 SFTTrainer 格式輸出：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一個有用的AI助手。"},
      {"role": "user", "content": "問題內容"},
      {"role": "assistant", "content": "答案內容"}
    ]
  }
]
```


