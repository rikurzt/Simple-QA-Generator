import streamlit as st
import tempfile
import os
import json
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from typing import List

# OpenAI客户端配置（將在運行時根據用戶設定初始化）
client = None

# Document loaders mapping
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def init_openai_client():
    """初始化OpenAI客戶端"""
    global client
    try:
        api_key = st.session_state.get('api_key', '')
        base_url = st.session_state.get('base_url', 'https://api.openai.com/v1')
        
        if not api_key:
            st.error("請先設定API Key")
            return False
            
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return True
    except Exception as e:
        st.error(f"初始化OpenAI客戶端失敗: {e}")
        return False

def get_completion(prompt, model=None, temperature=None, max_tokens=None):
    """獲取模型的響應"""
    global client
    
    # 如果客戶端未初始化，嘗試初始化
    if client is None:
        if not init_openai_client():
            return None
    
    # 使用用戶設定的參數或默認值
    model = model or st.session_state.get('model_name', 'gpt-4.1-nano')
    temperature = temperature if temperature is not None else st.session_state.get('temperature', 0.1)
    max_tokens = max_tokens or st.session_state.get('max_tokens', 4096)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"調用API時發生錯誤: {e}")
        return None

def get_default_qa_prompt():
    """獲取默認的QA生成提示詞"""
    return """基於以下給定的文本，生成多組高質量的問答對。請遵循以下指南：

1. 問題部分：
- 為不同的主題和概念創建多個問答對
- 每個問題應考慮用戶可能的多種問法，例如：
- 直接詢問（如"什麼是...？"）
- 請求確認（如"是否可以說...？"）
- 尋求解釋（如"請解釋一下...的含義。"）
- 假設性問題（如"如果...會怎樣？"）
- 例子請求（如"能否舉個例子說明...？"）
- 問題應涵蓋文本中的關鍵信息、主要概念和細節，確保不遺漏重要內容。

2. 答案部分：
- 提供一個全面、信息豐富的答案，涵蓋問題的所有可能角度，確保邏輯連貫。
- 答案應直接基於給定文本，確保準確性和一致性。
- 包含相關的細節，如日期、名稱、職位等具體信息，必要時提供背景信息以增強理解。

3. 格式：
- 使用 "Q:" 標記每個問題的開始
- 使用 "A:" 標記對應答案的開始
- 問答對之間用兩個空行分隔

4. 內容要求：
- 確保問答對緊密圍繞文本主題，避免偏離主題。
- 避免添加文本中未提及的信息，確保信息的真實性。

給定文本：
{text_content}

請基於這個文本生成多個問答對。"""

def get_default_json_system_prompt():
    """獲取默認的JSON轉換系統提示詞"""
    return """你是一個JSON格式轉換專家。將原始問答對文本轉換為標準JSON數組，每個問題必須成為獨立的QA對。

CRITICAL RULES:
- ONLY output valid JSON array format: [...]
- SEPARATE each question into individual QA pairs
- If one "Q:" contains multiple questions, split them into separate objects
- Each question gets its own JSON object with the same answer
- NO explanations, comments, or additional text
- NO markdown code blocks

TASK:
1. Find all Q: and A: pairs
2. If Q: contains multiple questions (separated by newlines), create separate QA pairs for each
3. Each question should be paired with the corresponding answer

EXAMPLE:
If input has: Q: Question1? Question2? Question3? A: Answer content
Output: [
  {"question": "Question1?", "answer": "Answer content"},
  {"question": "Question2?", "answer": "Answer content"},
  {"question": "Question3?", "answer": "Answer content"}
]"""

def test_api_connection(api_key, base_url, model_name):
    """測試API連接"""
    try:
        test_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # 簡單的測試請求
        response = test_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.1
        )
        
        return True
    except Exception as e:
        st.error(f"API測試失敗: {e}")
        return False

def generate_qa_pairs_with_progress(text_chunks):
    """生成問答對並顯示進度"""
    raw_qa_responses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 第一階段：生成原始QA對
    status_text.text("第一階段：生成原始QA對...")
    for i, chunk in enumerate(text_chunks):
        # 使用自定義的QA生成prompt
        qa_prompt = st.session_state.get('qa_generation_prompt', get_default_qa_prompt())
        prompt = qa_prompt.format(text_content=chunk.page_content)
        
        response = get_completion(prompt)
        if response:
            raw_qa_responses.append({
                "raw_response": response,
                "source_chunk": chunk.page_content
            })
        
        progress = (i + 1) / (len(text_chunks) * 2)  # 總進度的一半
        progress_bar.progress(progress)
    
    # 第二階段：將原始響應轉換為結構化JSON
    status_text.text("第二階段：處理並結構化QA對...")
    final_qa_pairs = []
    
    for i, qa_response in enumerate(raw_qa_responses):
        structured_qa_pairs = process_raw_qa_to_json(qa_response["raw_response"], qa_response["source_chunk"])
        final_qa_pairs.extend(structured_qa_pairs)
        
        progress = 0.5 + ((i + 1) / len(raw_qa_responses)) * 0.5  # 後半段進度
        progress_bar.progress(progress)
    
    status_text.text(f"✅ 完成！共生成 {len(final_qa_pairs)} 個QA對")
    return final_qa_pairs

def process_raw_qa_to_json(raw_response, source_chunk):
    """直接使用OpenAI API將原始QA響應轉換為結構化的JSON格式"""
    global client
    
    # 如果客戶端未初始化，嘗試初始化
    if client is None:
        if not init_openai_client():
            return []
    
    try:
        # 使用自定義的JSON轉換system prompt
        json_system_prompt = st.session_state.get('json_system_prompt', get_default_json_system_prompt())
        
        # 使用用戶設定的參數，JSON轉換可以使用獨立的模型
        model = st.session_state.get('json_model_name', st.session_state.get('model_name', 'gpt-4.1-nano'))
        temperature = st.session_state.get('json_temperature', 0.1)
        max_tokens = st.session_state.get('max_tokens', 4096)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": json_system_prompt
                },
                {
                    "role": "user", 
                    "content": f"INPUT TEXT:\n{raw_response}\n\nJSON OUTPUT:"
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1
        )
        
        json_response = response.choices[0].message.content.strip()
        
        # 直接解析JSON，不做額外處理
        start_bracket = json_response.find('[')
        end_bracket = json_response.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1:
            json_response = json_response[start_bracket:end_bracket + 1]
            qa_list = json.loads(json_response)
            
            # 添加source_chunk到每個QA對
            for qa in qa_list:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    qa["source_chunk"] = source_chunk
            
            return qa_list
        else:
            st.error("LLM未返回有效的JSON格式")
            return []
            
    except Exception as e:
        st.error(f"API調用失敗: {e}")
        return []



def load_single_document(file_path: str) -> List[Document]:
    """載入單個文檔"""
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def process_files(uploaded_files):
    """處理上傳的多個文件並生成文本塊"""
    all_text_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            documents = load_single_document(tmp_file_path)
            if not documents:
                st.error(f"文件 {uploaded_file.name} 處理失敗，請檢查文件格式是否正確。")
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            text_chunks = text_splitter.split_documents(documents)
            all_text_chunks.extend(text_chunks)
        except Exception as e:
            st.error(f"處理文件 {uploaded_file.name} 時發生錯誤: {e}")
        finally:
            os.unlink(tmp_file_path)
    
    return all_text_chunks

def download_qa_pairs_as_json(qa_pairs, filename="qa_pairs.json"):
    """下載QA對為標準JSON文件"""
    if qa_pairs:
        json_data = {
            "qa_pairs": qa_pairs,
            "total_count": len(qa_pairs),
            "generated_timestamp": st.session_state.get('generation_timestamp', '')
        }
        
        # 格式化JSON數據以提高可讀性
        json_str = json.dumps(json_data, ensure_ascii=False, indent=4)
        
        # 創建下載按鈕
        st.download_button(
            label="下載QA對為JSON文件",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )

def download_qa_pairs_as_sft_format(qa_pairs, system_prompt="你是一個有用的AI助手。", filename="sft_qa_pairs.json"):
    """下載QA對為SFTTrainer格式的JSON文件"""
    if qa_pairs:
        sft_data = []
        for qa in qa_pairs:
            sft_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qa.get('question', '')},
                    {"role": "assistant", "content": qa.get('answer', '')}
                ]
            }
            sft_data.append(sft_entry)
        
        # 格式化JSON數據
        json_str = json.dumps(sft_data, ensure_ascii=False, indent=2)
        
        # 創建下載按鈕
        st.download_button(
            label="下載SFTTrainer格式JSON文件",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )

def main():
    """主函數，設置Streamlit界面"""
    st.set_page_config(page_title="QA對生成器", layout="wide")
    st.title("QA對生成器")
    
    st.markdown("### 📚 上傳文件並生成問答對")
    st.markdown("支持的文件格式：TXT, PDF, DOCX, CSV, HTML, Markdown等")
    

    
    # 文件上傳
    uploaded_files = st.file_uploader(
        "選擇要處理的文件", 
        type=["txt", "pdf", "docx", "csv", "html", "md", "odt", "ppt", "pptx", "epub", "eml"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"已上傳 {len(uploaded_files)} 個文件！")
        
        # 顯示上傳的文件列表
        with st.expander("查看上傳的文件", expanded=False):
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")
        
        if st.button("🚀 開始處理文件並生成QA對", type="primary"):
            # 檢查API設定
            if not st.session_state.get('api_key'):
                st.error("❌ 請先在側邊欄設定API Key")
                return
            
            # 初始化OpenAI客戶端
            if not init_openai_client():
                st.error("❌ API初始化失敗，請檢查設定")
                return
            
            # 處理文件
            with st.spinner("正在處理文件..."):
                text_chunks = process_files(uploaded_files)
                if not text_chunks:
                    st.error("文件處理失敗，請檢查文件格式是否正確。")
                    return
                st.info(f"✅ 文件已分割成 {len(text_chunks)} 個文本段")

            # 生成QA對
            import time
            st.session_state.generation_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.qa_pairs = generate_qa_pairs_with_progress(text_chunks)
            if st.session_state.qa_pairs:
                st.success(f"🎉 生成完成！共產生 {len(st.session_state.qa_pairs)} 個獨立的QA對")
                st.info(f"💡 每個文本段平均產生 {len(st.session_state.qa_pairs)/len(text_chunks):.1f} 個QA對")
            else:
                st.error("❌ 未能生成任何QA對，請檢查文件內容或API配置")

    # 顯示生成的QA對
    if hasattr(st.session_state, 'qa_pairs') and st.session_state.qa_pairs:
        st.markdown("---")
        st.markdown("### 📋 生成的QA對")
        
        # 統計信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總QA對數量", len(st.session_state.qa_pairs))
        with col2:
            if 'generation_timestamp' in st.session_state:
                st.metric("生成時間", st.session_state.generation_timestamp)
        with col3:
            # 計算來源文件數量
            unique_sources = len(set(qa.get('source_chunk', '')[:100] for qa in st.session_state.qa_pairs))
            st.metric("文本段數量", unique_sources)
        with col4:
            st.metric("下載選項", "多種格式")
        
        # 下載選項
        st.markdown("#### 💾 下載選項")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            download_qa_pairs_as_json(st.session_state.qa_pairs)
        
        with col2:
            # SFT格式下載設定
            with st.expander("SFT格式設定", expanded=False):
                sft_system_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.get('sft_system_prompt', '你是一個有用的AI助手。'),
                    height=100,
                    help="用於SFT訓練的系統提示詞"
                )
                st.session_state.sft_system_prompt = sft_system_prompt
            
            download_qa_pairs_as_sft_format(
                st.session_state.qa_pairs, 
                system_prompt=st.session_state.get('sft_system_prompt', '你是一個有用的AI助手。')
            )
        
        with col3:
            st.markdown("**格式說明：**")
            st.markdown("• 標準JSON：原始QA對格式")
            st.markdown("• SFT格式：適用於模型微調")
        
        # QA對預覽
        st.markdown("#### 🔍 QA對預覽")
        
        # 分頁顯示
        items_per_page = 5
        total_pages = (len(st.session_state.qa_pairs) + items_per_page - 1) // items_per_page
        
        if total_pages > 1:
            page = st.selectbox("選擇頁面", range(1, total_pages + 1), key="page_selector")
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(st.session_state.qa_pairs))
        else:
            start_idx = 0
            end_idx = len(st.session_state.qa_pairs)
        
        # 顯示當前頁的QA對
        for i in range(start_idx, end_idx):
            qa = st.session_state.qa_pairs[i]
            with st.expander(f"**QA對 {start_idx + i + 1}**", expanded=False):
                st.markdown("**❓ 問題:**")
                st.markdown(qa['question'])
                st.markdown("**✅ 答案:**")
                st.markdown(qa['answer'])
                if 'source_chunk' in qa:
                    st.markdown("**📄 來源文本:**")
                    st.text_area(
                        "來源文本內容", 
                        value=qa['source_chunk'], 
                        height=100, 
                        key=f"source_{start_idx + i}", 
                        disabled=True,
                        label_visibility="collapsed"
                    )
    
    # 側邊欄信息
    with st.sidebar:
        st.markdown("### 🔑 API設定")
        
        # API設定功能
        with st.expander("🔧 API & 模型設定", expanded=True):
            # API Key設定
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get('api_key', ''),
                type="password",
                help="請輸入您的OpenAI API Key"
            )
            
            # Base URL設定
            base_url = st.text_input(
                "Base URL",
                value=st.session_state.get('base_url', 'https://api.openai.com/v1'),
                help="API的基礎URL，默認為OpenAI官方"
            )
            
            # 模型設定
            model_name = st.text_input(
                "QA生成模型名稱",
                value=st.session_state.get('model_name', 'gpt-4.1-nano'),
                help="用於生成QA對的模型名稱"
            )
            
            # JSON轉換專用模型設定
            json_model_name = st.text_input(
                "JSON轉換模型名稱",
                value=st.session_state.get('json_model_name', ''),
                help="用於JSON轉換的模型名稱，留空則使用QA生成模型"
            )
            
            # 參數設定
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "QA生成Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get('temperature', 0.1),
                    step=0.1,
                    help="控制生成的隨機性"
                )
            
            with col2:
                json_temperature = st.slider(
                    "JSON轉換Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('json_temperature', 0.1),
                    step=0.1,
                    help="JSON轉換的溫度參數"
                )
            
            max_tokens = st.number_input(
                "最大Token數",
                min_value=100,
                max_value=32000,
                value=st.session_state.get('max_tokens', 4096),
                step=100,
                help="每次API調用的最大token數"
            )
            
            # 保存設定按鈕
            if st.button("💾 保存API設定", key="save_api_settings"):
                st.session_state.api_key = api_key
                st.session_state.base_url = base_url
                st.session_state.model_name = model_name
                st.session_state.json_model_name = json_model_name if json_model_name else model_name
                st.session_state.temperature = temperature
                st.session_state.json_temperature = json_temperature
                st.session_state.max_tokens = max_tokens
                
                # 重新初始化客戶端
                global client
                client = None
                if init_openai_client():
                    st.success("✅ API設定已保存並測試成功")
                else:
                    st.error("❌ API設定保存失敗，請檢查設定")
            
            # 測試連接按鈕
            if st.button("🔌 測試API連接", key="test_api_connection"):
                if api_key:
                    # 測試QA生成模型
                    test_result_qa = test_api_connection(api_key, base_url, model_name)
                    if test_result_qa:
                        st.success("✅ QA生成模型連接測試成功")
                        
                        # 如果設定了不同的JSON轉換模型，也進行測試
                        if json_model_name and json_model_name != model_name:
                            test_result_json = test_api_connection(api_key, base_url, json_model_name)
                            if test_result_json:
                                st.success("✅ JSON轉換模型連接測試成功")
                            else:
                                st.error("❌ JSON轉換模型連接測試失敗")
                    else:
                        st.error("❌ QA生成模型連接測試失敗")
                else:
                    st.warning("⚠️ 請先輸入API Key")
        
        st.markdown("---")
        st.markdown("### ⚙️ 提示詞設定")
        
        # 提示詞自定義功能
        with st.expander("🔧 自定義提示詞", expanded=False):
            st.markdown("#### 第一階段：QA生成提示詞")
            
            # 初始化默認提示詞
            if 'qa_generation_prompt' not in st.session_state:
                st.session_state.qa_generation_prompt = get_default_qa_prompt()
            
            # QA生成提示詞編輯器
            qa_prompt = st.text_area(
                "QA生成提示詞 (使用 {text_content} 作為文本內容的占位符)",
                value=st.session_state.qa_generation_prompt,
                height=200,
                key="qa_prompt_editor"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存QA提示詞", key="save_qa_prompt"):
                    st.session_state.qa_generation_prompt = qa_prompt
                    st.success("✅ QA提示詞已保存")
            
            with col2:
                if st.button("🔄 重置QA提示詞", key="reset_qa_prompt"):
                    st.session_state.qa_generation_prompt = get_default_qa_prompt()
                    st.success("✅ QA提示詞已重置")
                    st.rerun()
            
            st.markdown("---")
            st.markdown("#### 第二階段：JSON轉換提示詞")
            
            # 初始化默認JSON提示詞
            if 'json_system_prompt' not in st.session_state:
                st.session_state.json_system_prompt = get_default_json_system_prompt()
            
            # JSON轉換提示詞編輯器
            json_prompt = st.text_area(
                "JSON轉換系統提示詞",
                value=st.session_state.json_system_prompt,
                height=200,
                key="json_prompt_editor"
            )
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("💾 保存JSON提示詞", key="save_json_prompt"):
                    st.session_state.json_system_prompt = json_prompt
                    st.success("✅ JSON提示詞已保存")
            
            with col4:
                if st.button("🔄 重置JSON提示詞", key="reset_json_prompt"):
                    st.session_state.json_system_prompt = get_default_json_system_prompt()
                    st.success("✅ JSON提示詞已重置")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ 使用說明")
        st.markdown("""
        **📋 處理流程：**
        1. 上傳一個或多個文件
        2. 點擊"開始處理"按鈕
        3. 系統將自動：
           - 分割文本為小段
           - 生成原始QA對
           - 用LLM轉換為JSON格式
           - 分離成獨立的QA對
        4. 查看和下載生成的QA對
        
        **🗂️ 支持的文件格式：**
        - 文本文件 (.txt)
        - PDF文件 (.pdf)  
        - Word文檔 (.docx)
        - CSV文件 (.csv)
        - HTML文件 (.html)
        - Markdown文件 (.md)
        - PowerPoint (.ppt, .pptx)
        - 電子郵件 (.eml)
        - 電子書 (.epub)
        - 等等...
        """)
        
        if hasattr(st.session_state, 'qa_pairs') and st.session_state.qa_pairs:
            st.markdown("---")
            st.markdown("### 📊 生成統計")
            st.write(f"📝 QA對總數: **{len(st.session_state.qa_pairs)}**")
            if 'generation_timestamp' in st.session_state:
                st.write(f"⏰ 生成時間: **{st.session_state.generation_timestamp}**")
            
            # 統計問題類型
            question_types = {}
            for qa in st.session_state.qa_pairs:
                question = qa.get('question', '').strip()
                if question.startswith('什麼'):
                    question_types['什麼'] = question_types.get('什麼', 0) + 1
                elif question.startswith('如何') or question.startswith('怎樣'):
                    question_types['如何/怎樣'] = question_types.get('如何/怎樣', 0) + 1
                elif question.startswith('為什麼'):
                    question_types['為什麼'] = question_types.get('為什麼', 0) + 1
                elif '？' in question:
                    question_types['其他問句'] = question_types.get('其他問句', 0) + 1
                else:
                    question_types['陳述式'] = question_types.get('陳述式', 0) + 1
            
            if question_types:
                st.markdown("**❓ 問題類型分布:**")
                for q_type, count in question_types.items():
                    st.write(f"  - {q_type}: {count}")
                
                st.markdown("**🔄 處理說明:**")
                st.write("- 直接使用OpenAI API處理")
                st.write("- LLM自動分離合併問題")
                st.write("- 純JSON格式輸出")
                
                st.markdown("**⚙️ 當前設定:**")
                st.write(f"- QA生成模型: {st.session_state.get('model_name', 'gpt-4.1-nano')}")
                json_model = st.session_state.get('json_model_name', st.session_state.get('model_name', 'gpt-4.1-nano'))
                st.write(f"- JSON轉換模型: {json_model}")
                st.write(f"- QA溫度: {st.session_state.get('temperature', 0.1)}")
                st.write(f"- JSON溫度: {st.session_state.get('json_temperature', 0.1)}")
                st.write(f"- 最大Token: {st.session_state.get('max_tokens', 4096)}")

if __name__ == "__main__":
    main()