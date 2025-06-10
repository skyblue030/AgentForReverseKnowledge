import dspy
import litellm
import os
import dotenv
from typing import List, Optional, Any, Dict, Union
import traceback
# import faiss # NO FAISS
# import numpy as np # NO NUMPY RETRIEVAL
# from sentence_transformers import SentenceTransformer # NO EMBEDDING
import time
import re # For parsing and splitting
# from langchain_text_splitters import RecursiveCharacterTextSplitter # Optional

# --- 1. 環境設定與 API Key ---
print("正在加載環境變數...")
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY: print("警告：未找到 GOOGLE_API_KEY 環境變數。")
litellm.set_verbose = False
print("環境變數加載完成。")

# --- 2. LiteLLMWrapper (保持最終修正版) ---
# --- 2. LiteLLMWrapper (再次修正 __call__ 和 basic_request) ---
# --- 2. LiteLLMWrapper (最終修正版 v3) ---
class LiteLLMWrapper(dspy.LM):
    def __init__(self, model_name="gemini/gemini-1.5-flash-latest", api_key=None, **kwargs):
        super().__init__(model_name)
        self.provider = "litellm"
        self.model_name = model_name
        self.api_key = api_key
        # 預設生成參數 - 稍微提高 temperature 看看是否影響解釋生成
        self.kwargs = {'temperature': 0.7, 'max_tokens': 4096, **kwargs}
        self.history = []
        print(f"LiteLLMWrapper initialized for model: {self.model_name}")

    # --- 核心請求方法 ---
    def basic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handles the basic request, sending the prompt string to the LLM
           and returning a dictionary structured similarly to OpenAI's format."""
        # print(f"LiteLLMWrapper basic_request called (prompt len={len(prompt)})") # Debug

        # 合併預設參數和運行時參數 (來自 DSPy 的 lm_kwargs)
        merged_kwargs = {**self.kwargs, **kwargs}
        litellm_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": merged_kwargs.get("temperature"),
            "max_tokens": merged_kwargs.get("max_tokens"),
            "stop": merged_kwargs.get("stop"),
            "top_p": merged_kwargs.get("top_p"),
            "api_key": self.api_key if self.api_key else None,
        }
        litellm_params = {k: v for k, v in litellm_params.items() if v is not None}

        try:
            response = litellm.completion(**litellm_params)
            # --- 構建 DSPy 通常期望的返回字典 ---
            if response and response.choices:
                 response_dict = {
                     "choices": [
                         {
                             # message or text format might be used by different DSPy parts
                             "message": {"role": "assistant", "content": c.message.content},
                             "text": c.message.content, # Include 'text' key as well
                             "finish_reason": getattr(c, 'finish_reason', None)
                         } for c in response.choices
                     ],
                     "usage": response.usage.dict() if hasattr(response, 'usage') and hasattr(response.usage, 'dict') else None,
                     # "_response": response # 可選：包含原始響應
                 }
                 return response_dict
            else:
                 print(f"Error: Unexpected liteLLM response structure: {response}")
                 return {"error": "Unexpected response structure", "choices": []}
        except Exception as e:
            print(f"Error during basic_request liteLLM call: {e}")
            traceback.print_exc()
            return {"error": str(e), "choices": []}

    # --- __call__ 作為 basic_request 的簡單封裝 ---
    # 它接收來自 Predict/CoT 的 prompt (理論上) 和其他 kwargs
    def __call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False,
                 **kwargs) -> List[str]:
        """
        Primary interface expected by DSPy modules like Predict.
        Delegates to basic_request and extracts completions.
        """
        # print(f"LiteLLMWrapper __call__ called (prompt len={len(prompt)})") # Debug

        # 將生成參數傳遞給 basic_request
        response_dict = self.basic_request(prompt=prompt, **kwargs)

        # 從 basic_request 的返回字典中提取完成的文本列表
        completions = []
        if response_dict and isinstance(response_dict.get('choices'), list):
            for choice in response_dict['choices']:
                # 優先使用 message.content，備用 text
                content = choice.get('message', {}).get('content')
                if content is None:
                    content = choice.get('text')

                if content is not None:
                    completions.append(content)
                else:
                    # 即使有 choice 但沒有 content/text
                    completions.append("[ERROR: No text found in choice]")
        elif response_dict and response_dict.get('error'):
             completions.append(f"[ERROR: basic_request failed - {response_dict['error']}]")
        else: # 其他無效響應
             completions.append("[ERROR: Invalid response from basic_request]")

        return completions if completions else ["[ERROR: No completions generated]"]

    # request_multimodal 可以保持不變，或考慮是否也應基於 basic_request (較難)
    def request_multimodal(self, messages: List[Dict[str, Any]], **kwargs) -> List[str]:
        print("LiteLLMWrapper request_multimodal called.")
        return self._call_litellm(messages=messages, **kwargs) # Keep direct call for now

    def _call_litellm(self, messages: List[Dict[str, Any]], **kwargs) -> List[str]:
        # This is now primarily used by request_multimodal
        merged_kwargs = {**self.kwargs, **kwargs}
        litellm_params = {"model": self.model_name, "messages": messages,
                         "temperature": merged_kwargs.get("temperature"), "max_tokens": merged_kwargs.get("max_tokens"),
                         "stop": merged_kwargs.get("stop"), "top_p": merged_kwargs.get("top_p"),
                         "api_key": self.api_key if self.api_key else None, }
        litellm_params = {k: v for k, v in litellm_params.items() if v is not None}
        try: response = litellm.completion(**litellm_params)
        except Exception as e: print(f"Error calling liteLLM: {e}"); traceback.print_exc(); return [f"[ERROR: {e}]"]
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            return [response.choices[0].message.content]
        else: print(f"Error: Unexpected liteLLM response: {response}"); return ["[ERROR: Unexpected Response]"]

    def get_history(self, **kwargs): return self.history

# --- 3. 文本載入 ---
def load_text(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: print(f"錯誤：找不到文件 {filepath}"); return None
    except Exception as e: print(f"讀取文件 {filepath} 時發生錯誤: {e}"); return None

# --- 4. 資料清洗函數 (需要您實現) ---
def clean_and_structure_ocr_text(ocr_text: str) -> str:
    """
    應用您的資料清洗和整理邏輯到 OCR 文本上。
    重要：確保清洗後的文本保留可用於分割的、格式一致的題目邊界標記 (例如 '1. ', '2. ')。
    """
    print("正在對 OCR 文本進行清洗和整理...")
    # ************************************************************
    # *** 在這裡實現您的資料清洗和整理邏輯 ***
    cleaned_text = ocr_text # 暫時不做操作，返回原文本
    # ... 您的清洗規則 ...
    # 例如：確保題號格式為 '數字. '
    # cleaned_text = re.sub(r'\n(\d+)\.\s*', r'\n\n\1. ', cleaned_text) # 確保題號前有空行
    # ************************************************************
    print("清洗整理完成。")
    return cleaned_text


# --- 5. NEW: 按 '數字.' 標記分割文本的函數 (再次修正正則表達式) ---
def split_text_by_question_markers(cleaned_text: str, questions_data: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    根據問題標記 (如 '1. ', '2. ') 將清洗後的文本分割成與每個問題對應的片段。
    使用 \s* 允許多種空格情況。
    """
    print("正在按問題標記 ('數字.') 分割清洗後的文本...")
    question_texts = {}
    # --- !!! 使用修正後的正則表達式 !!! ---
    # 將 \s+ 修改為 \s* 以匹配零個或多個空格
    markers = list(re.finditer(r'^(\d+)\.\s*', cleaned_text, re.MULTILINE))
    # --- !!! 正則表達式結束 !!! ---

    if not markers:
        print("警告：在清洗後的文本中未找到任何 '數字.' 格式的題目標記！")
        return {}

    print(f"找到 {len(markers)} 個標記。第一個標記是: '{markers[0].group(0).strip()}' 在位置 {markers[0].start()}")

    # ... (函數的其餘部分保持不變) ...
    num_markers = len(markers)
    for i, marker in enumerate(markers):
        try:
            q_num_str = marker.group(1)
            # print(f"  正在處理標記 {i+1}/{num_markers}: '{marker.group(0)}', 捕獲題號: '{q_num_str}'") # Debug
            q_num = int(q_num_str)
            start_pos = marker.start() # 從標記的行首開始
            end_pos = markers[i+1].start() if (i + 1) < num_markers else len(cleaned_text)
            question_segment = cleaned_text[start_pos:end_pos].strip()
            # print(f"    題號 {q_num}: 文本片段起止位置 {start_pos}-{end_pos}, 長度 {len(question_segment)}") # Debug
            question_texts[q_num] = question_segment
        except ValueError: print(f"警告：無法從標記 '{marker.group(0)}' 提取有效題號。")
        except Exception as e: print(f"分割問題 {marker.group(1)} 文本時出錯: {e}"); question_texts[int(marker.group(1))] = f"[文本分割錯誤: {e}]"

    print(f"文本分割完成，字典鍵: {list(question_texts.keys())}")

    all_found = True
    for q_data in questions_data:
        q_num = q_data.get('number')
        if q_num not in question_texts:
            print(f"警告：問題 {q_num} 未能在 question_texts 字典中找到。")
            question_texts[q_num] = "[未找到對應的文本片段]"
            all_found = False
    if all_found: print("所有解析出的問題都找到了對應的文本片段。")

    return question_texts

# --- 6. 解析問題文件的函數 (保持不變) ---
def parse_questions_from_file(filepath: str) -> List[Dict[str, Any]]:
    # ... (與之前版本相同) ...
    print(f"正在從文件解析問題: {filepath}")
    questions = []; current_question = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        blocks = content.split('--------------------')
        for block_idx, block in enumerate(blocks):
            block = block.strip();
            if not block: continue
            question_data = {'number': None, 'stem': None, 'options': {}}
            q_num_match = re.search(r'^\*\*問題\s*(\d+):\*\*', block, re.MULTILINE) # 仍按原格式解析問題文件
            if q_num_match: question_data['number'] = int(q_num_match.group(1))
            stem_match = re.search(r'\*\*題目:\*\*(.*?)\*\*選項:\*\*', block, re.DOTALL | re.MULTILINE)
            if stem_match: question_data['stem'] = stem_match.group(1).strip()
            else: print(f"警告：塊 {block_idx+1} 無 '題目:'/'選項:'。"); continue
            option_lines = re.findall(r'^\s*-?\s*\(([ABCD])\)\s*(.*)', block, re.MULTILINE)
            if option_lines:
                for letter, text in option_lines: question_data['options'][letter] = text.strip()
            else: print(f"警告：塊 {block_idx+1} 無選項。"); continue
            if question_data['stem'] and len(question_data['options']) == 4:
                questions.append(question_data)
            else: print(f"警告：跳過不完整的塊 {block_idx+1}。")
    except FileNotFoundError: print(f"錯誤：找不到問題文件 {filepath}")
    except Exception as e: print(f"解析問題文件 {filepath} 時出錯: {e}"); traceback.print_exc()
    print(f"總共解析出 {len(questions)} 個有效的選擇題。")
    return questions

# --- 7. 選擇題解釋器簽名與模塊 (使用 Predict, 保持不變) ---
class ExplainMcqSignature(dspy.Signature):
    """[保持不變]"""
    context = dspy.InputField(desc="與本題相關的教材文本片段。")
    question_stem = dspy.InputField(desc="選擇題的題幹部分。")
    option_a = dspy.InputField(desc="選項 A 的內容。")#... (其他選項)
    option_b = dspy.InputField(desc="選項 B 的內容。")
    option_c = dspy.InputField(desc="選項 C 的內容。")
    option_d = dspy.InputField(desc="選項 D 的內容。")
    explanation_a = dspy.OutputField(desc="對選項 A 的詳細解釋...")#... (其他解釋)
    explanation_b = dspy.OutputField(desc="對選項 B 的詳細解釋...")
    explanation_c = dspy.OutputField(desc="對選項 C 的詳細解釋...")
    explanation_d = dspy.OutputField(desc="對選項 D 的詳細解釋...")
    correct_option = dspy.OutputField(desc="最終判斷出的正確選項字母 (A, B, C 或 D)。")

class McqExplainer(dspy.Module):
    """使用 Predict 生成選擇題各選項的解釋並判斷答案"""
    def __init__(self):
        super().__init__()
        self.generate_explanation = dspy.Predict(ExplainMcqSignature)
    def forward(self, context: str, question_stem: str,
                option_a: str, option_b: str, option_c: str, option_d: str) -> dspy.Prediction:
        result = self.generate_explanation(context=context, question_stem=question_stem,
                                           option_a=option_a, option_b=option_b,
                                           option_c=option_c, option_d=option_d)
        return result

# --- 8. 主執行流程 (修改為 清洗+按題分割+逐題處理) ---
if __name__ == "__main__":
    print("\n--- 清洗+按題分割 選擇題批量解釋生成器 (無 RAG) ---")
    # --- 配置 ---
    question_filepath = "generated_materials.txt"       # 包含結構化問題的文件
    ocr_text_filepath = "control_group_litellm_image_ocr.txt" # 原始 OCR 文本文件
    cleaned_ocr_filepath = "cleaned_ocr_text.txt"          # 清洗後文本保存路徑
    output_filepath = "batch_explanations_no_rag.txt" # 新的輸出文件名
    llm_model_name = "gemini/gemini-1.5-flash-latest"

    # --- 初始化 LLM 和 DSPy ---
    print("初始化 LiteLLMWrapper...")
    my_llm = LiteLLMWrapper(model_name=llm_model_name)
    print("配置 DSPy...")
    dspy.configure(lm=my_llm)

    # --- 載入並清洗 OCR 文本 ---
    print(f"載入 OCR 文本從: {ocr_text_filepath}")
    document_text = load_text(ocr_text_filepath)
    if document_text is None or not document_text.strip(): print(f"錯誤：載入 OCR 文件失敗。"); exit()
    print(f"文本載入成功，長度: {len(document_text)} 字元。")
    cleaned_text = clean_and_structure_ocr_text(document_text) # 調用清洗函數
    if not cleaned_text.strip(): print("錯誤：清洗後的文本為空。"); exit()
    try: # 保存清洗後文件
        with open(cleaned_ocr_filepath, 'w', encoding='utf-8') as f: f.write(cleaned_text)
        print(f"清洗後的文本已保存到: {cleaned_ocr_filepath}")
    except Exception as save_err: print(f"警告：保存清洗後文件失敗: {save_err}")

    # --- 解析問題文件 ---
    parsed_questions = parse_questions_from_file(question_filepath)
    if not parsed_questions: print("錯誤：未能解析出問題。"); exit()

    # --- 按問題標記分割清洗後的文本 ---
    question_segments = split_text_by_question_markers(cleaned_text, parsed_questions)
    if not question_segments: print("錯誤：無法按問題分割文本。"); exit()

    # --- 初始化解釋器模塊 ---
    print("\n初始化 MCQ 解釋模塊 (使用 Predict)...")
    mcq_explainer_module = McqExplainer()
    print("--- 系統初始化完成 (無 RAG 檢索器) ---")

   # --- 處理所有問題並收集結果 ---
    print(f"\n--- 開始處理 {len(parsed_questions)} 個問題 ---")
    all_results_text = []
    start_overall_time = time.time()

    for i, question_data in enumerate(parsed_questions):
        q_num = question_data.get('number', i + 1)
        print(f"\n處理問題 {q_num}/{len(parsed_questions)}: {question_data['stem'][:50]}...")
        start_q_time = time.time()

        context_segment = None # 先設為 None
        formatted_output = f"**問題 {q_num}:**\n**題目:** {question_data['stem']}\n" # 預先準備輸出頭部

        try:
            # --- 獲取該問題對應的文本片段 ---
            print(f"  Attempting to get segment using key: {q_num}")
            context_segment = question_segments.get(q_num, None)

            # --- 檢查獲取的片段是否有效 ---
            if context_segment is None:
                raise ValueError("[未找到對應的文本片段]") # 拋出異常以便下面統一處理
            if "[文本分割錯誤:" in context_segment:
                 raise ValueError(context_segment) # 傳遞分割錯誤信息

            print(f"  上下文片段長度: {len(context_segment)} 字元。")
            # ... (可選的長度警告) ...

            # --- !!! 使用 .forward() 調用解釋器模塊 !!! ---
            response_prediction = mcq_explainer_module.forward(
                context=context_segment,
                question_stem=question_data['stem'],
                option_a=question_data['options'].get('A', '[N/A]'),
                option_b=question_data['options'].get('B', '[N/A]'),
                option_c=question_data['options'].get('C', '[N/A]'),
                option_d=question_data['options'].get('D', '[N/A]')
            )
            # --- !!! 調用結束 !!! ---

            # 提取結果
            explanation_a = getattr(response_prediction, 'explanation_a', "[未能生成解釋 A]")
            explanation_b = getattr(response_prediction, 'explanation_b', "[未能生成解釋 B]")
            explanation_c = getattr(response_prediction, 'explanation_c', "[未能生成解釋 C]")
            explanation_d = getattr(response_prediction, 'explanation_d', "[未能生成解釋 D]")
            correct_option = getattr(response_prediction, 'correct_option', "[未能判斷]")

            # 格式化成功結果
            formatted_output += "**選項:**\n"
            formatted_output += f"- (A) {question_data['options']['A']}\n"
            formatted_output += f"- (B) {question_data['options']['B']}\n"
            formatted_output += f"- (C) {question_data['options']['C']}\n"
            formatted_output += f"- (D) {question_data['options']['D']}\n"
            formatted_output += "**生成解釋:**\n"
            formatted_output += f"選項 A 解釋: {explanation_a}\n"
            formatted_output += f"選項 B 解釋: {explanation_b}\n"
            formatted_output += f"選項 C 解釋: {explanation_c}\n"
            formatted_output += f"選項 D 解釋: {explanation_d}\n"
            formatted_output += f"判斷出的正確選項: {correct_option}\n"
            end_q_time = time.time()
            print(f"問題 {q_num} 處理完成 (耗時: {end_q_time - start_q_time:.2f} 秒)")

        except Exception as e:
            # 統一處理循環中發生的所有錯誤 (包括獲取上下文失敗、調用LLM失敗等)
            print(f"處理問題 {q_num} 時發生錯誤: {e}")
            traceback.print_exc() # 打印詳細錯誤
            # 格式化錯誤結果
            formatted_output += "**處理時發生錯誤:**\n"
            formatted_output += f"{traceback.format_exc()}\n" # 將完整 traceback 記錄到文件

        finally:
             # 無論成功或失敗，都將結果添加到列表
             all_results_text.append(formatted_output)
             # 可選延遲
             # time.sleep(1)

    end_overall_time = time.time()
    print(f"\n--- 所有問題處理完成，總耗時: {end_overall_time - start_overall_time:.2f} 秒 ---")

    # --- 將所有結果寫入新文件 ---
    final_output_string = "\n\n--------------------\n\n".join(all_results_text)
    print(f"\n準備將所有結果寫入文件: {output_filepath}...")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f: f.write(final_output_string)
        print(f"已成功將所有解釋結果儲存到 {output_filepath}")
    except IOError as e: print(f"錯誤：無法寫入輸出文件 {output_filepath}: {e}"); traceback.print_exc()
    except Exception as e: print(f"儲存輸出文件時發生未預期的錯誤: {e}"); traceback.print_exc()

    print("\n腳本執行完畢。")