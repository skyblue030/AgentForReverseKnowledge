import dspy
from PyPDF2 import PdfReader
import re
import litellm
import os
import warnings
import json
from dotenv import load_dotenv 

class LiteLLMWrapper(dspy.LM):
    # --- 【修正】__init__ 方法 ---
    def __init__(self, model_name="gemini/gemini-1.5-flash-latest", api_key=None, **kwargs):
        super().__init__(model_name) # Pass model_name to parent LM class if needed
        self.provider = "gemini"
        self.model_name = model_name # Store the model name
        self.kwargs = {'temperature': 0.7, 'max_tokens': 3000, **kwargs} # Default generation kwargs
        self.history = []

        # --- 修正後的 API 金鑰獲取邏輯 ---
        effective_api_key = None
        # 1. 優先使用直接傳入的 api_key 參數
        if api_key:
            effective_api_key = api_key
            print("DEBUG: Using API key passed as parameter.")

        # 2. 如果未傳入參數，檢查環境變數 GEMINI_API_KEY (LiteLLM 常用)
        if not effective_api_key:
            effective_api_key = os.getenv("GEMINI_API_KEY")
            if effective_api_key:
                 print("DEBUG: Found API key in GEMINI_API_KEY environment variable.")

        # 3. 如果還是沒有，檢查環境變數 GOOGLE_API_KEY (有些用戶可能用這個)
        if not effective_api_key:
             effective_api_key = os.getenv("GOOGLE_API_KEY")
             if effective_api_key:
                  print("DEBUG: Found API key in GOOGLE_API_KEY environment variable.")

        # 4. 如果環境變數都沒有，最後嘗試從 .env 檔案載入
        if not effective_api_key:
             print("DEBUG: API key not found in params or env vars, attempting to load from .env file...")
             load_dotenv() # 載入 .env 文件
             effective_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") # 再次檢查環境變數
             if effective_api_key:
                  print("DEBUG: Found API key after loading .env file.")

        # 5. 最終確認並設定
        if effective_api_key:
            self.api_key = effective_api_key
            # 設定給 litellm 可能不再嚴格必要，因為我們每次都傳遞
            # 但設定一下也無妨
            litellm.api_key = self.api_key
            print(f"成功獲取 API Key 並配置 LiteLLMWrapper (使用模型: {self.model_name})。")
        else:
            # 如果最終還是找不到 Key，直接拋出錯誤，因為無法繼續
            raise ValueError("【錯誤】未找到 Gemini/Google API Key。請透過參數、環境變數 (建議 GEMINI_API_KEY) 或 .env 文件提供。")

    # --- basic_request 和 __call__ 方法保持不變 (使用上一版本即可) ---


    def basic_request(self, prompt: str = None, messages: list = None, **kwargs):
        """
        Performs a request to litellm.completion.
        Prioritizes 'messages' if provided, otherwise constructs messages from 'prompt'.
        Filters kwargs to pass only valid ones to litellm, excluding 'model'.
        """
        # 合併 kwargs
        combined_kwargs = {**self.kwargs, **kwargs}

        # 定義支援的參數，【移除 'model'】
        allowed_params = {
             "messages", "temperature", "max_tokens", "top_p", "n", # 移除了 "model"
             "stream", "stop", "presence_penalty", "frequency_penalty",
             "logit_bias", "user", "api_key", "api_base", "api_version", "response_format",
        }
        filtered_kwargs = {k: v for k, v in combined_kwargs.items() if k in allowed_params}

        # 決定 final_messages (邏輯不變)
        final_messages = None
        if messages:
            final_messages = messages
        elif prompt:
            final_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided to basic_request.")

        # print(f"DEBUG: Calling litellm.completion with model: {self.model_name}")
        # print(f"DEBUG: Final messages being sent: {json.dumps(final_messages, indent=2, ensure_ascii=False)}")
        # print(f"DEBUG: Raw combined kwargs received by basic_request: {combined_kwargs}")
        # print(f"DEBUG: Filtered kwargs passed to litellm: {filtered_kwargs}")

        try:
            response = litellm.completion(
                model=self.model_name,     # 明確傳遞 model
                messages=final_messages,
                api_key=self.api_key,
                **filtered_kwargs       # 傳遞過濾後 (且不含 model) 的 kwargs
            )
            return response
        except Exception as e:
            print(f"錯誤：呼叫 litellm.completion 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None # 返回 None 表示失敗


    def __call__(self, **kwargs):
        """
        Handles LM calls using keyword arguments passed by DSPy.
        Extracts 'messages' or 'prompt' from kwargs to pass to basic_request.
        Always returns the raw completion string(s) in a list.
        """
        # 從 kwargs 中提取並移除 messages 或 prompt
        messages = kwargs.pop('messages', None)
        prompt = kwargs.pop('prompt', None)

        if not messages and not prompt:
            print("錯誤：在傳遞給 LiteLLMWrapper.__call__ 的 kwargs 中既未找到 'prompt' 也未找到 'messages'。")
            print(f"收到的 kwargs: {kwargs}")
            return [""] # 返回包含空字串的列表

        # 處理 n 參數 (目前僅示意)
        n = kwargs.pop('n', 1)
        if n > 1:
             warnings.warn("LiteLLMWrapper 目前簡化處理，未完全支援 n > 1 的情況，僅返回第一個結果。")
        # 移除其他已知由 __call__ 處理的參數
        kwargs.pop('only_completed', None)
        kwargs.pop('return_sorted', None)

        # 呼叫 basic_request
        response_obj = self.basic_request(prompt=prompt, messages=messages, **kwargs)

        # 提取主要的回應文本
        completion_text = "" # 預設為空字串
        if response_obj is None:
            print("錯誤：basic_request 未能從 litellm 獲取有效回應。")
        else:
            try:
                if response_obj.choices and len(response_obj.choices) > 0:
                     completion_text = response_obj.choices[0].message.content.strip()
                else:
                     print("警告：LiteLLM 回應中沒有有效的 choices。")
                     print(f"完整回應物件: {response_obj}")
            except (AttributeError, IndexError, TypeError, KeyError) as e:
                print(f"錯誤：無法從 LiteLLM 回應中提取內容: {e}")
                print(f"完整回應物件: {response_obj}")

        # **【關鍵】: 始終返回包含原始文本字串的列表**
        # 將 JSON 解析完全交還給 DSPy 的 Adapter
        # print(f"DEBUG: LiteLLMWrapper.__call__ returning: {[completion_text]}") # 可用於除錯
        return [completion_text]


print("正在嘗試配置 Gemini 模型 (使用自訂 Wrapper)...")
gemini_model_name = "gemini/gemini-1.5-flash-latest"
# API 金鑰交由 LiteLLMWrapper 自行解析 (參數 -> GEMINI_API_KEY -> GOOGLE_API_KEY -> .env)，
# 完全找不到時會在 __init__ 直接拋出 ValueError。
llm = LiteLLMWrapper(model_name=gemini_model_name)

# 配置 DSPy 使用成功初始化的 Gemini 模型
dspy.configure(lm=llm)
print("DSPy 已成功配置使用 Gemini 模型 (透過自訂 Wrapper)。")

# 定義Signature
# --- 【修改後的 Signature】 ---
class GenerateAnswer(dspy.Signature):
    """
    【指令】請針對以下計算機科學選擇題及其選項，用【英文】撰寫一段清晰、準確、易於學生理解的【教材風格】解釋。
    請說明題目涉及的核心概念，解釋為何正確選項是對的，並分析其他選項錯誤的原因。請注重原理講解。
    """
    question = dspy.InputField(desc="一個計算機科學選擇題的題幹文字。")
    options = dspy.InputField(desc="該選擇題的所有選項列表。")
    explanation = dspy.OutputField(desc="將題目翻譯成英文，進行一次換行，再一段針對問題核心概念的清晰、準確的原理講解(英文)，包含對各選項的對錯分析。")



# 定义生成教材内容的ChainOfThought模块
class GenerateMaterial(dspy.Module):
    def __init__(self):
        super().__init__()
        # 在初始化方法中，創建一個 dspy.Predict 實例
        # 它會使用我們定義的 GenerateAnswer Signature
        self.generate_explanation = dspy.Predict(GenerateAnswer)

    def forward(self, question, options):
        """
        這是 dspy.Module 的標準執行方法。
        它接收簽名中定義的輸入欄位作為參數。
        """
        # 調用 self.generate_explanation (它是一個 dspy.Predict 模塊)
        # DSPy 會自動處理提示生成和與 LM 的互動
        # 它會返回一個包含簽名中定義的輸出欄位的物件 (dspy.Prediction)
        prediction = self.generate_explanation(question=question, options=options)

        # 從返回的 prediction 物件中提取我們需要的 explanation 欄位
        return prediction.explanation

print(f"DSPy configured to use LiteLLM with model: {llm.model_name}")
generator = GenerateMaterial()

# --- 試卷解析 ---
# 每頁的頁腳與「※尚有試題…※」翻頁提示會被夾在題目之間，必須先移除；
# 舊版是直接切到第一個「※尚有試題」為止，結果只處理到第 4 題就停了。
BANNER_PATTERN = re.compile(r'※[^※\n]*※')
FOOTER_PATTERN = re.compile(r'第\s*\d+\s*頁\s*，\s*共\s*\d+\s*頁')
QUESTION_MARKER_PATTERN = re.compile(r'(?m)^[ \t]*(\d{1,2})[ \t]*\.[ \t]*')
OPTION_SPLIT_PATTERN = re.compile(r'\(([A-D])\)')

# PyPDF2 會在中文字元之間插入空白，收斂後再把中文之間、全形括號內側的空格去掉。
_CJK = r'①-⓿　-〿一-鿿＀-￯'
_JOIN_CJK_PATTERN = re.compile(rf'(?<=[{_CJK}]) (?=[{_CJK}])')
_TRIM_BRACKET_PATTERN = re.compile(r'(?<=[（「【]) | (?=[）」】，。：？；])')


def normalize_spacing(text):
    """把 PDF 抽出的破碎空白收斂成可讀的一行。"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = _JOIN_CJK_PATTERN.sub('', text)
    return _TRIM_BRACKET_PATTERN.sub('', text)


def parse_questions(pdf_text):
    """把整份試卷解析成 [{'number', 'stem', 'options'}]。

    題目邊界只採用「位於行首、且題號恰好等於下一個預期號碼」的標記，
    因此 3.2G、A8.C16、1..4 這類出現在內文的數字不會被誤判成新題目。
    選項邊界採用 (A)~(D) 標記本身，不再依賴行的位置 —— 舊版把第一行當題幹、
    其餘每行當一個選項，跨行的題幹（例如第 2、3、8 題）因此被截斷。
    """
    body = pdf_text.split("單選擇題", 1)[-1]
    body = FOOTER_PATTERN.sub(' ', BANNER_PATTERN.sub(' ', body))

    markers, expected = [], 1
    for match in QUESTION_MARKER_PATTERN.finditer(body):
        if int(match.group(1)) == expected:
            markers.append((expected, match.start(), match.end()))
            expected += 1

    questions = []
    for index, (number, _, content_start) in enumerate(markers):
        content_end = markers[index + 1][1] if index + 1 < len(markers) else len(body)
        parts = OPTION_SPLIT_PATTERN.split(body[content_start:content_end])
        letters = parts[1::2]
        if letters != ['A', 'B', 'C', 'D']:
            print(f"警告：第 {number} 題的選項標記為 {letters}，不是 A~D，已跳過。")
            continue
        questions.append({
            'number': number,
            'stem': normalize_spacing(parts[0]),
            'options': [f"({letter}) {normalize_spacing(text)}"
                        for letter, text in zip(letters, parts[2::2])],
        })
    return questions


# 读取PDF文件并提取文本内容
file_path = '計算機概論.pdf'
with open(file_path, 'rb') as pdf_file:
    pdf_reader = PdfReader(pdf_file)
    pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)

questions = parse_questions(pdf_text)
if not questions:
    print("无法找到题目部分，请检查PDF内容格式。")
    exit()

print("提取到的问题数量:", len(questions))  # 调试信息

output_filename = 'generated_materials.txt'
with open(output_filename, 'w', encoding='utf-8') as outfile:
    print(f"準備將生成的教材內容保存到 {output_filename} 文件中...")

    # 逐一處理每個問題並生成對應的教材內容
    for question_data in questions:
        number = question_data['number']
        print(f"\n--- 正在處理問題 {number} ---")
        question_text = f"{number}. {question_data['stem']}"
        options = question_data['options']
        if not question_data['stem']: continue # 跳過空問題

        print(f"問題: {question_text}")
        print(f"選項: {options}")

        try: # 【新】增加錯誤處理
            # 【新】直接調用 generator 實例，傳入參數
            # 新：不需要組裝 dict，直接傳參
            # 新：返回的是最終的解釋字串，不是 steps 列表
            explanation_output = generator(question=question_text, options=options)
        
            explanation_text = explanation_output

            if not isinstance(explanation_text, str):
                 print(f"警告：從 generator 收到的結果不是預期的字串，而是 {type(explanation_text)}。內容：{repr(explanation_text)}")
                 # 處理非字串情況，例如轉換或設為預設值
                 explanation_text = str(explanation_text) # 嘗試轉換為字串
            # 【新】打印最終生成的教材內容
            print(f"\n生成的教材內容:\n{explanation_text}")

            # 【新】將問題、選項和最終生成的解釋內容，以更清晰的格式寫入文件
            outfile.write(f"**問題 {number}:**\n")
            outfile.write(f"**題目:** {question_text}\n")
            outfile.write(f"**選項:**\n") # 確保這裡有換行
            for option in options:
                outfile.write(f"- {option}\n")
            outfile.write(f"**教材內容:**\n{explanation_text}\n\n") # 使用 explanation_text
            outfile.write("-" * 20 + "\n\n")

        except Exception as e:
            error_message = f"處理問題 {number} 時發生錯誤: {e}"
            print(error_message)
            # 可以在這裡選擇是否將錯誤訊息也寫入檔案
            # f.write(f"問題 {question_number}: {error_message}\n---\n\n")
            import traceback
            traceback.print_exc() # 打印詳細的 traceback
            # 可以在 outfile 中記錄錯誤
            outfile.write(f"**問題 {number}:**\n")
            outfile.write(f"{question_text}\n")
            outfile.write(f"**選項:**\n")
            for option in options:
                outfile.write(f"- {option}\n")
            outfile.write(f"**教材內容:**\n--- 處理時發生錯誤 ---\n{e}\n\n")
            outfile.write("-" * 20 + "\n\n")

print(f"\n教材內容已全部處理並保存到 {output_filename} 文件中。")
