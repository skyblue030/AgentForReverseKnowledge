import dspy
from PyPDF2 import PdfReader
import re
import litellm
import os
import warnings
import json
from dotenv import load_dotenv 

class LiteLLMWrapper(dspy.LM):
    def __init__(self, model_name="gemini/gemini-1.5-flash-latest", api_key=None, **kwargs):
        super().__init__(model_name)
        self.provider = "gemini" # Or determine dynamically if needed
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") # Use provided or env var
        self.kwargs = {'temperature': 0.7, 'max_tokens': 400, **kwargs} # Default kwargs

        if not self.api_key:
             # Attempt to read from env file if not set
             load_dotenv()
             self.api_key = os.getenv("GOOGLE_API_KEY")
             if not self.api_key:
                 print("錯誤：找不到 GOOGLE_API_KEY 環境變數。請確保已設定該環境變數。")
                 # Optionally raise an error or exit if the key is critical
                 # raise ValueError("API key not found.")
             else:
                 # Set environment variable for litellm if loaded from .env
                 os.environ["GOOGLE_API_KEY"] = self.api_key


        # Configure LiteLLM to use the Gemini API key if found
        if self.api_key:
            litellm.api_key = self.api_key
            print("DSPy 已成功配置使用 Gemini 模型。")
        else:
            # Handle case where API key is still missing (e.g., skip configuration)
            print("警告：未找到 Gemini API 金鑰，LiteLLM 可能無法正常運作。")


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


gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key  is None:
    print("錯誤：找不到 GOOGLE_API_KEY 環境變數。請確保已設定該環境變數。")
    # 可以取消下面註解並填入金鑰以快速測試 (不建議提交)
    gemini_api_key = "您的金鑰放這裡"
    if gemini_api_key  is None: # 再次檢查，如果還是 None 就退出
        exit("API 金鑰未設定。")


# 初始化 gemini_lm 變數
gemini_lm = None

print("DSPy 已成功配置使用 Gemini 模型。")
print("正在嘗試配置 Gemini 模型 (使用自訂 Wrapper)...")
gemini_model_name = "gemini/gemini-1.5-flash-latest"
llm = LiteLLMWrapper(model=gemini_model_name)

# 配置 DSPy 使用成功初始化的 Gemini 模型
dspy.configure(lm=llm)
print("DSPy 已成功配置使用 Gemini 模型 (透過自訂 Wrapper)。")

# 定義Signature
class GenerateAnswer(dspy.Signature):
    question = dspy.InputField(description="題目")
    options = dspy.InputField(description="選項")
    explanation = dspy.OutputField(description="教材內容")

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

print(f"DSPy configured to use LiteLLM with model: {gemini_model_name}")
generator = GenerateMaterial()

# 读取PDF文件并提取文本内容
file_path = '計算機概論.pdf'
pdf_file = open(file_path, 'rb')

# 读取PDF文件
pdf_reader = PdfReader(pdf_file)
num_pages = len(pdf_reader.pages)

# 提取所有页面的文本内容
pdf_text = ""
for page_num in range(num_pages):
    page = pdf_reader.pages[page_num]
    pdf_text += page.extract_text()

pdf_file.close()

# 提取问题部分并清理
try:
    questions_text = pdf_text.split("單選擇題")[1].split("※尚有試題")[0].strip()
except IndexError:
    print("无法找到题目部分，请检查PDF内容格式。")
    exit()

# 使用正则表达式提取问题和选项
questions = re.findall(r'(\d+\.\s+.*?(?=\n\d+\.\s)|\d+\.\s+.*?$)', questions_text, re.DOTALL)
questions = [q.strip() for q in questions if q.strip()]

print("提取到的问题数量:", len(questions))  # 调试信息

output_filename = 'generated_materials.txt'
with open(output_filename, 'w', encoding='utf-8') as outfile:
    print(f"準備將生成的教材內容保存到 {output_filename} 文件中...")

    # 逐一處理每個問題並生成對應的教材內容
    for i, question_data in enumerate(questions): # 使用 enumerate 獲取索引
        print(f"\n--- 正在處理問題 {i+1} ---")
        # ... (提取 question_text 和 options 的程式碼，可以加上錯誤檢查) ...
        lines = question_data.strip().split('\n')
        if not lines: continue # 跳過空數據
        question_text = lines[0].strip()
        options = [line.strip() for line in lines[1:] if line.strip()]
        if not question_text: continue # 跳過空問題

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
            outfile.write(f"**問題 {i+1}:**\n")
            outfile.write(f"**題目:** {question_text}\n")
            outfile.write(f"**選項:**\n") # 確保這裡有換行
            for option in options:
                outfile.write(f"- {option}\n")
            outfile.write(f"**教材內容:**\n{explanation_text}\n\n") # 使用 explanation_text
            outfile.write("-" * 20 + "\n\n")

        except Exception as e:
            error_message = f"處理問題 {i + 1} 時發生錯誤: {e}"
            print(error_message)
            # 可以在這裡選擇是否將錯誤訊息也寫入檔案
            # f.write(f"問題 {question_number}: {error_message}\n---\n\n")
            import traceback
            traceback.print_exc() # 打印詳細的 traceback
            # 可以在 outfile 中記錄錯誤
            outfile.write(f"**問題 {i+1}:**\n")
            outfile.write(f"{question_text}\n")
            outfile.write(f"**選項:**\n")
            for option in options:
                outfile.write(f"- {option}\n")
            outfile.write(f"**教材內容:**\n--- 處理時發生錯誤 ---\n{e}\n\n")
            outfile.write("-" * 20 + "\n\n")

print(f"\n教材內容已全部處理並保存到 {output_filename} 文件中。")
