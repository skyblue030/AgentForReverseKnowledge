import dspy
from PyPDF2 import PdfReader
import re
import json

# 配置OllamaLocal模型，确保base_url正确
ollama = dspy.OllamaLocal(model='llama3:70b',
                          base_url='https://8080-01hxgf0zxany62p2fk6s1kj9p8.cloudspaces.litng.ai',
                          max_tokens=300)
dspy.configure(lm=ollama)

# 定义Signature
class GenerateAnswer(dspy.Signature):
    question = dspy.InputField(description="題目")
    options = dspy.InputField(description="選項")
    explanation = dspy.OutputField(description="教材內容")

# 定义生成教材内容的ChainOfThought模块
class GenerateMaterial(dspy.ChainOfThought):
    def generate_steps(self, inputs):
        question = inputs['question']
        options = inputs['options']
        steps = []

        # Chain of Thought prompts
        steps.append(f"題目：{question}\n")
        steps.append("答案選項：")
        steps.extend(options)
        steps.append("\n教材編寫：\n")

        # 逐步推理生成内容
        thought_prompts = [
            f"題目：{question}\n答案選項：{' '.join(options)}",
            "這是一個關於計算機科學的問題。",
            "我們需要解析題目中的關鍵詞來理解問題。",
            "根據關鍵詞生成相應的解釋內容。"
        ]
        
        # 用於存放模型生成的內容
        generated_explanations = []

        for prompt in thought_prompts:
            # 调试信息：打印请求数据
            print(f"请求数据：{prompt}")
            response = ollama.basic_request(prompt)
            # 调试信息：打印响应数据
            print(f"响应数据：{response}")

            try:
                # 处理响应数据，确保是字典格式
                response_json = response if isinstance(response, dict) else json.loads(response)
                text = response_json['choices'][0]['message']['content']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"JSON解析错误或键错误：{e}")
                print(f"原始响应文本：{response}")
                text = "无效的响应格式。"

            generated_explanations.append(text)
            steps.append(text)
        
        return steps

# 创建ChainOfThought模块实例，传递GenerateAnswer这个signature
chain_of_thought = GenerateMaterial(signature=GenerateAnswer)

# 读取PDF文件并提取文本内容
file_path = '/teamspace/studios/this_studio/子偉PPT/計算機概論.pdf'
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

# 逐一处理每个问题并生成对应的教材内容
for question in questions:
    print("处理问题:", question)  # 调试信息
    lines = question.split('\n')
    question_text = lines[0].strip()
    options = [line.strip() for line in lines[1:] if line.strip()]

    # 使用ChainOfThought模块生成教材内容
    inputs = {'question': question_text, 'options': options}
    steps = chain_of_thought.generate_steps(inputs)

    # 打印生成的教材内容（调试信息）
    for step in steps:
        print(step)

    # 将生成的教材内容保存到文件中
    with open('generated_materials.txt', 'a', encoding='utf-8') as file:
        for step in steps:
            file.write(str(step) + "\n")

print("教材内容已保存到generated_materials.txt文件中。")
