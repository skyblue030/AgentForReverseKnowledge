import fitz # PyMuPDF
import os
import traceback

def extract_and_save_pdf_text(pdf_path, output_txt_path="textLayer.txt"):
    """
    提取 PDF 文件的所有文本層並儲存到指定的文本文件中。

    Args:
        pdf_path (str): 輸入的 PDF 文件路徑。
        output_txt_path (str): 輸出的文本文件路徑。預設為 "textLayer.txt"。
    """
    if not os.path.exists(pdf_path):
        print(f"錯誤：文件不存在 - {pdf_path}")
        return
    # 確保輸入是 PDF 文件
    if not pdf_path.lower().endswith(".pdf"):
        print(f"錯誤：輸入文件 '{pdf_path}' 不是 PDF 文件。")
        return

    doc = None
    all_extracted_text = "" # 用於儲存所有頁面的文本

    try:
        doc = fitz.open(pdf_path)
        print(f"--- 開始處理 PDF: {pdf_path} ---")
        print(f"總頁數: {len(doc)}")

        # 逐頁提取文本
        for i, page in enumerate(doc):
            print(f"--- 正在提取第 {i+1}/{len(doc)} 頁的文本 ---")
            try:
                page_text = page.get_text("text")
                if page_text: # 如果提取到文本
                    cleaned_text = page_text.strip()
                    if cleaned_text: # 確保去除空白後仍有內容
                        all_extracted_text += cleaned_text + "\n\n" # 將當前頁面的文本附加到總文本中，並用兩個換行符分隔頁面
                        print(f"   提取到 {len(cleaned_text)} 字元的文本。")
                    else:
                        print("   該頁提取到的文本去除空白後為空。")
                else:
                    print("   未能從此頁提取到文本 (可能是圖像頁面或空白頁)。")
            except Exception as page_e:
                print(f"   處理第 {i+1} 頁時發生錯誤: {page_e}")
                traceback.print_exc() # 打印詳細錯誤，但繼續處理下一頁

        print("\n--- 所有頁面文本提取完成 ---")

        # 將提取到的所有文本寫入文件
        try:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(all_extracted_text)
            print(f"成功將所有提取的文本儲存到: {output_txt_path}")
        except IOError as e:
            print(f"錯誤：無法寫入文件 {output_txt_path} - {e}")
            traceback.print_exc()
        except Exception as e:
             print(f"儲存文件時發生未知錯誤: {e}")
             traceback.print_exc()

    except fitz.fitz.FileNotFoundError:
         print(f"錯誤：PyMuPDF 找不到文件 - {pdf_path}")
    except fitz.fitz.FileDataError as e:
         print(f"錯誤：無法讀取或解密 PDF 文件 '{pdf_path}' - {e}")
         print("   文件可能已損壞或受密碼保護。")
    except Exception as e:
        print(f"處理 PDF 時發生未預期的錯誤: {e}")
        traceback.print_exc()
    finally:
        if doc:
            try:
                doc.close()
                print(f"已關閉 PDF 文件: {pdf_path}")
            except Exception as close_e:
                 print(f"關閉 PDF 文件時發生錯誤: {close_e}")


# --- 使用 ---
pdf_file = "計算機概論.pdf" # <--- 替換成您的 PDF 文件路徑
output_file = "textLayer.txt" # 指定輸出的文本文件名稱

# 執行提取和儲存功能
extract_and_save_pdf_text(pdf_file, output_file)

# 你也可以測試其他文件 (確保它們是 PDF)
# pdf_file_other = "another_document.pdf"
# extract_and_save_pdf_text(pdf_file_other, "another_document_text.txt")