import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def png_to_pdf(directory, output_filename):
    # ディレクトリ内のすべての.pngファイルを取得
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    png_files.sort()  # ファイル名でソート

    if not png_files:
        print("PNGファイルが見つかりません。")
        return

    # PDFキャンバスを作成
    c = canvas.Canvas(output_filename)

    for png_file in png_files:
        img_path = os.path.join(directory, png_file)
        img = Image.open(img_path)
        
        # 画像サイズを取得
        width, height = img.size
        
        # PDFページサイズを画像に合わせる
        c.setPageSize((width, height))
        
        # 画像をPDFに描画
        c.drawImage(img_path, 0, 0)
        
        # 次のページへ
        c.showPage()

    # PDFを保存
    c.save()
    print(f"PDFファイルが作成されました: {output_filename}")

# 使用例
directory = "/work/n213304/learn/anime_retweet_2/analyze_end/count_tweet"  # PNGファイルがあるディレクトリのパス
output_filename = "/work/n213304/learn/anime_retweet_2/analyze_end/result/count_tweet.pdf"  # 出力するPDFファイル名

png_to_pdf(directory, output_filename)