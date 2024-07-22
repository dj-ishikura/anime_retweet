import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def png_to_pdf(input_dir, output_pdf):
    # 入力ディレクトリ内の全てのpngファイルを取得し、ソート
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    if not png_files:
        print("PNGファイルが見つかりません。")
        return

    # 最初の画像のサイズを取得
    first_image = Image.open(os.path.join(input_dir, png_files[0]))
    img_width, img_height = first_image.size

    # PDFを作成
    c = canvas.Canvas(output_pdf, pagesize=(img_width, img_height))

    for png_file in png_files:
        img_path = os.path.join(input_dir, png_file)
        c.drawImage(img_path, 0, 0, width=img_width, height=img_height)
        c.showPage()

    c.save()
    print(f"PDFファイルが作成されました: {output_pdf}")

if __name__ == "__main__":
    input_directory = "/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/plot_target_word_count_anime"  # PNGファイルがあるディレクトリのパス
    output_pdf = "/work/n213304/learn/anime_retweet_2/analyze_user_entropy/plot/plot_weekly_user_incloud_anime.pdf"  # 出力するPDFファイルの名前

    png_to_pdf(input_directory, output_pdf)