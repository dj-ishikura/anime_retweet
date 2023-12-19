import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader

def get_anime_ids_from_file(file_path):
    with open(file_path, 'r') as f:
        anime_ids = [line.split(":")[0].strip() for line in f.readlines()]
    return anime_ids

def create_pdf(anime_ids, png_directory, output_pdf):
    # アニメIDに関連するPNGファイルを見つけるのだ
    image_files = [os.path.join(png_directory, f"{anime_id}.png") for anime_id in anime_ids]
    
    # 実際に存在するPNGファイルだけを取得するのだ
    existing_image_files = [img for img in image_files if os.path.exists(img)]

    if existing_image_files:
        width, height = landscape(letter)
        c = canvas.Canvas(output_pdf, pagesize=(width, height))
        for image_file in existing_image_files:
            img = ImageReader(image_file)
            iw, ih = img.getSize()
            aspect = ih / float(iw)
            
            # キャンバスのアスペクト比を取得するのだ
            canvas_aspect = height / float(width)
            
            # 画像とキャンバスのアスペクト比を比較して、どの軸に合わせてリサイズするかを決定するのだ
            if aspect > canvas_aspect:
                # 画像の高さをキャンバスの高さに合わせて、幅をアスペクト比を保持しながら調整するのだ
                img_width = width * canvas_aspect / aspect
                img_height = height
            else:
                # 画像の幅をキャンバスの幅に合わせて、高さをアスペクト比を保持しながら調整するのだ
                img_width = width
                img_height = height * aspect / canvas_aspect
            
            # 画像をキャンバスの中央に配置するためのオフセットを計算するのだ
            x_offset = (width - img_width) / 2
            y_offset = (height - img_height) / 2

            # 画像を描画するのだ
            c.drawImage(image_file, x_offset, y_offset, width=img_width, height=img_height)
            c.showPage()
        c.save()

    else:
        print(f"失敗: {png_directory} で指定されたアニメIDのPNGファイルが見つかりませんでした。")

if __name__ == "__main__":
    text_files = ["result/trend_anime_list.txt", "result/hit_anime_list.txt", "result/miner_anime_list.txt"]
    png_directory = "weekly_anime_network_stat"
    
    for text_file in text_files:
        anime_ids = get_anime_ids_from_file(text_file)
        output_pdf = os.path.splitext(text_file)[0] + "_from_weekly_network.pdf"
        create_pdf(anime_ids, png_directory, output_pdf)


