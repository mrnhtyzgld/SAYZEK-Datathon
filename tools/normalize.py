import os
from PIL import Image


def normalize_labels(label_dir, image_dir, image_extension=".png"):
    """
    Bu fonksiyon, verilen etiket dizininde bulunan tüm etiket dosyalarındaki bounding box koordinatlarını normalize eder.

    Args:
        label_dir (str): Etiket dosyalarının bulunduğu dizin.
        image_dir (str): İlgili görüntülerin bulunduğu dizin.
        image_extension (str): Görüntü dosyalarının uzantısı (varsayılan olarak ".png").

    """
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            # İlgili görüntü dosyasını bul
            image_file = os.path.splitext(label_file)[0] + image_extension
            image_path = os.path.join(image_dir, image_file)

            if not os.path.exists(image_path):
                print(f"Warning: {image_file} bulunamadı, bu etiket dosyası işlenmedi.")
                continue

            # Görüntü boyutlarını elde et
            with Image.open(image_path) as img:
                width, height = img.size

            # Normalizasyon yap
            normalized_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])-1
                    x_center = float(parts[1]) / width
                    y_center = float(parts[2]) / height
                    box_width = float(parts[3]) / width
                    box_height = float(parts[4]) / height
                    normalized_lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

            # Dosyayı güncelle
            with open(label_path, 'w') as f:
                f.write("\n".join(normalized_lines))
            print(f"{label_file} dosyası başarıyla normalize edildi.")


# Klasör yollarını belirtin
label_directory = r"C:\Users\NihatEmreYüzügüldü\PycharmProjects\SAYZEK\data\model_data\val"  # Etiket dosyalarının olduğu dizin
image_directory = r"C:\Users\NihatEmreYüzügüldü\PycharmProjects\SAYZEK\data\model_data\val"  # İlgili görüntülerin olduğu dizin

normalize_labels(label_directory, image_directory)
