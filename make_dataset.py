from PIL import Image, ImageDraw
import os

out_dir = "demo_dataset"
os.makedirs(out_dir, exist_ok=True)

# Ảnh 1: hình vuông đỏ
img1 = Image.new("RGB", (128, 128), color="red")
img1.save(os.path.join(out_dir, "red_square.jpg"))

# Ảnh 2: hình vuông xanh
img2 = Image.new("RGB", (128, 128), color="blue")
img2.save(os.path.join(out_dir, "blue_square.jpg"))

# Ảnh 3: giống ảnh 1 (để test phát hiện trùng)
img3 = Image.new("RGB", (128, 128), color="red")
img3.save(os.path.join(out_dir, "red_square_copy.jpg"))

# Ảnh 4: hình tròn xanh trên nền trắng
img4 = Image.new("RGB", (128, 128), color="white")
draw = ImageDraw.Draw(img4)
draw.ellipse((32, 32, 96, 96), fill="blue")
img4.save(os.path.join(out_dir, "blue_circle.jpg"))

print(f"Dataset saved to {out_dir}")
