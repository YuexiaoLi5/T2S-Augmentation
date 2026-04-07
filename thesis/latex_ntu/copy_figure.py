import os
import shutil

src = r"C:\Users\lyx\.cursor\projects\d-T2S-Augmentation\assets\framework_overview.png"
dst_dir = r"d:\T2S-Augmentation\thesis\latex_ntu\assets\figures"
dst = os.path.join(dst_dir, "framework_overview.png")

os.makedirs(dst_dir, exist_ok=True)
shutil.copy2(src, dst)
print(f"Copied to: {dst}")
