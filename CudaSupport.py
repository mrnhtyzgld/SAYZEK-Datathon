import torch
print("CUDA available:", torch.cuda.is_available())

print("CUDA mevcut mu:", torch.cuda.is_available())
print("CUDA versiyonu:", torch.version.cuda)
print("GPU sayısı:", torch.cuda.device_count())
print("Cihaz adı:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU bulunamadı")
