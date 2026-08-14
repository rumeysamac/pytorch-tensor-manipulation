import torch

# 1. Adım: İlk tensörün oluşturulması (32, 10, 64)
input_tensor = torch.randn(32, 10, 64)
print("1. İlk Tensör Boyutu:", input_tensor.shape)

# 2. Adım: Boyutun (320, 64) haline getirilmesi
reshaped_tensor = input_tensor.reshape(320, 64)
print("2. Yassılaştırılmış Tensör Boyutu:", reshaped_tensor.shape)

# 3. Adım: Ağırlık matrisi oluşturma ve matris çarpımı
weight = torch.randn(64, 128)
output_tensor = reshaped_tensor @ weight
print("3. Matris Çarpımı Sonrası Boyut:", output_tensor.shape)

# 4. Adım: Yeniden 3D boyuta getirilmesi
final_tensor = output_tensor.reshape(32, 10, 128)
print("4. Son Tensör Boyutu:", final_tensor.shape)