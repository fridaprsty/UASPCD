import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def median_filter(image, kernel_size):
    # Mengambil ukuran gambar
    height, width = image.shape[:2]

    # Menghitung offset kernel
    offset = kernel_size // 2

    # Membuat array kosong untuk citra hasil median filter
    result = np.zeros(image.shape, dtype=np.uint8)

    # Memproses setiap piksel pada citra
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            # Mengambil nilai piksel pada area kernel
            neighborhood = image[i-offset:i+offset+1, j-offset:j+offset+1]

            # Menghitung median nilai piksel pada area kernel
            median_value = np.median(neighborhood, axis=(0, 1))

            # Menyimpan nilai median ke citra hasil
            result[i, j] = median_value

    return result

def calculate_mse(original, processed):
    # Menghitung selisih antara citra asli dan citra hasil
    diff = original.astype("float") - processed.astype("float")

    # Menghitung MSE
    mse = np.mean(diff ** 2)

    return mse

def calculate_psnr(original, processed):
    # Menghitung MSE
    mse = calculate_mse(original, processed)

    # Menghitung nilai maksimum piksel
    max_pixel = np.max(original)

    # Menghitung PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

# Baca citra asli
original_image = cv2.imread("input_image.jpg")

# Periksa apakah citra berhasil dibaca
if original_image is None:
    print("Gagal membaca citra")
    exit()

# Konversi citra ke format RGB
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Terapkan median filter dengan kernel 3x3
filtered_image = median_filter(original_image_rgb, kernel_size=3)

# Hitung MSE dan PSNR dari asli ke citra dengan filter median
mse_median = calculate_mse(original_image_rgb, filtered_image)
psnr_median = calculate_psnr(original_image_rgb, filtered_image)

# Membuat citra dengan efek blur
blur_image = cv2.GaussianBlur(original_image_rgb, (51, 51), 0)

# Menampilkan citra dan histogram dalam satu frame
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Menampilkan citra asli
plt.subplot(1, 3, 1)
plt.imshow(original_image_rgb)
plt.title("Citra Asli")


# Menampilkan citra hasil median filter (blur)
plt.subplot(1, 3, 3)
plt.imshow(blur_image)
plt.title(f"Median Filter (MSE: {mse_median:.2f}, PSNR: {psnr_median:.2f})")

# Menampilkan frame
plt.tight_layout()
plt.show()
