import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from skimage.feature import hog
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
def Show_HOG():
    global anh1, anh2

    # Kiểm tra xem đã chọn ảnh chưa
    if anh1 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Chuyển đổi ảnh gốc sang ảnh xám
    anh2 = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)

    # Tính toán HOG descriptors và visualization với tham số đã cho
    features, hog_image = hog(anh2, orientations=18, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), transform_sqrt=False,
                              block_norm='L2-Hys', visualize=True)

    # Rescale intensities để tăng độ tương phản
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    # Hiển thị ảnh gốc trên khung thứ nhất
    show_image(anh1, panel1)

    # Hiển thị ảnh HOG trên khung thứ hai
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.title('Enhanced HOG Image')
    plt.axis('off')
    plt.show()

def choose_image():
    # Hiển thị hộp thoại mở tệp và lấy đường dẫn tệp được chọn
    file_path = filedialog.askopenfilename()

    # Kiểm tra nếu người dùng không chọn tệp
    if not file_path:
        print("Bạn chưa chọn tệp.")
        return

    # Đọc ảnh từ đường dẫn đã chọn
    global anh1, anh2
    anh1 = cv2.imread(file_path)
    if anh1 is None:
        print("Không thể đọc ảnh từ đường dẫn đã chọn.")
        return

    # Chuyển đổi ảnh sang ảnh xám
    anh2 = anh1.copy()

    # Hiển thị ảnh trên cả hai khung
    show_image(anh1, panel1)
    show_image(anh1, panel2)

def show_image(image, panel):
    # Chuyển đổi ảnh từ OpenCV sang định dạng mà Tkinter có thể hiển thị
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Hiển thị ảnh trên khung hình
    panel.config(image=image)
    panel.image = image

def convert_to_gray():
    global anh1
    if anh1 is None:
        print("Bạn chưa chọn ảnh.")
        return
    # Chuyển đổi ảnh sang ảnh xám
    anh1 = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    # Hiển thị ảnh xám trên khung ảnh thứ hai
    show_image(anh1, panel2)

def apply_hog():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Hiển thị ảnh HOG
    Show_HOG()


def clear_images():
    # Xóa cả ảnh gốc và ảnh xám trên cả hai khung hình
    panel1.config(image=None)
    panel2.config(image=None)

    # Cập nhật giao diện
    root.update_idletasks()
    # Cập nhật biến lưu trữ ảnh thành None
    global anh1, anh2
    anh1 = None
    anh2 = None

def resize_image(value):
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Lấy giá trị kích thước mới từ thanh trượt
    new_size = int(value)

    # Thay đổi kích thước ảnh bằng cách sử dụng cv2.resize
    anh2_resized = cv2.resize(anh2, (new_size, new_size))

    # Hiển thị ảnh đã thay đổi kích thước trên khung thứ hai
    show_image(anh2_resized, panel2)
def update_blur_level(value):
    global anh1, anh2
    global kernel_size
    if anh1 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Cập nhật kích thước kernel
    kernel_size = int(value) #value sigma

    # Đảm bảo kernel_size luôn là số lẻ và lớn hơn 0
    kernel_size = max(1, kernel_size)  # Giữ kernel_size lớn hơn hoặc bằng 1
    kernel_size = kernel_size + (1 - kernel_size % 2)  # Làm cho kernel_size trở thành số lẻ

    # Làm mờ ảnh gốc
    blurred_image = cv2.GaussianBlur(anh2, (kernel_size, kernel_size), 0)

    # Hiển thị ảnh đã làm mờ trên khung ảnh thứ hai
    show_image(blurred_image, panel2)

# Khởi tạo giá trị ban đầu cho kernel_size
kernel_size = 15
#xoay ảnh
global count
count = 0
def rotate_image():
    global anh1, anh2
    if anh1 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Xoay ảnh 90 độ theo chiều kim đồng hồ

    global count
    if count == 0:
        anh2 = np.rot90(anh2)
    elif count == 1:
        anh2 = np.rot90(np.rot90(anh2))
    elif count == 2:
        anh2 = np.rot90(np.rot90(np.rot90(anh2)))
    elif count == 3:
        anh2 = np.rot90(np.rot90(np.rot90(np.rot90(anh2))))
        count = 0
        return
    count = count + 1



    # Hiển thị ảnh sau khi xoay trên khung thứ hai
    show_image(anh2, panel2)

def save_image():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh hoặc ảnh không tồn tại.")
        return

    # Chọn nơi lưu trữ và tên tệp
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])

    # Kiểm tra xem người dùng đã chọn nơi lưu trữ và tên tệp chưa
    if file_path:
        # Lưu ảnh xám
        cv2.imwrite(file_path, anh2)
        print("Đã lưu ảnh thành công.")
def Show_CatNguong():
    global anh1, anh2
    global nguong
    nguong = 150
    xam = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    m, n = xam.shape
    xam_uint8 = cv2.convertScaleAbs(xam)
    anh2 = np.zeros([m, n], dtype=np.uint8)
    if xam is None:
        print("Error: Image not loaded properly.")
        return
    for i in range(m):
        for j in range(n):
            if (xam[i, j] < nguong):
                anh2[i, j] = 0
            else:
                anh2[i, j] = 225

    show_image(anh2, panel2)
def update_blur_Gaussian(blur_value):
    global anh1, anh2
    gray_image = gray_image = cv2.cvtColor(anh1, cv2.COLOR_BGR2GRAY)
    image_a = np.array(gray_image)
    blur_value = int(blur_value)
    blurred_image = cv2.GaussianBlur(image_a, (blur_value, blur_value), 0)

    # Hiển thị ảnh sau khi làm mờ trên giao diện
    show_image(blurred_image, panel2)

def update_contrast(value):
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Lấy giá trị độ tương phản từ thanh trượt
    contrast = float(value)

    # Áp dụng độ tương phản cho ảnh
    anh2_contrast = cv2.convertScaleAbs(anh2, alpha=contrast)

    # Hiển thị ảnh đã cập nhật độ tương phản
    show_image(anh2_contrast, panel2)

def convert_to_negative():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Lấy giá trị mức xám lớn nhất trong ảnh
    L = anh2.max()

    # Biến đổi ảnh thành ảnh âm bản
    anh2_negative = L - anh2

    # Hiển thị ảnh âm bản trên giao diện
    show_image(anh2_negative, panel2)


def apply_average_filter():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc trung bình
    filtered_image = cv2.blur(anh2, (5, 5))  # Kích thước kernel là 5x5

    # Hiển thị ảnh đã được lọc trên giao diện
    show_image(filtered_image, panel2)


def apply_weighted_average_filter():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Xây dựng một kernel có trọng số
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # Tổng trọng số bằng 16 để chuẩn hóa

    # Áp dụng bộ lọc trung bình có trọng số
    filtered_image = cv2.filter2D(anh2, -1, kernel)

    # Hiển thị ảnh đã được lọc trên giao diện
    show_image(filtered_image, panel2)


def apply_median_filter():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc trung vị
    filtered_image = cv2.medianBlur(anh2, 5)  # Kích thước kernel là 5x5

    # Hiển thị ảnh đã được lọc trên giao diện
    show_image(filtered_image, panel2)


def apply_min_filter():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc min
    filtered_image = cv2.erode(anh2, None, iterations=1)

    # Hiển thị ảnh đã được lọc trên giao diện
    show_image(filtered_image, panel2)


def apply_max_filter():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc max
    filtered_image = cv2.dilate(anh2, None, iterations=1)

    # Hiển thị ảnh đã được lọc trên giao diện
    show_image(filtered_image, panel2)
def apply_canny():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng thuật toán Canny để phát hiện biên cạnh
    edges = cv2.Canny(anh2, 100, 200)

    # Hiển thị ảnh biên cạnh trên khung thứ hai
    show_image(edges, panel2)

def apply_laplacian():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc Laplacian
    laplacian = cv2.Laplacian(anh2, cv2.CV_64F)

    # Chuyển đổi kết quả thành ảnh 8-bit unsigned integer
    laplacian = np.uint8(np.absolute(laplacian))

    # Hiển thị ảnh đã được xử lý trên khung thứ hai
    show_image(laplacian, panel2)

def apply_sobelX():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc Sobel theo trục X
    sobelX = cv2.Sobel(anh2, cv2.CV_64F, 1, 0, ksize=3)

    # Chuyển đổi kết quả thành ảnh 8-bit unsigned integer
    sobelX = np.uint8(np.absolute(sobelX))

    # Hiển thị ảnh đã được xử lý trên khung thứ hai
    show_image(sobelX, panel2)

def apply_sobelY():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc Sobel theo trục Y
    sobelY = cv2.Sobel(anh2, cv2.CV_64F, 0, 1, ksize=3)

    # Chuyển đổi kết quả thành ảnh 8-bit unsigned integer
    sobelY = np.uint8(np.absolute(sobelY))

    # Hiển thị ảnh đã được xử lý trên khung thứ hai
    show_image(sobelY, panel2)

def apply_sobelCombined():
    global anh2
    if anh2 is None:
        print("Bạn chưa chọn ảnh.")
        return

    # Áp dụng bộ lọc Sobel theo trục X
    sobelX = cv2.Sobel(anh2, cv2.CV_64F, 1, 0, ksize=3)

    # Áp dụng bộ lọc Sobel theo trục Y
    sobelY = cv2.Sobel(anh2, cv2.CV_64F, 0, 1, ksize=3)

    # Kết hợp kết quả từ cả hai bộ lọc Sobel
    sobelCombined = cv2.addWeighted(cv2.convertScaleAbs(sobelX), 0.5, cv2.convertScaleAbs(sobelY), 0.5, 0)

    # Hiển thị ảnh đã được xử lý trên khung thứ hai
    show_image(sobelCombined, panel2)

def apply_prewitt_x():
    global anh2

    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    # Áp dụng prewitt x
    prewitt_x = cv2.filter2D(anh2, -1, kernel_prewitt_x)

    # Hiển thị ảnh
    show_image(prewitt_x, panel2)

def apply_prewitt_y():
    global anh2

    kernel_prewitt_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

    # Áp dụng prewitt y
    prewitt_y = cv2.filter2D(anh2, -1, kernel_prewitt_y)

    # Hiển thị ảnh
    show_image(prewitt_y, panel2)

def apply_prewitt_combine():
    global anh2

    # Kernel Prewitt X
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    # Áp dụng bộ lọc Prewitt X
    prewitt_x = cv2.filter2D(anh2, -1, kernel_prewitt_x)

    # Kernel Prewitt Y
    kernel_prewitt_y = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])

    # Áp dụng bộ lọc Prewitt Y
    prewitt_y = cv2.filter2D(anh2, -1, kernel_prewitt_y)

    # Kết hợp hai đạo hàm gradient từ Prewitt X và Prewitt Y
    prewitt_combine = np.abs(prewitt_x) + np.abs(prewitt_y)

    # Chuẩn hóa lại giá trị pixel
    prewitt_combine = cv2.normalize(prewitt_combine, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Hiển thị ảnh kết quả
    show_image(prewitt_combine, panel2)


root = tk.Tk()
root.title("Phần mềm chỉnh sửa ảnh")

title_label = tk.Label(root, text="Phần mềm chỉnh sửa ảnh", font=("Arial", 20))
title_label.pack()

# Tạo Frame chứa cả hai khung hình
image_frame = tk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

# Tạo khung hình để hiển thị ảnh gốc
panel1 = tk.Label(image_frame, text="Ảnh gốc", font=("Arial", 15), relief="solid")
panel1.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True)

# Tạo khung hình để hiển thị ảnh đã chỉnh sửa
panel2 = tk.Label(image_frame, text="Ảnh đã chỉnh sửa", font=("Arial", 15), relief="solid")
panel2.pack(padx=10, pady=10, side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Tạo một Frame mới để chứa các thanh trượt
slider_frame = tk.Frame(root)
slider_frame.pack(fill=tk.X)

# Tạo thanh trượt Làm mờ Gaussian
blur_Gaussian = tk.Scale(slider_frame, from_=1, to=9, orient=tk.HORIZONTAL, length=200,
                          label="Làm mờ Gaussian", resolution=2, command=update_blur_Gaussian)
blur_Gaussian.grid(row=0, column=0, padx=10, pady=10)

# Tạo thanh trượt Độ mờ
blur_slider = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, length=200,
                        label="Độ mờ", command=update_blur_level)
blur_slider.grid(row=0, column=1, padx=10, pady=10)

# Tạo thanh trượt Kích thước
resize_slider = tk.Scale(slider_frame, from_=50, to=500, orient=tk.HORIZONTAL, length=200,
                          label="Kích thước", command=resize_image)
resize_slider.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

# Tạo thanh trượt Độ tương phản
contrast_slider = tk.Scale(slider_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, length=200,
                            label="Độ tương phản", resolution=0.1, command=update_contrast)
contrast_slider.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

# Tạo khung chứa nút và thanh trượt
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Tạo nút chọn ảnh
button_choose = tk.Button(button_frame, text="Chọn ảnh", command=choose_image)
button_choose.grid(row=0, column=0, padx=10, pady=10)

button_laplacian = tk.Button(button_frame, text="Laplacian", command=apply_laplacian)
button_laplacian.grid(row=0, column=1, padx=10, pady=10)

button_canny = tk.Button(button_frame, text="Canny", command=apply_canny)
button_canny.grid(row=0, column=2, padx=10, pady=10)

button_sobelX = tk.Button(button_frame, text="SobelX", command=apply_sobelX)
button_sobelX.grid(row=1, column=0, padx=10, pady=10)

button_sobelY = tk.Button(button_frame, text="SobelY", command=apply_sobelY)
button_sobelY.grid(row=1, column=1, padx=10, pady=10)

button_sobelCombined = tk.Button(button_frame, text="SobelCombine", command=apply_sobelCombined)
button_sobelCombined.grid(row=1, column=2, padx=10, pady=10)

button_prewittX = tk.Button(button_frame, text="PrewittX", command=apply_prewitt_x)
button_prewittX.grid(row=2, column=0, padx=10, pady=10)

button_prewittY = tk.Button(button_frame, text="PrewittY", command=apply_prewitt_y)
button_prewittY.grid(row=2, column=1, padx=10, pady=10)

button_prewittCombined = tk.Button(button_frame, text="PrewittCombine", command=apply_prewitt_combine)
button_prewittCombined.grid(row=2, column=2, padx=10, pady=10)

button_segmentation = tk.Button(button_frame, text="Phân vùng", command=Show_CatNguong)
button_segmentation.grid(row=3, column=0, padx=10, pady=10)

button_average_filter = tk.Button(button_frame, text="Lọc trung bình", command=apply_average_filter)
button_average_filter.grid(row=3, column=1, padx=10, pady=10)

button_weighted_average_filter = tk.Button(button_frame, text="Lọc trung bình có trọng số", command=apply_weighted_average_filter)
button_weighted_average_filter.grid(row=3, column=2, padx=10, pady=10)

button_median_filter = tk.Button(button_frame, text="Lọc trung vị", command=apply_median_filter)
button_median_filter.grid(row=4, column=0, padx=10, pady=10)

button_min_filter = tk.Button(button_frame, text="Lọc min", command=apply_min_filter)
button_min_filter.grid(row=4, column=1, padx=10, pady=10)

button_max_filter = tk.Button(button_frame, text="Lọc max", command=apply_max_filter)
button_max_filter.grid(row=4, column=2, padx=10, pady=10)

button_convert = tk.Button(button_frame, text="Chuyển đổi thành ảnh âm bản", command=convert_to_negative)
button_convert.grid(row=5, column=0, padx=10, pady=10)

button_hog = tk.Button(button_frame, text="Áp dụng HOG", command=apply_hog)
button_hog.grid(row=5, column=1, padx=10, pady=10)

button_convert_gray = tk.Button(button_frame, text="Đổi màu xám", command=convert_to_gray)
button_convert_gray.grid(row=5, column=2, padx=10, pady=10)



button_rotate = tk.Button(button_frame, text="Xoay ảnh", command=rotate_image)
button_rotate.grid(row=8, column=0, padx=10, pady=10)

button_save = tk.Button(button_frame, text="Lưu ảnh", command=save_image)
button_save.grid(row=8, column=1, padx=10, pady=10)

button_clear = tk.Button(button_frame, text="Xóa ảnh", command=clear_images)
button_clear.grid(row=8, column=2, padx=10, pady=10)
# Biến lưu trữ ảnh gốc và ảnh xám
anh1 = None
anh2 = None

# Hiển thị cửa sổ
root.mainloop()
