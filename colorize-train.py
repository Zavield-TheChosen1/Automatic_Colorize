import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import image_dataset_from_directory

# Đường dẫn đến dữ liệu huấn luyện và kiểm tra
train_data_dir = 'E:\\Code\\Machine Learning\\picture\\huanluyen\\dentrang'
train_labels_dir = 'E:\\Code\\Machine Learning\\picture\\huanluyen\\mau'
test_data_dir = 'E:\\Code\\Machine Learning\\picture\\kiemtra'

# Tạo dataset cho dữ liệu huấn luyện và kiểm tra
image_size = (350, 350)
batch_size = 16
epochs = 10

train_dataset = image_dataset_from_directory(
    train_data_dir,
    labels='inferred',
    label_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=123,
)

train_labels_dataset = image_dataset_from_directory(
    train_labels_dir,
    labels='inferred',
    label_mode=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=123,
)

test_dataset = image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=123,
)

def tomau_anh_model():
    model = Sequential()
    # Lớp Conv2D đầu để trích xuất đặc trưng từ dữ liệu đầu vào
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(350, 350, 1)))
    
    # Thêm các lớp Conv2D khác
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    
    # Tạo lớp Flatten dùng để biến đổi dữ liệu thành 1D
    model.add(Flatten())
    
    # Tạo lớp Dense kết nối neuron với các đặc trưng trích xuất từ trước và ánh xạ biểu diễn màu theo mỗi điểm ảnh
    model.add(Dense(350 * 350 * 2, activation='sigmoid'))
    
    # Tạo lớp Reshape để định dạng lại dữ liệu
    model.add(Reshape((350, 350, 2)))
    
    return model

# Định nghĩa hàm mất mát dùng MSE
def tomau_loss(y_true, y_pred):
    mse = MeanSquaredError()
    loss = mse(y_true, y_pred)
    return loss

# Tạo mô hình
model = tomau_anh_model()

# Biên dịch mô hình
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=tomau_loss)

# In thông báo bắt đầu huấn luyện
print("Bắt đầu việc huấn luyện...")

# Quá trình huấn luyện mô hình (sử dụng dữ liệu train dataset và train labels dataset)
model.fit(train_dataset, train_labels_dataset, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Lưu mô hình sau khi huấn luyện
model.save("tomautudong.h5")

# In ra thông báo kết thúc huấn luyện
print("Hoàn thành việc huấn luyện")

# Đánh giá mô hình trên tập kiểm tra
test_loss = model.evaluate(test_dataset)

# In ra giá trị mất mát trên tập kiểm tra
print("Mất mát trên tập kiểm tra:", test_loss)
