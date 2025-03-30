import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ TensorFlow 成功偵測到 GPU 裝置：")
        for gpu in gpus:
            print(f" - {gpu}")
    else:
        print("❌ TensorFlow 沒有偵測到 GPU，請確認 CUDA / cuDNN 是否安裝正確。")

if __name__ == "__main__":
    check_gpu()
