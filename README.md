Projek deteksi gambar menggunakan metode yolo dalam bhs pemrograman phython
```python
# 1. Install library YOLO dari ultralytics
!pip install --upgrade ultralytics opencv-python-headless matplotlib --quiet

# 2. Import library
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. Unggah gambar
print("Silakan unggah gambar yang akan digunakan:")
uploaded = files.upload()

# Ambil nama file gambar yang diunggah
image_path = list(uploaded.keys())[0]

# 4. Load model YOLO
model = YOLO("yolov5s.pt")  # YOLOv5s pretrained model

# Kamus label ke bahasa Indonesia
label_dict = {
    'person': 'Orang',
    'bicycle': 'Sepeda',
    'car': 'Mobil',
    'motorbike': 'Motor',
    'airplane': 'Pesawat/Helicopter',
    'helicopter': 'Helicopter',
    'bus': 'Bus',
    'train': 'Kereta',
    'truck': 'Tank',
    'boat': 'Kapal',
    'dog': 'Anjing',
    'cat': 'Kucing',
    'horse': 'Kuda',
    'sheep': 'Domba',
    'cow': 'Sapi',
    'elephant': 'Gajah',
    'bear': 'Beruang',
    'zebra': 'Zebra',
    'giraffe': 'Jerapah'

}

# 5. Fungsi untuk deteksi objek
def detect_objects(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Deteksi objek
    results = model(image_path)
    
    # Annotate gambar dengan label dalam bahasa Indonesia
    for result in results[0].boxes:
        # Dapatkan label dalam bahasa Inggris
        label = model.names[int(result.cls)]
        # Ubah label ke bahasa Indonesia menggunakan kamus
        label_indonesia = label_dict.get(label, label)  # Jika tidak ditemukan, tetap gunakan label asli
        
        # Dapatkan koordinat dan confidence
        coordinates = result.xyxy.cpu().numpy()[0]  # Koordinat bounding box
        confidence = result.conf.item()  # Confidence score
        
        # Gambar bounding box dan label di atas gambar
        cv2.rectangle(img, (int(coordinates[0]), int(coordinates[1])), 
                      (int(coordinates[2]), int(coordinates[3])), (255, 0, 0), 2)  # Menggambar kotak
        cv2.putText(img, f"{label_indonesia} ({confidence:.2f})", 
                    (int(coordinates[0]), int(coordinates[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Menambahkan teks label

    # Plot hasil deteksi
    annotated_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()

    # Print hasil deteksi ke terminal
    print("Hasil Deteksi Objek:")
    for result in results[0].boxes:
        # Dapatkan label dalam bahasa Inggris
        label = model.names[int(result.cls)]
        # Ubah label ke bahasa Indonesia
        label_indonesia = label_dict.get(label, label)
        confidence = result.conf.item()  # Confidence score
        coordinates = result.xyxy.cpu().numpy()  # Koordinat bounding box
        print(f"Label: {label_indonesia}, Kepercayaan: {confidence:.2f}, Koordinat: {coordinates}")

# 6. Jalankan deteksi objek
detect_objects(image_path)
```
# Contoh Gambar
![Pesawat](https://github.com/user-attachments/assets/9e891c2d-5bfb-4021-9278-a589308d64a4)
![Tank](https://github.com/user-attachments/assets/a1bab4a2-4266-47ae-9bdd-f26bb8104006)
![OIP](https://github.com/user-attachments/assets/df39c096-2b43-4108-b100-2e31d394d881)
![Kapal](https://github.com/user-attachments/assets/526bafba-dd81-4063-b8e1-0d1ced1ed536)
# Hasil Gambar 
![Screenshot 2024-12-18 113240](https://github.com/user-attachments/assets/bc47c65b-68b9-412c-b4c9-efc94778b946)
![image](https://github.com/user-attachments/assets/a7b54f7c-c435-4e5e-a77a-c785f3063ccf)
![image](https://github.com/user-attachments/assets/7e34dccf-d954-464d-b3fa-604b6ed5937e)
![image](https://github.com/user-attachments/assets/939b5948-cbb1-4808-a129-c7edf69ac0b4)
