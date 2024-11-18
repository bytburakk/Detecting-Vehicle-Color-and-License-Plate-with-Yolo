import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Eğitilmiş YOLO 
model = YOLO("best.pt") 


image_path = 'car8.png'  # Buraya analiz etmek istediğimiz fotoğraf
frame = cv2.imread(image_path)

# Renk Algılama Fonksiyonu
def detect_color_hsv(hsv_color):
    h, s, v = hsv_color

    
    if (0 <= h <= 10 or 160 <= h <= 180) and s > 100 and v > 50:
        return "Kirmizi"
    
    elif 35 <= h <= 85 and s > 100 and v > 50:
        return "Yeşil"
    
    elif 100 <= h <= 140 and s > 100 and v > 50:
        return "Mavi"

    elif 20 <= h <= 30 and s > 100 and v > 50:
        return "Sari"

    elif s < 50 and v > 200:
        return "Beyaz"
    
    elif v < 50:
        return "Siyah"
    
    elif s < 50 and 50 < v < 200:
        return "Gri"
    
    else:
        return "Bilinmeyen"


def process_frame(frame):
    results = model(frame)  

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            label = int(box.cls)  
            conf = box.conf  

            
            if label == 0:  
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2 - 5 
                
                if 0 <= center_y < frame.shape[0] and 0 <= center_x < frame.shape[1]: 
                    bgr_color = frame[center_y, center_x]  
                    hsv_color = cv2.cvtColor(bgr_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0][0]  
                    detected_color = detect_color_hsv(hsv_color)
                    
                    print(f"Koordinat ({center_x}, {center_y}): HSV = {hsv_color}, Renk = {detected_color}")
                else:
                    detected_color = "Bilinmeyen"
                    print(f"Koordinat sınır dışında: ({center_x}, {center_y})")

                # Bounding Box çizimi (Araç)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Arac, Renk: {detected_color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

processed_frame = process_frame(frame)

if image_path is None:
    print("Görüntü yüklenemedi, yolun doğru olduğundan emin olun.")
else:
    print("Görüntü yüklendi, ön işleme geçiliyor...")

# Görüntüyü iyileştirme
enhanced_img = cv2.bilateralFilter(frame, 9, 75, 75)  # Gürültüyü azaltma
enhanced_img_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)  # Gri tonlama
enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)  # RGB formatı

print("Model çalıştırılıyor...")
results = model(enhanced_img_rgb, conf=0.25)  # Tespit eşiği


if results[0].boxes.xyxy.shape[0] == 0: 
    print("Plaka tespit edilemedi. Algılanabilir plaka yok.")
else:
    print("Plaka tespit edildi. İşleme başlıyor...")

    for box in results[0].boxes:  
        box_data = box.xyxy.cpu().numpy().flatten()  

        if len(box_data) >= 4: 
            x1, y1, x2, y2 = map(int, box_data[:4]) 

            print(f"Algılanan plaka koordinatları: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            
            cropped_plate = frame[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                print("Kırpılan plaka bölgesi boş, devam edilemiyor.")
                continue

            # OCR için plakayı iyileştir
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)  # Gri tonlama
            enhanced_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)  # Gürültü azaltma

            # OCR ile plaka metnini tespit et
            try:
                plate_text = pytesseract.image_to_string(
                    Image.fromarray(enhanced_plate), config="--psm 8"
                )
                plate_text = plate_text.strip()  

                
                if len(plate_text) > 5:
                    print(f"Tespit edilen plaka: {plate_text}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                    cv2.putText(
                        frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                else:
                    print("Tespit edilen plaka 5 karakterden kısa, gösterilmiyor.")
            except Exception as e:
                print(f"OCR sırasında hata oluştu: {e}")

        else:
            print("Bounding box formatı beklenen değerlerden eksik.")

cv2.imshow("Arac ve Renk Tespiti", processed_frame)
cv2.waitKey(0)  

# Sonucu Kaydetme :
cv2.imwrite("processed_image.jpg", processed_frame)

cv2.destroyAllWindows()
