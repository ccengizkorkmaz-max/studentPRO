import cv2
from ultralytics import YOLO
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import os

# Premium UI Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StudentMonPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("StudentMon Pro | Akıllı Sınıf & Duygu Analizi (YOLOv11)")
        self.geometry("1200x800")
        
        # Load YOLOv11 Model (Objects & People)
        self.model = YOLO("yolo11n.pt")
        
        # Load OpenCV Cascades for Face & Smile (Mood)
        self.face_cascade = cv2.CascadeClassifier('face.xml')
        self.smile_cascade = cv2.CascadeClassifier('smile.xml')
        
        # Application State
        self.running = False
        self.cap = None
        self.total_count = 0
        self.current_mood = "Analiz Ediliyor..."
        self.conf_threshold = 0.5
        
        self.setup_ui()
        
    def setup_ui(self):
        # Grid Configuration
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Main Feed Area
        self.feed_container = ctk.CTkFrame(self, corner_radius=15, fg_color="#0a0a0a")
        self.feed_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.feed_container, text="Kamera Hazırlanıyor...", font=("Inter", 18))
        self.video_label.pack(expand=True, fill="both")

        # 2. Sidebar Controls & Stats
        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=350)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        
        # Logo & Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🔍 StudentMon PRO", font=("Outfit", 24, "bold"))
        self.logo_label.pack(pady=30, padx=20)

        # Stats Card
        self.stats_card = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a", corner_radius=12)
        self.stats_card.pack(pady=10, padx=20, fill="x")
        
        # Total Count
        self.count_label_title = ctk.CTkLabel(self.stats_card, text="GÖRÜNTÜDEKİ NESNE SAYISI", font=("Inter", 10, "bold"), text_color="#aaaaaa")
        self.count_label_title.pack(pady=(15, 0))
        self.count_val_frame = ctk.CTkFrame(self.stats_card, fg_color="transparent")
        self.count_val_frame.pack()
        
        self.count_val = ctk.CTkLabel(self.count_val_frame, text="0", font=("Inter", 48, "bold"), text_color="#00e5ff")
        self.count_val.pack(side="left")
        
        self.count_suffix = ctk.CTkLabel(self.count_val_frame, text=" tane obje var", font=("Inter", 14, "bold"), text_color="#00e5ff")
        self.count_suffix.pack(side="left", padx=(5, 0), pady=(15, 0))

        # Mood Card
        self.mood_card = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a", corner_radius=12)
        self.mood_card.pack(pady=10, padx=20, fill="x")
        
        self.mood_title = ctk.CTkLabel(self.mood_card, text="MOD / DUYGU DURUMU", font=("Inter", 10, "bold"), text_color="#aaaaaa")
        self.mood_title.pack(pady=(15, 0))
        
        self.mood_val = ctk.CTkLabel(self.mood_card, text="Bekleniyor", font=("Inter", 20, "bold"), text_color="#ffeb3b")
        self.mood_val.pack(pady=(0, 15))
        
        self.mood_progress = ctk.CTkProgressBar(self.mood_card, height=10, width=200)
        self.mood_progress.set(0)
        self.mood_progress.pack(pady=(0, 20))

        # Activity Card (New)
        self.activity_card = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a", corner_radius=12)
        self.activity_card.pack(pady=10, padx=20, fill="x")
        
        self.activity_title = ctk.CTkLabel(self.activity_card, text="AKTİVİTE / DURUM", font=("Inter", 10, "bold"), text_color="#aaaaaa")
        self.activity_title.pack(pady=(15, 0))
        
        self.activity_val = ctk.CTkLabel(self.activity_card, text="Ders Dinleniyor", font=("Inter", 14, "bold"), text_color="#00c853")
        self.activity_val.pack(pady=(0, 15))

        # Status Badge
        self.status_badge = ctk.CTkLabel(self.sidebar, text="● SİSTEM DURDURULDU", font=("Inter", 10, "bold"), text_color="#ff5252")
        self.status_badge.pack(pady=10)

        # Slider Control
        self.conf_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.conf_frame.pack(pady=20, padx=20, fill="x")
        
        self.slider_label = ctk.CTkLabel(self.conf_frame, text=f"Hassasiyet: {int(self.conf_threshold*100)}%", font=("Inter", 12))
        self.slider_label.pack()
        
        self.slider = ctk.CTkSlider(self.conf_frame, from_=0.1, to=1.0, number_of_steps=18, command=self.update_conf)
        self.slider.set(self.conf_threshold)
        self.slider.pack(pady=10)

        # Action Buttons
        self.start_btn = ctk.CTkButton(self.sidebar, text="Kamerayı Başlat", height=55, fg_color="#7c4dff", hover_color="#5e35b1", font=("Inter", 15, "bold"), command=self.toggle_camera)
        self.start_btn.pack(pady=(40, 10), padx=20, fill="x")
        
        self.info_text = ctk.CTkLabel(self.sidebar, text="YOLOv11 Pro Engine + MoodX\nWindows Native v1.1", font=("Inter", 10), text_color="#555555")
        self.info_text.pack(side="bottom", pady=20)

    def update_conf(self, value):
        self.conf_threshold = value
        self.slider_label.configure(text=f"Hassasiyet: {int(value*100)}%")

    def toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.video_label.configure(text="HATA: Kamera Açılamadı!")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            self.start_btn.configure(text="Kamerayı Durdur", fg_color="#ff5252")
            self.status_badge.configure(text="● SİSTEM AKTİF", text_color="#00c853")
            
            # Start Processing Thread
            self.thread = threading.Thread(target=self.video_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.start_btn.configure(text="Kamerayı Başlat", fg_color="#7c4dff")
            self.status_badge.configure(text="● SİSTEM DURDURULDU", text_color="#ff5252")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image="", text="Kamera Hazırlanıyor...")
            self.count_val.configure(text="0")
            self.mood_val.configure(text="Bekleniyor")
            self.mood_progress.set(0)

    def analyze_emotion(self, frame):
        # Mood recognition logic - Hyper-tuned for the user's smile
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces (Frontal faces only for focus detection)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        mood = "Odaklanmış / Nötr 😐"
        score = 0.4
        color = "#00e5ff"
        face_found = len(faces) > 0
        
        for (x, y, w, h) in faces:
            # SMILE ROI: Focus only on the bottom half of the face
            smile_roi_gray = gray[y + int(h/1.8):y + h, x:x + w]
            smile_roi_color = frame[y + int(h/1.8):y + h, x:x + w]
            
            # HYPER-SENSITIVE TUNING for smile
            smiles = self.smile_cascade.detectMultiScale(smile_roi_gray, 1.1, 10, minSize=(15, 15))
            
            if len(smiles) > 0:
                mood = "Mutlu / Gulumseyen 😊"
                score = 0.95
                color = "#00c853"
            
        return mood, score, color, face_found

    def video_loop(self):
        time.sleep(1)
        while self.running:
            if self.cap is None: break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # 1. Object Detection (YOLO) - Aggressive NMS for zoom stability
            # iou=0.2 makes it very strict about overlapping boxes (merges them)
            # agnostic_nms=True ensures different scales are treated the same
            results = self.model.predict(frame, conf=self.conf_threshold, iou=0.2, classes=[0, 67], agnostic_nms=True, verbose=False)
            
            # --- Anti-Spam Filter ---
            # If camera zoom causes multiple "person" boxes, let's only keep the most confident ones
            detections = results[0].boxes
            has_phone = False
            has_person = False
            
            # Filter to keep only 1 best person if multiple appear due to zoom jitter
            person_boxes = [b for b in detections if int(b.cls[0]) == 0]
            phone_boxes = [b for b in detections if int(b.cls[0]) == 67]
            
            if person_boxes:
                has_person = True
                # We can keep all or just the top 1. Let's keep all but YOLO will now merge them better.
            if phone_boxes:
                has_phone = True

            annotated_frame = results[0].plot(line_width=2, font_size=10)
            self.total_count = len(results[0].boxes)
            
            # Logic uses the flags we set above
            
            # 2. Emotion Analysis & Focus Check
            mood_text, mood_score, mood_color, face_visible = self.analyze_emotion(annotated_frame)
            
            # --- Advanced Activity Logic ---
            if has_phone:
                activity_text = "⚠️ TELEFONLA OYNUYOR!"
                activity_color = "#ff5252"
            elif has_person and not face_visible:
                activity_text = "⚠️ ODAK KAYBOLDU! (Bakmıyor)"
                activity_color = "#ff9800"
                mood_text = "Başka Yere Bakıyor"
                mood_score = 0.2
                mood_color = "#ff9800"
            elif has_person and face_visible:
                activity_text = "Ders Dinleniyor / Odaklı"
                activity_color = "#00c853"
            elif not has_person:
                activity_text = "Sınıf Boş / Bekleniyor"
                activity_color = "#aaaaaa"
                mood_text = "Analiz Ediliyor..."
                mood_score = 0
            else:
                activity_text = "Gözlemleniyor..."
                activity_color = "#00e5ff"

            # UI Update
            try:
                cv2_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                
                width = self.feed_container.winfo_width()
                height = self.feed_container.winfo_height()
                if width > 10 and height > 10:
                    img.thumbnail((width, height))
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                self.after(0, self.update_ui_elements, ctk_img, self.total_count, mood_text, mood_score, mood_color, activity_text, activity_color)
            except Exception as e:
                print(f"UI Error: {e}")
            
            time.sleep(0.01)

    def update_ui_elements(self, image, count, mood, score, color, activity, activity_color):
        self.video_label.configure(image=image, text="")
        self.count_val.configure(text=str(count))
        self.mood_val.configure(text=mood, text_color=color)
        self.mood_progress.set(score)
        self.mood_progress.configure(progress_color=color)
        self.activity_val.configure(text=activity, text_color=activity_color)

if __name__ == "__main__":
    app = StudentMonPro()
    app.mainloop()
