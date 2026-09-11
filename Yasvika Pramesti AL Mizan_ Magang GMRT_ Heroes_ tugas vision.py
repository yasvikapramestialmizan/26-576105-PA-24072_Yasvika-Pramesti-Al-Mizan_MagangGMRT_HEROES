import cv2 
import numpy as np

def detect_aruco(): #Mendefinisikan fungsi untuk mendeteksi marker ArUco
    aruco_dict_type = cv2.aruco.DICT_6X6_250 # Mendefinisikan tipe dictionary ArUco yang akan digunakan (6x6 dengan 250 marker)
    if hasattr(cv2.aruco, 'ArucoDetector'): # Mengecek apakah versi OpenCV yang digunakan mendukung ArUcoDetector (versi 4.7.0 ke atas)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type) # Mendapatkan dictionary ArUco yang telah ditentukan
        aruco_params = cv2.aruco.DetectorParameters() # Membuat parameter deteksi ArUco
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) # Membuat objek detektor ArUco dengan dictionary dan parameter yang telah ditentukan
        use_new_detector = True # Menandai bahwa detektor baru akan digunakan
    else: 
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type) # Mendapatkan dictionary ArUco yang telah ditentukan (untuk versi OpenCV sebelum 4.7.0)
        aruco_params = cv2.aruco.DetectorParameters_create() # Membuat parameter deteksi ArUco (untuk versi OpenCV sebelum 4.7.0)
        detector = None # Tidak menggunakan detektor baru, karena versi OpenCV yang digunakan tidak mendukung ArUcoDetector
        use_new_detector = False # Menandai bahwa detektor lama akan digunakan
    cap = cv2.VideoCapture(0) # Membuka kamera default (indeks 0) untuk menangkap video

    if not cap.isOpened(): # Mengecek apakah kamera berhasil dibuka
        print("Error: Could not open video capture.") # Jika kamera tidak berhasil dibuka, mencetak pesan error dan mengembalikan fungsi
        return 
    print("=== ArUco Marker Detection Started ===") # Mencetak pesan bahwa deteksi marker ArUco telah dimulai
    print("Press 'q' to quit.") # Mencetak pesan bahwa pengguna dapat menekan tombol 'q' untuk keluar dari program

    while True:
        ret, frame = cap.read() # Membaca frame dari kamera. 'ret' adalah boolean yang menunjukkan apakah frame berhasil dibaca, dan 'frame' adalah gambar yang dibaca dari kamera.
        if not ret: # Jika frame tidak berhasil dibaca, mencetak pesan error dan keluar dari loop
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Mengubah frame menjadi grayscale untuk mempermudah deteksi marker ArUco

        if use_new_detector: 
            corners, ids, rejected = detector.detectMarkers(gray) # Jika menggunakan detektor baru, mendeteksi marker ArUco pada frame grayscale menggunakan metode detectMarkers dari objek detektor. Metode ini mengembalikan tiga nilai: 'corners' (koordinat sudut marker yang terdeteksi), 'ids' (ID marker yang terdeteksi), dan 'rejected' (marker yang ditolak).
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params) # Jika menggunakan detektor lama, mendeteksi marker ArUco pada frame grayscale menggunakan metode detectMarkers dari modul cv2.aruco. Metode ini juga mengembalikan tiga nilai: 'corners', 'ids', dan 'rejected'.

        if ids is not None: # Jika ada marker ArUco yang terdeteksi (yaitu, jika 'ids' tidak kosong), menggambar marker yang terdeteksi pada frame menggunakan fungsi drawDetectedMarkers dari modul cv2.aruco, dan mencetak ID marker yang terdeteksi ke konsol.
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected ArUco markers with IDs: {ids.flatten()}")

        cv2.imshow('ArUco Marker Detection', frame) # Menampilkan frame dengan marker ArUco yang terdeteksi dalam jendela bernama 'ArUco Marker Detection'

        if cv2.waitKey(1) & 0xFF == ord('q'): # Jika pengguna menekan tombol 'q', keluar dari loop dan menghentikan deteksi marker ArUco
            break

cap.release() # Melepaskan kamera setelah selesai digunakan
cv2.destroyAllWindows() # Menutup semua jendela OpenCV yang telah dibuka
if __name__ == "__main__": # Mengecek apakah skrip ini dijalankan sebagai program utama (bukan diimpor sebagai modul). Jika ya, memanggil fungsi
detect_aruco() # Memanggil fungsi detect_aruco() untuk memulai deteksi marker ArUco