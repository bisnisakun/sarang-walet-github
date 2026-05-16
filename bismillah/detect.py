import cv2
from ultralytics import YOLO
import time
import numpy as np
import math
import os
import requests

# =========================
# LOAD MODEL
# =========================
model = YOLO("majukena.pt")

# =========================
# KALIBRASI PIXEL -> CM
# =========================
PIXEL_TO_CM = 0.01

# =========================
# FOLDER INPUT & OUTPUT
# =========================
folder_input = "folder_gambar"
folder_output = "hasil"

os.makedirs(
    folder_output,
    exist_ok=True
)

# =========================
# URL FLASK SERVER
# =========================
UPLOAD_URL = "http://127.0.0.1:5000/upload"

# =========================
# LOOP SEMUA GAMBAR
# =========================
for filename in os.listdir(folder_input):

    if filename.endswith((
        ".jpg",
        ".jpeg",
        ".png"
    )):

        image_path = os.path.join(
            folder_input,
            filename
        )

        print(f"\nMemproses: {filename}")

        # =========================
        # BACA GAMBAR
        # =========================
        frame = cv2.imread(
            image_path
        )

        # =========================
        # CEK GAMBAR
        # =========================
        if frame is None:

            print(
                f"Gagal membaca {filename}"
            )

            continue

        # =========================
        # UKURAN GAMBAR
        # =========================
        height, width = frame.shape[:2]

        # =========================
        # TIMER
        # =========================
        start_time = time.time()

        # =========================
        # PREDIKSI YOLO
        # =========================
        results = model.predict(
            frame,
            conf=0.35
        )

        segmented_frame = frame.copy()

        result = results[0]

        # =========================
        # CEK MASK
        # =========================
        if result.masks is not None:

            masks = result.masks.data.cpu().numpy()

            hsv_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV
            )

            # =========================
            # LOOP OBJECT
            # =========================
            for i, mask in enumerate(masks):

                # Resize mask
                mask = cv2.resize(
                    mask,
                    (width, height)
                )

                # Binary mask
                mask = (
                    mask > 0.5
                ).astype(np.uint8)

                # =========================
                # WARNA SEGMENTASI
                # =========================
                color = np.zeros_like(
                    frame
                )

                color[:] = (
                    0,
                    255,
                    0
                )

                segmented_frame = np.where(
                    mask[:, :, np.newaxis] == 1,
                    cv2.addWeighted(
                        segmented_frame,
                        0.5,
                        color,
                        0.5,
                        0
                    ),
                    segmented_frame
                )

                # =========================
                # HITUNG LUAS
                # =========================
                area_pixel = cv2.countNonZero(
                    mask
                )

                area_cm2 = (
                    area_pixel
                    * (
                        PIXEL_TO_CM ** 2
                    )
                )

                # =========================
                # CONTOUR
                # =========================
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                circularity_percent = 0
                cx, cy = 0, 0

                if len(contours) > 0:

                    cnt = max(
                        contours,
                        key=cv2.contourArea
                    )

                    perimeter = cv2.arcLength(
                        cnt,
                        True
                    )

                    contour_area = cv2.contourArea(
                        cnt
                    )

                    # =========================
                    # CIRCULARITY
                    # =========================
                    if perimeter > 0:

                        circularity = (
                            4
                            * math.pi
                            * contour_area
                        ) / (
                            perimeter ** 2
                        )

                        circularity_percent = (
                            circularity * 100
                        )

                    # =========================
                    # TITIK TENGAH
                    # =========================
                    M = cv2.moments(cnt)

                    if M["m00"] != 0:

                        cx = int(
                            M["m10"]
                            / M["m00"]
                        )

                        cy = int(
                            M["m01"]
                            / M["m00"]
                        )

                # =========================
                # HSV
                # =========================
                hsv_pixels = hsv_frame[
                    mask == 1
                ]

                hsv_text = "Unknown"

                mean_h = 0
                mean_s = 0
                mean_v = 0

                if len(hsv_pixels) > 0:

                    mean_h = np.mean(
                        hsv_pixels[:, 0]
                    )

                    mean_s = np.mean(
                        hsv_pixels[:, 1]
                    )

                    mean_v = np.mean(
                        hsv_pixels[:, 2]
                    )

                    white_percent = (
                        mean_v / 255
                    ) * 100

                    # =========================
                    # KLASIFIKASI WARNA
                    # =========================
                    if (
                        mean_s < 25
                        and mean_v > 200
                    ):

                        hsv_text = (
                            f"Sangat Putih "
                            f"({white_percent:.1f}%)"
                        )

                    elif (
                        mean_s < 45
                        and mean_v > 170
                    ):

                        hsv_text = (
                            f"Putih "
                            f"({white_percent:.1f}%)"
                        )

                    elif (
                        mean_s < 70
                        and mean_v > 130
                    ):

                        hsv_text = (
                            f"Agak Putih "
                            f"({white_percent:.1f}%)"
                        )

                    else:

                        hsv_text = (
                            f"Gelap "
                            f"({white_percent:.1f}%)"
                        )

                # =========================
                # NOMOR SARANG
                # =========================
                cv2.putText(
                    segmented_frame,
                    str(i + 1),
                    (
                        cx - 10,
                        cy - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

                # =========================
                # TITIK TENGAH
                # =========================
                cv2.circle(
                    segmented_frame,
                    (cx, cy),
                    8,
                    (0, 0, 255),
                    -1
                )

                # =========================
                # POSISI TEXT
                # =========================
                text_y = (
                    80
                    + (i * 160)
                )

                # =========================
                # TEXT INFO
                # =========================
                cv2.putText(
                    segmented_frame,
                    f"Sarang {i+1}",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    segmented_frame,
                    f"Luas : {area_cm2:.2f} cm2",
                    (
                        20,
                        text_y + 35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    segmented_frame,
                    f"Circularity : {circularity_percent:.2f}%",
                    (
                        20,
                        text_y + 70
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    segmented_frame,
                    f"Warna : {hsv_text}",
                    (
                        20,
                        text_y + 105
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    segmented_frame,
                    f"H:{mean_h:.1f} S:{mean_s:.1f} V:{mean_v:.1f}",
                    (
                        20,
                        text_y + 140
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

        # =========================
        # PROCESS TIME
        # =========================
        process_time = (
            time.time()
            - start_time
        )

        cv2.putText(
            segmented_frame,
            f"Time : {process_time:.3f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        # =========================
        # SIMPAN HASIL
        # =========================
        output_filename = f"hasil_{filename}"

        output_path = os.path.join(
            folder_output,
            output_filename
        )

        cv2.imwrite(
            output_path,
            segmented_frame
        )

        print(
            f"Hasil disimpan: {output_path}"
        )

        # =========================
        # AUTO UPLOAD KE FLASK
        # =========================
        try:

            files = {
                "image": open(
                    output_path,
                    "rb"
                )
            }

            response = requests.post(
                UPLOAD_URL,
                files=files
            )

            print(
                "Upload berhasil:",
                response.text
            )

        except Exception as e:

            print(
                "Gagal upload:",
                e
            )

print("\nSemua gambar selesai diproses.")