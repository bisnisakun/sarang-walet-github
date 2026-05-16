import cv2
from ultralytics import YOLO
import time
import numpy as np
import math

# =========================
# LOAD MODEL
# =========================
model = YOLO("majukena.pt")

# =========================
# KALIBRASI PIXEL -> CM
# =========================
PIXEL_TO_CM = 0.01

# =========================
# BUKA KAMERA
# =========================
cam = cv2.imread("z4202334869918_b827a8c6ec2969f0bcfa0b9fbb7ab0f9_jpg.rf.c1d78097daea835a7c9ffb80327016af.jpg")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, )
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, )

while True:

    # =========================
    # AMBIL FRAME
    # =========================
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame")
        break

    start_time = time.time()

    # =========================
    # PREDIKSI YOLO
    # =========================
    results = model.predict(frame, conf=0.5)

    segmented_frame = frame.copy()

    result = results[0]

    # =========================
    # CEK ADA MASK ATAU TIDAK
    # =========================
    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        # Konversi HSV sekali
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # =========================
        # LOOP SETIAP SARANG
        # =========================
        for i, mask in enumerate(masks):

            # Resize mask
            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0])
            )

            # Binary mask
            mask = (mask > 0.5).astype(np.uint8)

            # =========================
            # WARNA SEGMENTASI
            # =========================
            color = np.zeros_like(frame)
            color[:] = (0, 255, 0)

            segmented_frame = np.where(
                mask[:, :, np.newaxis] == 1,
                cv2.addWeighted(segmented_frame, 0.5, color, 0.5, 0),
                segmented_frame
            )

            # =========================
            # HITUNG LUAS
            # =========================
            area_pixel = cv2.countNonZero(mask)

            area_cm2 = area_pixel * (PIXEL_TO_CM ** 2)

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

                cnt = max(contours, key=cv2.contourArea)

                perimeter = cv2.arcLength(cnt, True)

                contour_area = cv2.contourArea(cnt)

                # =========================
                # CIRCULARITY
                # =========================
                if perimeter > 0:

                    circularity = (
                        4 * math.pi * contour_area
                    ) / (perimeter ** 2)

                    circularity_percent = circularity * 100

                # =========================
                # TITIK TENGAH OBJECT
                # =========================
                M = cv2.moments(cnt)

                if M["m00"] != 0:

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

            # =========================
            # HSV ANALYSIS
            # =========================
            hsv_pixels = hsv_frame[mask == 1]

            hsv_text = "Unknown"

            mean_h = 0
            mean_s = 0
            mean_v = 0

            if len(hsv_pixels) > 0:

                mean_h = np.mean(hsv_pixels[:, 0])
                mean_s = np.mean(hsv_pixels[:, 1])
                mean_v = np.mean(hsv_pixels[:, 2])

                # =========================
                # PERSENTASE PUTIH
                # =========================
                white_percent = (mean_v / 255) * 100

                # =========================
                # KLASIFIKASI PUTIH
                # =========================
                if mean_s < 25 and mean_v > 200:
                    hsv_text = f"Sangat Putih ({white_percent:.1f}%)"

                elif mean_s < 45 and mean_v > 170:
                    hsv_text = f"Putih ({white_percent:.1f}%)"

                elif mean_s < 70 and mean_v > 130:
                    hsv_text = f"Agak Putih ({white_percent:.1f}%)"

                else:
                    hsv_text = f"Gelap ({white_percent:.1f}%)"

            # =========================
            # POSISI TEXT
            # =========================
            text_y = 30 + (i * 150)

            # =========================
            # TAMPILKAN DATA
            # =========================
            cv2.putText(
                segmented_frame,
                f"Sarang {i+1}",
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                segmented_frame,
                f"Luas : {area_cm2:.2f} cm2",
                (20, text_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                segmented_frame,
                f"Circularity : {circularity_percent:.2f} %",
                (20, text_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                segmented_frame,
                f"Tingkat Putih : {hsv_text}",
                (20, text_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                segmented_frame,
                f"H:{mean_h:.1f} S:{mean_s:.1f} V:{mean_v:.1f}",
                (20, text_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # =========================
            # TITIK TENGAH OBJECT
            # =========================
            cv2.circle(
                segmented_frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                segmented_frame,
                f"{i+1}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # =========================
    # PROCESS TIME
    # =========================
    process_time = time.time() - start_time

    cv2.putText(
        segmented_frame,
        f"Time : {process_time:.3f} s",
        (20, 700),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # =========================
    # TAMPILKAN FRAME
    # =========================
    cv2.imshow(
        "YOLO Multi Sarang HSV Analysis",
        segmented_frame
    )

    # =========================
    # EXIT
    # =========================
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =========================
# RELEASE
# =========================
cam.release()
cv2.destroyAllWindows()