# pip install pyqt5 ultralytics opencv-python
import sys, os, glob
from PyQt5 import QtWidgets, QtCore
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes  # [web:86]
import cv2
import numpy as np

import sys, os
def data_path(rel):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.dirname(__file__), rel)

FACE_WEIGHTS  = data_path(os.path.join("models", "yolov11n-face.pt"))
PLATE_WEIGHTS = data_path(os.path.join("models", "yolov11x-license-plate.pt"))


TILE    = 1536
OVERLAP = 0.25
STRIDE  = int(TILE * (1 - OVERLAP))
CONF    = 0.05
IOU     = 0.5

def blur_box(im, b, w0, h0, k=51):
    x1, y1, x2, y2 = map(int, b)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w0, x2); y2 = min(h0, y2)
    roi = im[y1:y2, x1:x2]
    if roi.size:
        kk = max(3, k | 1)
        kk = min(kk, max(3, (y2 - y1)//2*2 - 1))
        kk = min(kk, max(3, (x2 - x1)//2*2 - 1))
        if kk < 3 or kk % 2 == 0: kk = 3
        im[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kk, kk), 0)

class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)  # percent, message
    finished = QtCore.pyqtSignal()

    def __init__(self, in_dir, out_dir, do_faces=True, do_plates=True, parent=None):
        super().__init__(parent)
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.do_faces = do_faces
        self.do_plates = do_plates
        self.stop_flag = False

        # Load models once
        self.face_model  = YOLO(FACE_WEIGHTS) if do_faces else None
        self.plate_model = YOLO(PLATE_WEIGHTS) if do_plates else None

    def run(self):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
        files = [p for e in exts for p in glob.glob(os.path.join(self.in_dir, e))]
        total = len(files)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        for idx, in_path in enumerate(files, start=1):
            if self.stop_flag:
                break
            img0 = cv2.imread(in_path)
            if img0 is None:
                self.progress.emit(int(idx/total*100), f"Skip (read error): {os.path.basename(in_path)}")
                continue
            h0, w0 = img0.shape[:2]

            xs = list(range(0, max(1, w0 - 1), STRIDE))
            ys = list(range(0, max(1, h0 - 1), STRIDE))
            if xs[-1] + TILE < w0: xs.append(max(0, w0 - TILE))
            if ys[-1] + TILE < h0: ys.append(max(0, h0 - TILE))

            for y in ys:
                for x in xs:
                    x2 = min(w0, x + TILE)
                    y2 = min(h0, y + TILE)
                    patch = img0[y:y2, x:x2]
                    if patch.size == 0:
                        continue

                    # Plates
                    if self.plate_model is not None:
                        rp = self.plate_model.predict(
                            source=patch, imgsz=TILE, conf=CONF, iou=IOU, augment=True, verbose=False
                        )[0]  # [web:36]
                        if rp.boxes is not None and len(rp.boxes) > 0:
                            bp = rp.boxes.xyxy.detach().clone().contiguous().cpu()
                            bp = scale_boxes(rp.orig_shape, bp, patch.shape[:2])  # [web:86]
                            for bx in bp.numpy():
                                bx[0] += x; bx[2] += x
                                bx[1] += y; bx[3] += y
                                blur_box(img0, bx, w0, h0, k=51)

                    # Faces
                    if self.face_model is not None:
                        rf = self.face_model.predict(
                            source=patch, imgsz=TILE, conf=CONF, iou=IOU, augment=True, verbose=False
                        )[0]  # [web:36]
                        if rf.boxes is not None and len(rf.boxes) > 0:
                            bf = rf.boxes.xyxy.detach().clone().contiguous().cpu()
                            bf = scale_boxes(rf.orig_shape, bf, patch.shape[:2])  # [web:86]
                            for bx in bf.numpy():
                                bx[0] += x; bx[2] += x
                                bx[1] += y; bx[3] += y
                                blur_box(img0, bx, w0, h0, k=51)

            out_path = os.path.join(self.out_dir, os.path.basename(in_path))
            cv2.imwrite(out_path, img0)
            self.progress.emit(int(idx/total*100), f"Saved: {os.path.basename(in_path)}")

        self.finished.emit()

    def stop(self):
        self.stop_flag = True

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face & License Plate Blurring")

        self.in_edit  = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit()
        self.btn_in   = QtWidgets.QPushButton("Browse Input")
        self.btn_out  = QtWidgets.QPushButton("Browse Output")
        self.chk_faces  = QtWidgets.QCheckBox("Blur Faces")
        self.chk_plates = QtWidgets.QCheckBox("Blur Plates")
        self.chk_faces.setChecked(True)
        self.chk_plates.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.status   = QtWidgets.QLabel("Ready")

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(QtWidgets.QLabel("Input Folder:"), 0, 0)
        layout.addWidget(self.in_edit, 0, 1)
        layout.addWidget(self.btn_in, 0, 2)
        layout.addWidget(QtWidgets.QLabel("Output Folder:"), 1, 0)
        layout.addWidget(self.out_edit, 1, 1)
        layout.addWidget(self.btn_out, 1, 2)
        layout.addWidget(self.chk_faces, 2, 0)
        layout.addWidget(self.chk_plates, 2, 1)
        layout.addWidget(self.start_btn, 3, 0)
        layout.addWidget(self.stop_btn, 3, 1)
        layout.addWidget(self.progress, 4, 0, 1, 3)
        layout.addWidget(self.status, 5, 0, 1, 3)

        self.btn_in.clicked.connect(self.pick_input)
        self.btn_out.clicked.connect(self.pick_output)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        self.worker = None

    def pick_input(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d:
            self.in_edit.setText(d)

    def pick_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.out_edit.setText(d)

    def start(self):
        in_dir = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            self.status.setText("Please choose a valid input folder.")
            return
        if not out_dir:
            self.status.setText("Please choose an output folder.")
            return

        self.progress.setValue(0)
        self.status.setText("Loading models and starting...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = Worker(
            in_dir=in_dir,
            out_dir=out_dir,
            do_faces=self.chk_faces.isChecked(),
            do_plates=self.chk_plates.isChecked()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Stopping...")

    def on_progress(self, percent, message):
        self.progress.setValue(percent)
        self.status.setText(message)

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Done")

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(700, 200)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
