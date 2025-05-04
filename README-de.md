# 🔍 Vision-Validator

**Deutsche Version** | [English Version](README.md)

Vision-Validator ist eine leistungsstarke Web-Anwendung zum Testen, Vergleichen und Anwenden von YOLO-Modellen für Objekterkennung und Instanzsegmentierung. Mit einer benutzerfreundlichen Oberfläche können Sie verschiedene Modelle laden, Bilder analysieren und Live-Videostreams verarbeiten.

![Yolo Vision Interface](https://placeholder.image)

## ✨ Features

- 🎯 **Multi-Model Support**: Unterstützt YOLO .pt, .onnx und .engine Modelle
- 📸 **Live Camera Detection**: Echtzeit-Objekterkennung über Webcam
- 🖼️ **Image Upload & Detection**: Bild-Upload für Batch-Verarbeitung
- 🎨 **Instance Segmentation**: Pixel-genaue Objektumrisse
- ⚙️ **Anpassbare Einstellungen**: Konfidenz-Schwellenwerte, Maskenfarben und Transparenz
- 💾 **Automatisches Speichern**: Erkannte Objekte werden lokal gespeichert
- 🔄 **Model Management**: Upload, Umbenennen und Löschen von Modellen
- 🚀 **GPU-Beschleunigung**: Automatische GPU-Nutzung wenn verfügbar
- 🧹 **Memory Management**: Automatische Speicherbereinigung für stabilen Betrieb

## 🛠️ Installation

### Systemvoraussetzungen
- Python 3.8 oder höher
- Webcam (optional, für Live-Erkennung)
- CUDA-fähige GPU (optional, für Beschleunigung)

### Schritt 1: Repository klonen
```bash
git clone https://github.com/Loky-Coffee/Yolo-Vision.git
cd Yolo-Vision
```

### Schritt 2: Virtual Environment erstellen
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Schritt 3: Abhängigkeiten installieren

Für CPU-only Installation:
```bash
pip install -r requirements.txt
```

Für GPU-Beschleunigung (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Schritt 4: Anwendung starten
```bash
python app.py
```

Die Anwendung ist nun unter http://localhost:5000 verfügbar.

## 📁 Projektstruktur

```
Yolo-Vision/
├── app.py                  # Hauptanwendung
├── requirements.txt        # Python-Abhängigkeiten
├── models/                 # Modell-Speicher
├── uploads/               # Hochgeladene Bilder
├── results/               # Erkennungsergebnisse
├── templates/
│   └── index.html         # Frontend-Template
└── README.md              # Diese Datei
```

## 🚦 Erste Schritte

### 1. Modell hochladen
- Klicken Sie auf "Choose File" um ein YOLO-Modell (.pt, .onnx, .engine) auszuwählen
- Das Modell wird automatisch erkannt und geladen

### 2. Objekterkennung

**Für Bilder:**
- Klicken Sie auf "Choose Image File"
- Wählen Sie ein Bild aus
- Klicken Sie auf "Detect"
- Das Ergebnis wird angezeigt

**Für Live-Video:**
- Stellen Sie sicher, dass eine Webcam angeschlossen ist
- Das Live-Feed wird automatisch gestartet
- Erkannte Objekte werden automatisch markiert und gespeichert

### 3. Einstellungen anpassen
- **Confidence Threshold**: Mindestwahrscheinlichkeit für Erkennungen (0-1)
- **Camera Resolution**: Webcam-Auflösung einstellen
- **Mask Color**: Farbe der Segmentierungsmasken
- **Mask Alpha**: Transparenz der Masken
- **Model Input Size**: Eingabegröße für das Modell

## 🎮 Verwendung

### Modell-Management
```python
# Modell laden
POST /load_model
{
    "model_index": 0  # Index des Modells in der Liste
}

# Modell umbenennen
POST /rename_model
{
    "model_index": 0,
    "new_name": "my_model.pt"
}

# Modell löschen
POST /delete_model/0  # Model index
```

### Erkennung
```python
# Bild-Erkennung
POST /detect_image
Files: {'image': file}

# Cooldown zurücksetzen
POST /reset_cooldown
```

## ⚙️ Konfiguration

### Einstellungen in app.py
```python
settings = {
    'conf_threshold': 0.5,      # Konfidenz-Schwellenwert
    'camera_width': 1280,       # Kamera-Breite
    'camera_height': 720,       # Kamera-Höhe
    'mask_alpha': 0.5,          # Masken-Transparenz
    'mask_color': '#FF0000',    # Masken-Farbe (Hex)
    'model_input_width': 640,   # Modell-Eingabebreite
    'model_input_height': 640   # Modell-Eingabehöhe
}
```

## 🐛 Fehlerbehandlung

### Häufige Probleme

**Kamera nicht erkannt:**
- Überprüfen Sie Berechtigungen
- Stellen Sie sicher, dass die Kamera nicht von einer anderen Anwendung verwendet wird
- Schließen Sie andere Tabs, die die Kamera verwenden könnten
- Testen Sie mit: `cv2.VideoCapture(0)`

**Modell lädt nicht:**
- Überprüfen Sie Kompatibilität (YOLO Format)
- Stellen Sie ausreichend RAM/VRAM sicher

**GPU nicht erkannt:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Logs
Anwendung mit Logging starten:
```bash
python app.py 2>&1 | tee app.log
```

## 🚀 Performance-Optimierung

- **GPU verwenden**: Automatisch wenn CUDA verfügbar
- **Modelloptimierung**: TensorRT-Modelle (.engine) für beste Performance
- **Speicherverwaltung**: Automatische Bereinigung alle 5 Minuten
- **Frame-Rate**: Angepasst auf 30 FPS für flüssige Anzeige

## 🔒 Sicherheit

- Automatische Bereinigung alter Dateien (max. 20)
- Sichere Datei-Uploads
- Thread-sichere Operationen
- Speicher-Lecks werden automatisch behoben

## 📚 Verwendete Technologien

- **Backend**: Flask, OpenCV, PyTorch, Ultralytics
- **Frontend**: HTML, JavaScript, Bootstrap
- **Computer Vision**: YOLO, OpenCV
- **Threading**: Python threading, concurrent.futures

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte erstellen Sie einen Fork und öffnen Sie eine Pull Request.

## 📄 Lizenz

MIT License - siehe LICENSE Datei für Details.

## 🙏 Danksagungen

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Community](https://opencv.org/)
- Alle Mitwirkenden und Tester

Entwickelt mit ❤️ von [Loky Coffee](https://github.com/Loky-Coffee)