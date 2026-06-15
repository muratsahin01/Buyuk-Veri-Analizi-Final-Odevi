# Büyük Veri Analizi — Final Projesi
[SONUÇ ÖZETİ DOSYALARIN EN ALT KISMINDA](url)

**Heart Disease Risk Factors and Patient Health Survey**

---

## Projeyi Çalıştırma

### Gereksinimler

- Python 3.10 veya üzeri
- `Heart_Dataset_Cleaned.csv` dosyası (proje klasöründe olmalı)

### 1. Projeyi indir / aç

```bash
git clone https://github.com/muratsahin01/Buyuk-Veri-Analizi-Final-Odevi.git
cd Buyuk-Veri-Analizi-Final-Odevi
```

Veya ZIP olarak indirdiysen klasörü aç.

### 2. Kütüphaneleri kur (ilk seferde bir kez)

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Analizi çalıştır

```bash
python heart_disease_analysis.py
```

### 4. Çıktılar

Script çalışınca:

- Terminalde **Bölüm A–F** sonuçları yazdırılır
- Aynı klasöre **PNG grafikler** kaydedilir (`plot1_...png`, `plot14_...png` vb.)

> **Not:** `Heart_Dataset_Cleaned.csv` ile `heart_disease_analysis.py` aynı klasörde olmalıdır.

---

## Proje Hakkında

Bu proje, **327 bireye** ait sağlık ve yaşam tarzı verilerini içeren `Heart_Dataset_Cleaned.csv` veri setinin analizini kapsar. Amaç; kalp hastalığı risk faktörlerini incelemek ve `Heart patient` (Yes/No) hedef değişkenini tahmin eden makine öğrenmesi modelleri kurmaktır.

### Veri Seti

| Özellik | Değer |
|---|---|
| Satır | 327 |
| Sütun | 21 |
| Hedef değişken | `Heart patient` (Yes: 108, No: 219) |
| Eksik değer | Yok |
| Tekrarlayan kayıt | Yok |

### Kullanılan Teknolojiler

- **Python**
- **Pandas** — veri işleme
- **Matplotlib & Seaborn** — görselleştirme
- **Scikit-learn** — makine öğrenmesi

---

## Analiz Bölümleri

| Bölüm | İçerik |
|---|---|
| A | Veri setini tanıma |
| B | Tanımlayıcı istatistikler ve aykırı değer kontrolü |
| C | Görselleştirmeler (bar, boxplot, histogram, korelasyon vb.) |
| D | Risk faktörü analizi (14 değişken) |
| E | Makine öğrenmesi (Logistic Regression, Decision Tree, Random Forest) |
| F | Model yorumlama özeti |

### Makine Öğrenmesi Sonuçları (özet)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.7727 | 0.7903 |
| Decision Tree | 0.7273 | 0.7810 |
| Random Forest | 0.7424 | 0.7996 |

---

## Dosya Yapısı

```
Buyuk-Veri-Analizi-Final-Odevi/
├── heart_disease_analysis.py    # Ana analiz kodu
├── Heart_Dataset_Cleaned.csv    # Veri seti
├── finalödevi.docx / .pdf       # Analiz raporu
├── sonuc_ozeti.docx / .pdf      # Kısa sonuç özeti
├── README.md                    # Bu dosya
└── plot*.png                    # Grafikler
```
