# 🧠 Crypto AI: Hibrit Derin Öğrenme ile Fiyat Analizi

Bu proje, **CNN (Evrişimli Sinir Ağları)** ve **LSTM (Uzun Kısa Süreli Bellek)** mimarilerini birleştiren hibrit bir yapay zeka modeli kullanarak Bitcoin (BTC) ve Solana (SOL) fiyat hareketlerini tahmin etmeyi amaçlar.

Proje, **Yahoo Finance** üzerinden canlı veri çeker, **RSI ve MACD** gibi teknik indikatörlerle veriyi zenginleştirir ve **Log-Return (Yüzdesel Getiri)** öğrenme stratejisi ile geleceği tahmin eder.

## 🚀 Proje Özellikleri

* **🧬 Hibrit Mimari (CNN + LSTM):** CNN ile fiyat grafiğindeki desenleri yakalar, LSTM ile zamansal trendleri analiz eder.
* **📊 Çoklu Özellik (Multi-Feature):** Model sadece fiyata değil, **RSI (Momentum)** ve **MACD (Trend)** verilerine de bakarak karar verir.
* **🎯 Delta Learning:** Model fiyatın kendisini değil, **değişim oranını (Log-Return)** öğrenir. Bu sayede "lagging" (gecikme) sorunu çözülmüştür.
* **🔒 Kararlı Sonuçlar:** `Seed` sabitleme yöntemi ile her eğitimde tutarlı ve tekrarlanabilir sonuçlar üretir.
* **🌐 Web Arayüzü:** Gradio tabanlı modern bir analiz paneli sunar.

## 📂 Proje Yapısı

* **`model.py`**: Hibrit (CNN+LSTM) Yapay Sinir Ağı mimarisinin tanımlandığı dosya.
* **`train.py`**: Veri çekme, indikatör hesaplama (RSI/MACD), model eğitimi ve başarı grafiklerinin oluşturulduğu modül.
* **`serve.py`**: Eğitilen modeli kullanarak canlı analiz yapan kullanıcı dostu web arayüzü.
* **`requirements.txt`**: Projenin çalışması için gerekli kütüphaneler.

## 🛠️ Kurulum

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Modeli Eğitin:**
    ```bash
    python train.py
    ```
    *Bu işlem veri setini indirecek, teknik indikatörleri hesaplayacak ve yapay zeka modellerini oluşturacaktır.*

3.  **Arayüzü Başlatın:**
    ```bash
    python serve.py
    ```
    *Terminalde verilen linke tıklayarak tarayıcınızda sistemi kullanabilirsiniz.*

## 📊 Model Performansı (Test Verileri)

Modelimiz, farklı volatilite seviyelerine sahip varlıklar üzerinde test edilmiştir. **Bitcoin (Daha Stabil)** üzerinde yüksek yön başarısı sağlanırken, **Solana (Yüksek Volatilite)** üzerinde piyasa ortalaması yakalanmıştır.

| Varlık | 📉 MAPE (Fiyat Hatası) | 🧭 Yön Başarısı | Analiz |
| :--- | :--- | :--- | :--- |
| **Bitcoin (BTC)** | **%1.43** | **%56.22** | ✅ Model piyasa yönünü yüksek başarıyla tahmin etmektedir. |
| **Solana (SOL)** | **%3.14** | **%50.24** | ⚖️ Yüksek volatilite nedeniyle model fiyatı takip etmekte, ancak anlık kırılımlarda nötr kalmaktadır. |

*(Detaylı başarı grafikleri proje klasöründe `grafik_tahmin_BTC-USD.png` ve `grafik_tahmin_SOL-USD.png` dosyalarında mevcuttur.)*

## 🧠 Kullanılan Teknolojiler

* **Dil:** Python 3.9+
* **Yapay Zeka:** PyTorch (CNN & LSTM Layers)
* **Veri Analizi:** Pandas, NumPy, Scikit-learn
* **Teknik Analiz:** RSI, MACD, Log-Return Hesaplamaları
* **Görselleştirme:** Matplotlib
* **Arayüz:** Gradio
* **Veri Kaynağı:** Yahoo Finance API (yfinance)
