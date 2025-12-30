# 🚀 Crypto AI: GRU Tabanlı Fiyat Değişim Analizi

Bu proje, **GRU (Gated Recurrent Unit)** derin öğrenme mimarisini kullanarak Bitcoin (BTC) ve Solana (SOL) için kısa vadeli fiyat hareketlerini tahmin etmeyi amaçlar.

Sistem, klasik fiyat tahmini yerine **"Delta Learning" (Fark Öğrenme)** yöntemini kullanır. Model, bir sonraki gün fiyatın kaç dolar olacağını değil, bugüne göre **ne kadar artacağını veya azalacağını** (değişim miktarını) öğrenir.

## 🌟 Proje Özellikleri

* **⚡ Verimli Mimari (GRU):** LSTM'e göre daha hızlı eğitim sağlayan ve daha az bellek tüketen, 2 katmanlı ve 256 hücreli GRU yapısı kullanılmıştır.
* **📉 Delta (Fark) Tahmini:** Model, `Close(t) - Close(t-1)` formülüyle hesaplanan değişimi analiz eder. Bu yöntem veriyi durağanlaştırır ve modelin trendleri daha iyi yakalamasını sağlar.
* **🔄 Kayan Pencere (Sliding Window):** Geçmiş 30 günün kapanış verilerine bakarak 31. günün hareketini tahmin eder.
* **🌐 İnteraktif Web Arayüzü:** Gradio ile oluşturulmuş, canlı veri çeken ve tahminleri görselleştiren modern bir analiz paneli sunar.
* **📊 Görsel Raporlama:** Eğitim sonrası kayıp (loss) grafikleri ve fiyat karşılaştırma grafikleri otomatik olarak üretilir.

## 📂 Proje Yapısı

* **`model.py`**: PyTorch kullanılarak oluşturulan **GRU** ağ mimarisini içerir (Input Size: 1, Hidden Size: 256, Layers: 2).
* **`train.py`**:
    * Yahoo Finance API (`yfinance`) üzerinden veri çeker.
    * Veriyi işler ve normalize eder (MinMaxScaler).
    * Modeli eğitir (`MSELoss` ve `Adam` optimizasyonu ile).
    * Sonuçları `.pth` (model ağırlıkları) ve `.pkl` (ölçekleyiciler) dosyalarına kaydeder.
* **`serve.py`**: Eğitilmiş modeli yükler ve son kullanıcı için canlı analiz yapan bir web sunucusu başlatır.
* **`requirements.txt`**: Projenin çalışması için gerekli kütüphane listesi.

## 🛠️ Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edin:

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
