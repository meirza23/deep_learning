import gradio as gr
import torch
import numpy as np
import joblib
import yfinance as yf
from model import GRUModel
import datetime
import matplotlib.pyplot as plt
import os
import requests
import pandas as pd

def download_icons():
    icons = {
        "btc_logo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/64px-Bitcoin.svg.png",
        "sol_logo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Solana_cryptocurrency_two.jpg/64px-Solana_cryptocurrency_two.jpg"
    }
    for filename, url in icons.items():
        if not os.path.exists(filename):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'} 
                response = requests.get(url, headers=headers)
                with open(filename, 'wb') as f: f.write(response.content)
            except: pass
download_icons()

def analyze_crypto(symbol):
    fig = plt.figure(figsize=(10, 5))
    try:
        model_path = f"model_{symbol}.pth"
        input_scaler_path = f"scaler_input_{symbol}.pkl"
        target_scaler_path = f"scaler_target_{symbol}.pkl"
        
        if not os.path.exists(model_path): return fig, "⚠️ Önce train.py çalıştırın."

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        model = GRUModel(input_size=1, hidden_size=256, num_layers=2)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except: return fig, "⚠️ Model uyumsuz. train.py çalıştırın."

        scaler_input = joblib.load(input_scaler_path)
        scaler_target = joblib.load(target_scaler_path)
        model.eval()

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=150) 
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs('Close', level=0, axis=1)
                except: df = df.xs('Adj Close', level=0, axis=1)
            if len(df.columns) > 1 and 'Close' in df.columns: df = df[['Close']]
            df.columns = ['Close']
            
            if len(df) < 30: return fig, "Yetersiz veri."
            
            values = df.values.astype(float).flatten()
            last_seq = values[-30:].reshape(-1, 1) # Son 30 günün fiyatları
            current_price = values[-1]

        except Exception as e: return fig, f"Veri hatası: {e}"

        # Fiyatları ölçekle
        input_scaled = scaler_input.transform(last_seq)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction_diff_scaled = model(input_tensor)
            
        # Farkı gerçek boyuta çevir (Örn: +200$)
        pred_diff = scaler_target.inverse_transform(prediction_diff_scaled.numpy())
        predicted_change = float(pred_diff.item())
        
        # TAHMİNİ HESAPLA: Bugün + Değişim
        predicted_price = round(current_price + predicted_change, 2)

        plot_data = values[-90:] 
        plt.plot(range(len(plot_data)), plot_data, label='Geçmiş', color='#1f77b4', linewidth=2)
        plt.scatter(len(plot_data), predicted_price, color='red', s=100, label='Tahmin', zorder=5)
        plt.plot([len(plot_data)-1, len(plot_data)], [plot_data[-1], predicted_price], color='red', linestyle='--', alpha=0.7)
        plt.title(f"{symbol} Analizi (Delta Tahmini)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        yuzde = (predicted_change / current_price) * 100
        icon = "🚀 YÜKSELİŞ" if predicted_change > 0 else "🔻 DÜŞÜŞ"
        fark_str = f"+${predicted_change:.2f}" if predicted_change > 0 else f"-${abs(predicted_change):.2f}"
        
        report = (
            f"✅ DELTA ANALİZİ: {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Fiyat: ${current_price:.2f}\n"
            f"🔮 Tahmin: ${predicted_price}\n"
            f"📊 Fark: {fark_str} (%{yuzde:.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Yön: {icon}"
        )
        return fig, report

    except Exception as e: return fig, f"Hata: {str(e)}"

custom_css = "footer {visibility: hidden !important;} .gradio-container {min-height: 0px !important;}"
def click_btc(): return analyze_crypto("BTC-USD")
def click_sol(): return analyze_crypto("SOL-USD")

with gr.Blocks(title="Kripto AI Terminal") as demo:
    gr.Markdown("# 🧠 Kripto Para Yapay Zeka Terminali")
    gr.Markdown("Fiyat Farkı (Delta) Öğrenme Modeli - Yüksek Yön Başarısı")
    with gr.Row():
        btn_btc = gr.Button("Bitcoin (BTC)", icon="btc_logo.png")
        btn_sol = gr.Button("Solana (SOL)", icon="sol_logo.png")
    with gr.Row():
        plot_output = gr.Plot(label="Grafik")
        text_output = gr.Textbox(label="Rapor", lines=10)
    btn_btc.click(click_btc, None, [plot_output, text_output])
    btn_sol.click(click_sol, None, [plot_output, text_output])

if __name__ == "__main__":
    demo.launch(share=True, css=custom_css)