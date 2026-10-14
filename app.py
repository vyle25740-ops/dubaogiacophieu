pip install streamlit yfinance pandas numpy matplotlib scikit-learn tensorflow
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import random
import tensorflow as tf
from datetime import datetime, timedelta

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Dự đoán giá cổ phiếu LSTM",
    page_icon="📈",
    layout="wide"
)

# Fix seed để kết quả ổn định
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ==================== HÀM HỖ TRỢ ====================
@st.cache_data(show_spinner=False)
def download_data(ticker: str, start: str, end: str):
    """Tải dữ liệu từ Yahoo Finance"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    return df

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(window_size=60):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(ticker: str, start_date: str, end_date: str, window_size: int = 60, epochs: int = 15):
    """Train model và trả về kết quả dự đoán"""
    df = download_data(ticker, start_date, end_date)
    if df is None or len(df) < window_size + 50:
        return None, "Không đủ dữ liệu hoặc mã cổ phiếu không hợp lệ."

    # Chỉ lấy giá Close
    data = df[['Close']].values
    dates = df.index

    # Chia train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Tạo sequence
    X_train, y_train = create_sequences(train_scaled, window_size)
    test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
    X_test, y_test = create_sequences(test_combined, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Validation split
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    # Build & train model
    model = build_lstm_model(window_size)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict trên test set
    test_pred = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
    mae = mean_absolute_error(y_test_real, test_pred)
    r2 = r2_score(y_test_real, test_pred)

    # Dự báo 10 ngày tới
    last_60 = np.vstack((train_scaled, test_scaled))[-window_size:]
    last_60 = last_60.reshape(1, window_size, 1)
    future_preds = []
    for _ in range(10):
        next_pred = model.predict(last_60, verbose=0)
        future_preds.append(next_pred[0, 0])
        last_60 = np.append(last_60[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds = scaler.inverse_transform(future_preds)

    # Ngày tương lai
    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(10)]

    # Hướng dự đoán (so sánh giá cuối test với giá trung bình 10 ngày tới)
    last_real_price = float(y_test_real[-1][0])
    avg_future = float(np.mean(future_preds))
    direction = "TĂNG" if avg_future > last_real_price else "GIẢM"
    change_pct = ((avg_future - last_real_price) / last_real_price) * 100

    results = {
        "history": history.history,
        "y_test_real": y_test_real.flatten(),
        "test_pred": test_pred.flatten(),
        "future_preds": future_preds.flatten(),
        "future_dates": future_dates,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "last_price": last_real_price,
        "avg_future": avg_future,
        "direction": direction,
        "change_pct": change_pct,
        "dates_test": dates[train_size + window_size:]  # gần đúng
    }
    return results, None

# ==================== GIAO DIỆN ====================
st.title("📈 Ứng dụng Dự đoán Giá Cổ phiếu bằng LSTM")
st.markdown("Nhập mã chứng khoán (ví dụ: `VNM.VN`, `FPT.VN`, `AAPL`, `TSLA`) để dự đoán xu hướng giá trong 10 ngày tới.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    ticker = st.text_input("Mã chứng khoán", value="VNM.VN").upper().strip()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày", value=datetime(2018, 1, 1))
    with col2:
        end_date = st.date_input("Đến ngày", value=datetime.now())

    window_size = st.slider("Window size (số ngày)", 30, 90, 60)
    epochs = st.slider("Số epochs", 5, 30, 12)

    run_button = st.button("🚀 Chạy dự đoán", type="primary", use_container_width=True)

# Main content
if run_button:
    if not ticker:
        st.error("Vui lòng nhập mã chứng khoán!")
    else:
        with st.spinner(f"Đang tải dữ liệu và huấn luyện model cho **{ticker}**... (có thể mất 30-90 giây)"):
            results, error = train_and_predict(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                window_size=window_size,
                epochs=epochs
            )

        if error:
            st.error(error)
        else:
            # ========== KẾT QUẢ TỔNG QUAN ==========
            st.success(f"Dự đoán hoàn tất cho **{ticker}**")

            # Card hướng đi
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Giá hiện tại (cuối test)", f"{results['last_price']:,.0f}")
            with col_b:
                st.metric("Giá TB 10 ngày tới", f"{results['avg_future']:,.0f}")
            with col_c:
                delta_color = "normal" if results['direction'] == "TĂNG" else "inverse"
                st.metric(
                    "Xu hướng dự đoán",
                    results['direction'],
                    delta=f"{results['change_pct']:+.2f}%",
                    delta_color=delta_color
                )
            with col_d:
                st.metric("R² Score", f"{results['r2']:.4f}")

            # Metrics chi tiết
            st.subheader("📊 Đánh giá model trên tập Test")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{results['rmse']:,.2f}")
            m2.metric("MAE", f"{results['mae']:,.2f}")
            m3.metric("R²", f"{results['r2']:.4f}")

            # ========== BIỂU ĐỒ ==========
            st.subheader("📉 Biểu đồ Loss (Train vs Validation)")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(results['history']['loss'], label='Train Loss', color='#1f77b4')
            ax1.plot(results['history']['val_loss'], label='Validation Loss', color='#ff7f0e')
            ax1.set_title("Loss Train vs Validation")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close()

            st.subheader("📈 Giá thật vs Dự báo trên tập Test")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(results['y_test_real'], label="Giá thật", color='#2ca02c', linewidth=1.5)
            ax2.plot(results['test_pred'], label="Dự báo", color='#d62728', linewidth=1.5, alpha=0.8)
            ax2.set_title(f"So sánh Giá thật và Dự báo - {ticker}")
            ax2.set_xlabel("Ngày (test set)")
            ax2.set_ylabel("Giá")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()

            st.subheader("🔮 Dự báo 10 ngày tới")
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            # Vẽ phần test cuối
            ax3.plot(results['y_test_real'], label="Giá thật (test)", color='#2ca02c', alpha=0.7)
            ax3.plot(results['test_pred'], label="Dự báo test", color='#d62728', alpha=0.7)
            
            # Vẽ future
            future_x = range(len(results['y_test_real']), len(results['y_test_real']) + 10)
            ax3.plot(future_x, results['future_preds'], marker='o', markersize=6,
                     color='#9467bd', linewidth=2, label="Dự báo 10 ngày tới")
            
            # Đánh dấu điểm nối
            ax3.axvline(x=len(results['y_test_real']) - 1, color='gray', linestyle='--', alpha=0.5)
            
            ax3.set_title(f"Dự báo 10 ngày tới - {ticker} | Xu hướng: {results['direction']} ({results['change_pct']:+.2f}%)")
            ax3.set_xlabel("Ngày")
            ax3.set_ylabel("Giá")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close()

            # Bảng dự báo 10 ngày
            st.subheader("📋 Chi tiết dự báo 10 ngày")
            future_df = pd.DataFrame({
                "Ngày dự kiến": [d.strftime("%Y-%m-%d") for d in results['future_dates']],
                "Giá dự đoán": [f"{p:,.0f}" for p in results['future_preds']]
            })
            st.dataframe(future_df, use_container_width=True, hide_index=True)

            # Ghi chú
            st.info(
                f"**Ghi chú:** Model LSTM được huấn luyện trên dữ liệu từ {start_date} đến {end_date}. "
                f"Dự báo chỉ mang tính chất tham khảo, không phải lời khuyên đầu tư."
            )

else:
    st.info("👈 Nhập mã chứng khoán ở sidebar và bấm **Chạy dự đoán** để bắt đầu.")
    st.markdown("""
    ### Hướng dẫn nhanh
    - **Mã Việt Nam**: thêm `.VN` (ví dụ `VNM.VN`, `FPT.VN`, `HPG.VN`, `VCB.VN`)
    - **Mã quốc tế**: `AAPL`, `MSFT`, `TSLA`, `GOOGL`, `BTC-USD`...
    - Thời gian train mặc định từ 2018 đến hiện tại.
    - Model dùng LSTM 2 tầng + EarlyStopping.
    """)
