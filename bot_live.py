import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import pandas_ta as ta
import schedule
import time
import logging
import sys
from datetime import datetime
from data_processor_utils import *

from model import CTTS

# =========================
# Configuration
# =========================
API_BASE_URL         = "http://localhost:8080/api"
TIMEFRAME            = "30"
CONFIDENCE_THRESHOLD = 0.95
RISK_DOLLARS         = 800.0
MIN_RR               = 0.2
MIN_N_ATR_SL         = 0
MAX_N_ATR_SL         = 1000
N_CANDLES  = 1000
ONE_TRADE_AT_A_TIME = False

# =========================
# Position auto‑close settings
# =========================
POSITION_MAX_AGE_MINUTES = 60 * 24        # Close position after this many minutes
ENABLE_AUTO_CLOSE_PROFIT_AGED = True # Master switch
MIN_PROFIT_TO_CLOSE = 0.0            # Minimum profit (in account currency) to close

SYMBOLS = [
    'EURJPY',
    'EURNZD',
    'USDCAD',
    'USDCHF',
    'NZDUSD',
    'NZDCHF',
    'XAUUSD',
    'AUDUSD'
]



MODEL_PATH = 'best_ctts_model.pth'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Logging — encodage UTF-8 explicite pour Windows
# =========================
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s | %(levelname)s | %(message)s',
    handlers = [
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# =========================
# Chargement modele
# =========================
log.info("Chargement du modele...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = CTTS(
    input_dim      = 18,
    seq_len        = 80,
    cnn_kernel_size= 16,
    cnn_stride     = 8,
    d_model        = 128,
    nhead          = 4,
    num_layers     = 4,
    dropout        = 0.3,
    num_classes    = 2
).to(DEVICE)
model.load_state_dict(
    checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint
    else checkpoint.get('state_dict', checkpoint)
)
model.to(DEVICE)
model.eval()
log.info("Modele charge.")

# =========================
# Fonctions API MT5
# =========================
def get_symbol_info(symbol):
    r = requests.get(f"{API_BASE_URL}/infos/{symbol}", timeout=10)
    r.raise_for_status()
    return r.json()

def get_candles(symbol, timeframe, count):
    r = requests.get(
        f"{API_BASE_URL}/rates/{symbol}/{timeframe}",
        params={'count': count},
        timeout=10
    )
    r.raise_for_status()
    return r.json()

def get_positions(symbol):
    r = requests.get(f"{API_BASE_URL}/positions/{symbol}", timeout=10)
    r.raise_for_status()
    return r.json()

def open_position(symbol, order_type, size, tp, sl):
    r = requests.post(
        f"{API_BASE_URL}/positions/open/{symbol}/{order_type}",
        params={'size': size, 'stop_loss': sl, 'tp_distance': tp},
        timeout=10
    )
    r.raise_for_status()
    return r.json()

def close_position(symbol, ticket):
    """Close a specific position by ticket number."""
    try:
        r = requests.delete(
            f"{API_BASE_URL}/positions/close/{symbol}/{ticket}",
            timeout=10
        )
        r.raise_for_status()
        result = r.json()
        log.info(f"[{symbol}] Closed position ticket {ticket}: {result}")
        return result
    except Exception as e:
        log.exception(f"[{symbol}] Failed to close position {ticket}: {e}")
        return None

    
# =========================
# Position sizing
# =========================
def compute_lot_size(risk_dollars, sl_distance, pip_size, tick_size,
                     tick_value, volume_min, volume_max):
    if pip_size <= 0 or tick_value <= 0 or sl_distance <= 0 or tick_size <= 0:
        return None

    tick_value_per_pip = tick_value * (pip_size / tick_size)
    sl_pips = sl_distance / pip_size
    lots    = risk_dollars / (sl_pips * tick_value_per_pip)

    if volume_min > 0:
        lots = round(lots / volume_min) * volume_min
        lots = round(lots, 2)
        lots = max(lots, volume_min)
    if volume_max > 0:
        lots = min(lots, volume_max)

    return lots if lots > 0 else None



def apply_local_scaling(X_raw, scale_mask):
    X_scaled = np.copy(X_raw)
    for f in range(X_raw.shape[1]):
        if scale_mask[f]:
            seq        = X_raw[:, f]
            minv, maxv = np.min(seq), np.max(seq)
            if maxv - minv > 1e-9:
                X_scaled[:, f] = (seq - minv) / (maxv - minv)
            else:
                X_scaled[:, f] = 0.0
    return X_scaled

def prepare_sequence(df):
    df = df.copy()
    df.ta.atr(high=df['high'], low=df['low'], close=df['close'],
              length=ATR_PERIOD, append=True)
    atr_col = [c for c in df.columns if 'ATR' in c.upper()][0]
    df      = df.rename(columns={atr_col: 'atr'})
    df['ema_short'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema_long']  = df['close'].ewm(span=EMA_LONG,  adjust=False).mean()
    df = df.dropna(subset=['atr', 'ema_short', 'ema_long']).reset_index(drop=True)

    if len(df) < SEQ_LEN:
        return None, None

    swing_highs, swing_lows = find_swings(
            df['high'].values,
            df['low'].values,
            SWING_LEN
        )
    data_matrix, scale_mask = build_feature_matrix(df, swing_highs, swing_lows)
    seq_raw    = data_matrix[-SEQ_LEN:]
    seq_scaled = apply_local_scaling(seq_raw, scale_mask)
    return seq_scaled, df


# =========================
# Traitement d'une paire
# =========================

def close_aged_positive_positions(symbol):
    """Check all open positions for `symbol` and close those that are older than
    POSITION_MAX_AGE_MINUTES and show a positive profit."""
    try:
        positions = get_positions(symbol)
        if not isinstance(positions, list) or len(positions) == 0:
            return

        now_ts = time.time()
        for pos in positions:
            # pos is a dict from the API: {"ticket": ..., "time": ..., "profit": ...}
            open_time = pos.get('time')
            profit = pos.get('profit', 0.0)
            ticket = pos.get('ticket')
            if open_time is None or ticket is None:
                continue

            age_minutes = (now_ts - open_time) / 60.0
            if age_minutes >= POSITION_MAX_AGE_MINUTES and profit > MIN_PROFIT_TO_CLOSE:
                log.info(f"[{symbol}] Position {ticket} age={age_minutes:.1f}min, profit=${profit:.2f} → closing")
                close_position(symbol, ticket)
            else:
                log.debug(f"[{symbol}] Position {ticket} age={age_minutes:.1f}min, profit=${profit:.2f} → not closing")
    except Exception as e:
        log.exception(f"[{symbol}] Error checking aged positions: {e}")

def manage_close_all_positions():
    """Iterate over all symbols and close aged positive positions."""
    if not ENABLE_AUTO_CLOSE_PROFIT_AGED:
        return
    log.info("Checking for aged positive positions...")
    for symbol in SYMBOLS:
        close_aged_positive_positions(symbol)
        time.sleep(0.5)  # slight delay to avoid hammering the API

def process_symbol(symbol):
    try:
        if ONE_TRADE_AT_A_TIME:
            positions = get_positions(symbol)
            if isinstance(positions, list) and len(positions) > 0:
                log.info(f"[{symbol}] Position deja ouverte - skip")
                return

        # 2. Infos symbole
        info       = get_symbol_info(symbol)
        pip_size   = info.get('pip_size',            0)
        tick_size  = info.get('trade_tick_size',      0)
        tick_value = info.get('trade_tick_value',     0)
        volume_min = info.get('volume_min',           0.01)
        volume_max = info.get('volume_max',           100.0)

        if pip_size <= 0 or tick_value <= 0 or tick_size <= 0:
            log.warning(f"[{symbol}] Infos invalides - skip")
            return

        tick_value_per_pip = tick_value * (pip_size / tick_size)
        log.info(f"[{symbol}] tick_value_per_pip={tick_value_per_pip:.4f}$")

        # 3. Bougies
        candles = get_candles(symbol, TIMEFRAME, N_CANDLES)
        if not candles or len(candles) < N_CANDLES:
            log.warning(f"[{symbol}] Donnees insuffisantes - skip")
            return

        # 4. DataFrame — retire la bougie courante non fermee
        df = pd.DataFrame(candles)
        df = df.rename(columns={'time': 'end_time', 'volume': 'tick_volume'})
        df['datetime'] = pd.to_datetime(df['end_time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.iloc[:-1].reset_index(drop=True)

        # 5. Sequence
        seq_scaled, df_processed = prepare_sequence(df)
        if seq_scaled is None:
            log.warning(f"[{symbol}] Sequence invalide - skip")
            return

        # 6. Inference
        X = torch.FloatTensor(seq_scaled).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs      = torch.softmax(model(X), dim=1).cpu().numpy()[0]
            confidence = float(probs.max())
            direction  = int(probs.argmax())

        direction_str = "BUY" if direction == 1 else "SELL"
        log.info(f"[{symbol}] {direction_str} | Confidence: {confidence:.4f}")

        if confidence < CONFIDENCE_THRESHOLD:
            log.info(f"[{symbol}] Confidence insuffisante - skip")
            return

        # 7. Swings
        swing_highs, swing_lows = find_swings(
            df_processed['high'].values,
            df_processed['low'].values,
            SWING_LEN
        )

        entry_price   = df_processed['close'].iloc[-1]
        atr_at_signal = df_processed['atr'].iloc[-1]
        n             = len(df_processed)

        # 8. Niveaux TP/SL
        tp_level, sl_level = get_trade_levels(
            direction, entry_price, atr_at_signal,
            swing_highs, swing_lows, n
        )

        if tp_level is None or sl_level is None:
            log.warning(f"[{symbol}] Swings insuffisants - skip")
            return

        # 9. Filtres risk/reward
        risk   = abs(entry_price - sl_level)
        reward = abs(tp_level    - entry_price)

        if risk <= 0 or reward <= 0:
            log.warning(f"[{symbol}] Risk/Reward invalide - skip")
            return

        sl_pips = risk / pip_size
        rr      = reward / risk

        if rr < MIN_RR:
            log.info(f"[{symbol}] RR={rr:.2f} < {MIN_RR} - skip")
            return

        order_type = 'BUY' if direction == 1 else 'SELL'

        # 10. Lot size
        lots = compute_lot_size(
            risk_dollars = RISK_DOLLARS,
            sl_distance  = risk,
            pip_size     = pip_size,
            tick_size    = tick_size,
            tick_value   = tick_value,
            volume_min   = volume_min,
            volume_max   = volume_max,
        )

        if lots is None:
            log.warning(f"[{symbol}] Lot size invalide - skip")
            return

        log.info(
            f"[{symbol}] {order_type} | "
            f"Entry={entry_price:.5f} | TP={tp_level:.5f} | SL={sl_level:.5f} | "
            f"SL={sl_pips:.1f} pips | RR={rr:.2f} | Lots={lots} | "
            f"Risque=${RISK_DOLLARS}"
        )

        # 11. Envoi ordre
        result = open_position(symbol, order_type, lots, round(tp_level, 5), round(sl_level, 5))
        log.info(f"[{symbol}] Ordre envoye: {result}")

    except requests.exceptions.ConnectionError:
        log.error(f"[{symbol}] API MT5 inaccessible")
    except Exception as e:
        log.exception(f"[{symbol}] Erreur: {e}")


# =========================
# Boucle principale
# =========================
def run_all_symbols():
    log.info("=" * 50)
    log.info(f"Tick - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(
        f"Risk: ${RISK_DOLLARS} | Threshold: {CONFIDENCE_THRESHOLD} | "
        f"MIN_RR: {MIN_RR}"
    )
    log.info("=" * 50)

    for symbol in SYMBOLS:
        process_symbol(symbol)
        time.sleep(1)


# =========================
# Scheduler
# =========================
if __name__ == '__main__':
    log.info(f"Bot demarre | Paires: {SYMBOLS}")
    log.info(f"Device: {DEVICE}")

    run_all_symbols()

    schedule.every().hour.at(":01").do(run_all_symbols)
    schedule.every().hour.at(":31").do(run_all_symbols)
    schedule.every(1).minutes.do(manage_close_all_positions)
    log.info("Scheduler actif...")
    while True:
        schedule.run_pending()
        time.sleep(30)