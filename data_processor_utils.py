# =========================
# TP/SL — fonction centralisée (identique live bot / backtest)
# =========================
import numpy as np

# =========================
# Configuration
# =========================
ATR_PERIOD = 14
EMA_SHORT = 21
EMA_LONG = 50
HORIZON = 48
SEQ_LEN = 80
SWING_LEN = 10
MIN_SWING_ATR = 1.0
MIN_GAP = 5

# Filtres trade — IDENTIQUES au backtest et au live bot
MIN_RR = 0.55
MIN_N_ATR_SL = 1.5
MAX_N_ATR_SL = 5.5

n_features = 18

print(f"MIN_RR={MIN_RR} | SL [{MIN_N_ATR_SL}–{MAX_N_ATR_SL}] ATR | HORIZON={HORIZON}\n")



def get_trade_levels(direction, entry_price, atr, swing_highs, swing_lows, n):
    if direction == 1:  # BUY
        sh_above = np.where(
            (~np.isnan(swing_highs[:n])) & (swing_highs[:n] > entry_price)
        )[0]
        sl_below = np.where(
            (~np.isnan(swing_lows[:n])) & (swing_lows[:n] < entry_price)
        )[0]
        if len(sh_above) == 0 or len(sl_below) == 0:
            return None, None

        tp_level = swing_highs[sh_above[-1]]  # résistance structurelle
        sl_structure = swing_lows[sl_below[-1]]  # support structurel

        # ── SL : on prend le PLUS PROTECTEUR des deux
        # Si le support structural est trop proche (< 1.5 ATR), on recule
        # Si le support structural est trop loin (> 5.5 ATR), on resserre
        sl_atr_floor = entry_price - MAX_N_ATR_SL * atr  # plancher absolu
        sl_atr_ceil = entry_price - MIN_N_ATR_SL * atr  # plafond minimum

        # Le SL final respecte la structure ET les limites ATR
        sl_level = np.clip(sl_structure, sl_atr_floor, sl_atr_ceil)

        # ── Vérification de cohérence : TP doit être au-dessus du SL
        if tp_level <= sl_level:
            return None, None

    else:  # SELL
        sl_below = np.where(
            (~np.isnan(swing_lows[:n])) & (swing_lows[:n] < entry_price)
        )[0]
        sh_above = np.where(
            (~np.isnan(swing_highs[:n])) & (swing_highs[:n] > entry_price)
        )[0]
        if len(sl_below) == 0 or len(sh_above) == 0:
            return None, None

        tp_level = swing_lows[sl_below[-1]]
        sl_structure = swing_highs[sh_above[-1]]

        sl_atr_floor = entry_price + MIN_N_ATR_SL * atr
        sl_atr_ceil = entry_price + MAX_N_ATR_SL * atr

        sl_level = np.clip(sl_structure, sl_atr_floor, sl_atr_ceil)

        if tp_level >= sl_level:
            return None, None

    return tp_level, sl_level


# =========================
# Détection des swings
# =========================

# =========================
# Détection des swings
# =========================

def find_swings(highs, lows, swing_len=SWING_LEN, min_gap=MIN_GAP):
    """
    Deux corrections par rapport à l'original :

    1. Détection STRICTE (> au lieu de >=)
       Une barre plate n'est jamais un swing high/low.
       Évite la génération de swings parasites sur des séries plates
       qui, avec keep-first, auraient bloqué les vrais pics.

    2. Déduplication CAUSALE (keep='first')
       Quand deux swings sont dans la fenêtre min_gap, on garde le PREMIER
       (le plus ancien). Aucun remplacement → aucune information future
       n'est utilisée. Le passé ne dépend plus du futur.
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    if n <= swing_len:
        return swing_highs, swing_lows

    shape = (n - swing_len, swing_len)
    strides = highs.strides + highs.strides
    windows_h = np.lib.stride_tricks.as_strided(highs, shape=shape, strides=strides)
    windows_l = np.lib.stride_tricks.as_strided(lows, shape=shape, strides=strides)

    raw_sh = np.full(n, np.nan)
    raw_sl = np.full(n, np.nan)

    for j in range(swing_len, n):
        w_idx = j - swing_len
        # FIX 1 : > strict — une barre plate n'est pas un swing
        if highs[j] > np.max(windows_h[w_idx]):
            raw_sh[j] = highs[j]
        if lows[j] < np.min(windows_l[w_idx]):
            raw_sl[j] = lows[j]

    sh_indices = np.where(~np.isnan(raw_sh))[0]
    sl_indices = np.where(~np.isnan(raw_sl))[0]

    def deduplicate_causal(indices, gap=min_gap):
        """
        FIX 2 : keep-first causale — premier arrivé = gardé.
        Aucun remplacement → le passé ne dépend jamais du futur.
        """
        if len(indices) == 0:
            return indices
        result = [indices[0]]
        for idx in indices[1:]:
            if idx - result[-1] >= gap:
                result.append(idx)
            # trop proche → IGNORÉ, jamais de remplacement
        return np.array(result)

    for idx in deduplicate_causal(sh_indices):
        swing_highs[idx] = highs[idx]
    for idx in deduplicate_causal(sl_indices):
        swing_lows[idx] = lows[idx]

    return swing_highs, swing_lows


# =========================
# Labeling — v6 : aligné sur le résultat réel du trade
# =========================

def market_structure_label(df, horizon, swing_len, min_swing_atr):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values
    n = len(closes)

    swing_highs, swing_lows = find_swings(highs, lows, swing_len, MIN_GAP)
    raw_labels = np.full(n, np.nan)

    for i in range(swing_len, n - horizon):
        past_sh = np.where(~np.isnan(swing_highs[:i]))[0]
        past_sl = np.where(~np.isnan(swing_lows[:i]))[0]
        if len(past_sh) == 0 or len(past_sl) == 0:
            continue

        last_sh = swing_highs[past_sh[-1]]
        last_sl = swing_lows[past_sl[-1]]
        current_atr = atrs[i]

        if abs(last_sh - closes[i]) < min_swing_atr * current_atr:
            last_sh = np.nan
        if abs(closes[i] - last_sl) < min_swing_atr * current_atr:
            last_sl = np.nan
        if np.isnan(last_sh) and np.isnan(last_sl):
            continue

        buy_bar = sell_bar = np.inf
        for j in range(1, horizon + 1):
            if not np.isnan(last_sh) and highs[i + j] > last_sh and buy_bar == np.inf:
                buy_bar = j
            if not np.isnan(last_sl) and lows[i + j] < last_sl and sell_bar == np.inf:
                sell_bar = j
            if buy_bar < np.inf and sell_bar < np.inf:
                break

        if buy_bar < sell_bar:
            raw_labels[i] = 1
        elif sell_bar < buy_bar:
            raw_labels[i] = -1

    return raw_labels


# =========================
# Features (identique v5)
# =========================

def build_feature_matrix(df, swing_highs, swing_lows):  # ← ajouter les 2 paramètres
    log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    hl_range = ((df['high'] - df['low']) / df['close']).fillna(0).values

    atr_values = df['atr'].values
    assert np.all(atr_values[~np.isnan(atr_values)] > 0)
    atr_norm = atr_values / df['close'].values

    hour = df['datetime'].dt.hour.values
    dow = df['datetime'].dt.dayofweek.values
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    ema_dist = ((df['close'] - df['ema_short']) / df['close']).fillna(0)
    ema_spread = ((df['ema_short'] - df['ema_long']) / df['close']).fillna(0).values
    ema_momentum = ema_dist.diff(5).fillna(0).values
    ema_dist = ema_dist.values

    rsi = df.ta.rsi(close=df['close'], length=ATR_PERIOD).fillna(50).values / 100
    macd = (df['ema_short'] - df['ema_long']) / df['close']
    macd_hist = (macd - macd.ewm(span=9, adjust=False).mean()).fillna(0).values

    # =========================
    # Distances aux swings (nouvelles features)
    # =========================
    closes = df['close'].values
    atrs = df['atr'].values
    n = len(closes)

    dist_to_sh = np.zeros(n)  # distance au swing high le plus proche au-dessus
    dist_to_sl = np.zeros(n)  # distance au swing low le plus proche en-dessous
    n_sh_above = np.zeros(n)  # nombre de swing highs disponibles au-dessus
    n_sl_below = np.zeros(n)  # nombre de swing lows disponibles en-dessous
    rr_estimate = np.zeros(n)  # RR estimé (swing_high - close) / (close - swing_low)

    for i in range(n):
        close = closes[i]
        atr = atrs[i] if atrs[i] > 0 else 1e-9

        sh_vals = swing_highs[:i][
            (~np.isnan(swing_highs[:i])) & (swing_highs[:i] > close)
            ]
        sl_vals = swing_lows[:i][
            (~np.isnan(swing_lows[:i])) & (swing_lows[:i] < close)
            ]

        if len(sh_vals) > 0:
            nearest_sh = sh_vals[-1]  # le plus récent au-dessus
            dist_to_sh[i] = (nearest_sh - close) / atr  # en unités ATR
            n_sh_above[i] = min(len(sh_vals), 10) / 10  # normalisé 0-1

        if len(sl_vals) > 0:
            nearest_sl = sl_vals[-1]  # le plus récent en-dessous
            dist_to_sl[i] = (close - nearest_sl) / atr  # en unités ATR
            n_sl_below[i] = min(len(sl_vals), 10) / 10

        if dist_to_sh[i] > 0 and dist_to_sl[i] > 0:
            rr_estimate[i] = dist_to_sh[i] / dist_to_sl[i]  # RR potentiel BUY

    # Cap pour éviter les outliers (ex: swing très loin)
    dist_to_sh = np.clip(dist_to_sh, 0, 10)
    dist_to_sl = np.clip(dist_to_sl, 0, 10)
    rr_estimate = np.clip(rr_estimate, 0, 5)

    feats = [log_returns, hl_range, hour_sin, hour_cos, dow_sin, dow_cos,
             atr_norm, ema_dist, ema_spread, rsi, ema_momentum, macd_hist,
             dist_to_sh, dist_to_sl, n_sh_above, n_sl_below, rr_estimate]

    scale_mask = [False, True, False, False, False, False, True,
                  False, False, False, False, False,
                  True, True, False, False, True]  # ← 5 nouvelles entrées

    if 'tick_volume' in df.columns:
        feats.insert(1, df['tick_volume'].values)
        scale_mask.insert(1, True)

    return np.column_stack(feats), scale_mask


# =========================
# Séquences & Scaling
# =========================

def create_sequences(data, targets, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len: i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def apply_local_scaling(X_raw, scale_mask):
    X_scaled = np.copy(X_raw)
    for i in range(X_raw.shape[0]):
        for f in range(X_raw.shape[2]):
            if scale_mask[f]:
                seq = X_raw[i, :, f]
                minv, maxv = np.min(seq), np.max(seq)
                X_scaled[i, :, f] = (seq - minv) / (maxv - minv) if maxv - minv > 1e-9 else 0.0
    return X_scaled
