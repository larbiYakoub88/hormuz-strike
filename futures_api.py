"""
Futures Trading API — Interactive Brokers (ib_insync)
======================================================
Équivalent de l'API MT5 fournie, adapté pour les futures CME/CBOT/NYMEX/COMEX.

Démarrage :
    python futures_api.py <server_port> <ib_host> <ib_port> <client_id>

    Exemples de ports IB :
        TWS Paper  : 7497    TWS Live    : 7496
        Gateway Paper : 4002  Gateway Live : 4001

    Exemple :
        python futures_api.py 5001 127.0.0.1 7497 1

Endpoints disponibles :
    POST   /api/positions/open/<symbol>/<exchange>/<expiry>/<BUY|SELL>
    DELETE /api/positions/close/<symbol>/<exchange>/<expiry>
    DELETE /api/positions/close-all/<symbol>/<exchange>/<expiry>
    PUT    /api/positions/stop-loss/<order_id>
    PUT    /api/positions/take-profit/<order_id>
    GET    /api/positions
    GET    /api/positions/<symbol>/<exchange>/<expiry>
    GET    /api/infos/<symbol>/<exchange>/<expiry>
    GET    /api/rates/<symbol>/<exchange>/<expiry>/<timeframe>/current
    GET    /api/rates/<symbol>/<exchange>/<expiry>/<timeframe>/previous
    GET    /api/rates/<symbol>/<exchange>/<expiry>/<timeframe>?count=N
"""

from dataclasses import dataclass, asdict
from ib_insync import IB, Future, MarketOrder, LimitOrder, StopOrder, util
from waitress import serve
from flask import Flask, jsonify, request
import logging
import sys
import threading

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_trading_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

app_logger = logging.getLogger('futures_trading_api')
util.logToConsole(logging.WARNING)   # silence ib_insync verbosity

app = Flask(__name__)

# Objet IB global (thread-safe via ib_insync event loop)
ib = IB()
_ib_lock = threading.Lock()

OPEN_POSITION_RETRY_NUMBER = 8

# Timeframe string → (durée historique, bar size IB)
TIMEFRAME_MAP = {
    "1":    ("2 D",  "1 min"),
    "5":    ("5 D",  "5 mins"),
    "15":   ("5 D",  "15 mins"),
    "30":   ("10 D", "30 mins"),
    "60":   ("20 D", "1 hour"),
    "240":  ("30 D", "4 hours"),
    "1440": ("1 M",  "1 day"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    order_successfully_placed: bool
    errorCode: int
    errorReason: str
    orderId: int = None


@dataclass
class Position:
    symbol: str
    exchange: str
    expiry: str
    side: str           # LONG / SHORT
    quantity: float
    avgCost: float
    unrealizedPNL: float
    realizedPNL: float
    account: str
    conId: int


@dataclass
class Bar:
    end_time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class ClosePositionResult:
    closed: bool
    partially: bool


# ─────────────────────────────────────────────────────────────────────────────
# CONNEXION IB
# ─────────────────────────────────────────────────────────────────────────────

def connect_ib(host: str, port: int, client_id: int):
    app_logger.info(f"Connexion à IB TWS/Gateway : host={host} port={port} clientId={client_id}")
    ib.connect(host, port, clientId=client_id, readonly=False, timeout=20)
    account = ib.wrapper.accounts[0] if ib.wrapper.accounts else "N/A"
    app_logger.info(f"Connecté. Compte actif : {account}")
    account_info = ib.accountSummary()
    for item in account_info:
        if item.tag in ("NetLiquidation", "AvailableFunds", "TotalCashValue"):
            app_logger.info(f"  {item.tag} = {item.value} {item.currency}")


def get_contract(symbol: str, exchange: str, expiry: str, currency: str = "USD") -> Future:
    """
    Résout un contrat Future sur IB.
    expiry : format YYYYMM (ex: 202409) ou YYYYMMDD
    """
    app_logger.debug(f"Résolution contrat : {symbol} {exchange} {expiry}")
    contract = Future(
        symbol=symbol,
        exchange=exchange,
        lastTradeDateOrContractMonth=expiry,
        currency=currency
    )
    contracts = ib.qualifyContracts(contract)
    if not contracts:
        raise ValueError(f"Contrat introuvable : {symbol} {exchange} {expiry}")
    app_logger.debug(f"Contrat résolu : conId={contracts[0].conId}")
    return contracts[0]


# ─────────────────────────────────────────────────────────────────────────────
# OPEN POSITION
# ─────────────────────────────────────────────────────────────────────────────

def open_position_with_retry(symbol, exchange, expiry, order_type, size,
                              limit_price=None, stop_loss=None, take_profit=None):
    app_logger.info(
        f"Ouverture avec retry : {order_type} {size}x {symbol} {exchange} {expiry} "
        f"limit={limit_price} sl={stop_loss} tp={take_profit}"
    )
    result = None
    for attempt in range(1, OPEN_POSITION_RETRY_NUMBER + 1):
        app_logger.info(f"Tentative {attempt}/{OPEN_POSITION_RETRY_NUMBER}")
        result = open_position(symbol, exchange, expiry, order_type, size,
                               limit_price, stop_loss, take_profit)
        if result.order_successfully_placed:
            app_logger.info(f"Position ouverte à la tentative {attempt}")
            return result
    app_logger.error(
        f"Échec après {OPEN_POSITION_RETRY_NUMBER} tentatives. "
        f"Raison : {result.errorReason if result else 'inconnue'}"
    )
    return result


def open_position(symbol, exchange, expiry, order_type, size,
                  limit_price=None, stop_loss=None, take_profit=None):
    app_logger.info(f"Envoi ordre : {order_type} {size}x {symbol} {exchange} {expiry}")
    try:
        with _ib_lock:
            contract = get_contract(symbol, exchange, expiry)
            action = order_type.upper()  # "BUY" ou "SELL"

            # Ordre principal : Market ou Limit
            if limit_price is not None:
                order = LimitOrder(action=action, totalQuantity=size, lmtPrice=limit_price)
                app_logger.debug(f"Ordre Limit à {limit_price}")
            else:
                order = MarketOrder(action=action, totalQuantity=size)
                app_logger.debug("Ordre Market")

            trade = ib.placeOrder(contract, order)
            ib.sleep(1.5)   # laisse le temps au fill

            status = trade.orderStatus.status
            app_logger.info(f"Statut ordre : {status} (orderId={trade.order.orderId})")

            if status not in ("Filled", "PreSubmitted", "Submitted"):
                return OrderResult(
                    order_successfully_placed=False,
                    errorCode=-1,
                    errorReason=f"Statut inattendu : {status}"
                )

            order_id = trade.order.orderId

            # Bracket SL
            if stop_loss is not None:
                sl_action = "SELL" if action == "BUY" else "BUY"
                sl_order = StopOrder(action=sl_action, totalQuantity=size, stopPrice=stop_loss)
                sl_order.parentId = order_id
                ib.placeOrder(contract, sl_order)
                app_logger.info(f"Stop-loss placé à {stop_loss}")

            # Bracket TP
            if take_profit is not None:
                tp_action = "SELL" if action == "BUY" else "BUY"
                tp_order = LimitOrder(action=tp_action, totalQuantity=size, lmtPrice=take_profit)
                tp_order.parentId = order_id
                ib.placeOrder(contract, tp_order)
                app_logger.info(f"Take-profit placé à {take_profit}")

            return OrderResult(
                order_successfully_placed=True,
                errorCode=0,
                errorReason=None,
                orderId=order_id
            )

    except Exception as e:
        app_logger.exception("Exception lors de l'ouverture de la position")
        return OrderResult(order_successfully_placed=False, errorCode=-99, errorReason=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CLOSE POSITION
# ─────────────────────────────────────────────────────────────────────────────

def close_position(symbol, exchange, expiry):
    """Ferme UNE position (par contrat exact)."""
    app_logger.info(f"Fermeture position : {symbol} {exchange} {expiry}")
    try:
        with _ib_lock:
            positions = ib.positions()
            for pos in positions:
                c = pos.contract
                if (c.symbol == symbol
                        and c.exchange == exchange
                        and c.lastTradeDateOrContractMonth.startswith(expiry)):
                    side = "SELL" if pos.position > 0 else "BUY"
                    qty = abs(pos.position)
                    order = MarketOrder(action=side, totalQuantity=qty)
                    trade = ib.placeOrder(c, order)
                    ib.sleep(1.5)
                    status = trade.orderStatus.status
                    app_logger.info(f"Fermeture envoyée : {status}")
                    return ClosePositionResult(closed=True, partially=False)

            app_logger.warning(f"Aucune position trouvée pour {symbol} {exchange} {expiry}")
            return ClosePositionResult(closed=False, partially=False)
    except Exception as e:
        app_logger.exception("Exception lors de la fermeture")
        return ClosePositionResult(closed=False, partially=False)


def close_all_positions(symbol, exchange, expiry):
    """Ferme TOUTES les positions sur un contrat donné."""
    app_logger.info(f"Fermeture toutes positions : {symbol} {exchange} {expiry}")
    return close_position(symbol, exchange, expiry)


# ─────────────────────────────────────────────────────────────────────────────
# MODIFY SL / TP
# ─────────────────────────────────────────────────────────────────────────────

def _find_trade(order_id: int):
    for trade in ib.trades():
        if trade.order.orderId == order_id:
            return trade
    return None


def modify_stop_loss(symbol, exchange, expiry, order_id: int, stop_loss: float):
    app_logger.info(f"Modification SL : orderId={order_id} → {stop_loss}")
    try:
        with _ib_lock:
            trade = _find_trade(order_id)
            if not trade:
                app_logger.error(f"Ordre {order_id} introuvable")
                return OrderResult(
                    order_successfully_placed=False,
                    errorCode=-2,
                    errorReason="Ordre introuvable"
                )
            # Mise à jour du prix stop sur un ordre StopOrder existant
            trade.order.auxPrice = stop_loss
            ib.placeOrder(trade.contract, trade.order)
            ib.sleep(0.5)
            app_logger.info(f"SL modifié pour orderId={order_id}")
            return OrderResult(order_successfully_placed=True, errorCode=0, errorReason=None, orderId=order_id)
    except Exception as e:
        app_logger.exception("Exception lors de la modification du SL")
        return OrderResult(order_successfully_placed=False, errorCode=-99, errorReason=str(e))


def modify_take_profit(symbol, exchange, expiry, order_id: int, take_profit: float):
    app_logger.info(f"Modification TP : orderId={order_id} → {take_profit}")
    try:
        with _ib_lock:
            trade = _find_trade(order_id)
            if not trade:
                app_logger.error(f"Ordre {order_id} introuvable")
                return OrderResult(
                    order_successfully_placed=False,
                    errorCode=-2,
                    errorReason="Ordre introuvable"
                )
            trade.order.lmtPrice = take_profit
            ib.placeOrder(trade.contract, trade.order)
            ib.sleep(0.5)
            app_logger.info(f"TP modifié pour orderId={order_id}")
            return OrderResult(order_successfully_placed=True, errorCode=0, errorReason=None, orderId=order_id)
    except Exception as e:
        app_logger.exception("Exception lors de la modification du TP")
        return OrderResult(order_successfully_placed=False, errorCode=-99, errorReason=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# GET POSITIONS
# ─────────────────────────────────────────────────────────────────────────────

def _to_position(pos) -> Position:
    c = pos.contract
    return Position(
        symbol=c.symbol,
        exchange=c.exchange,
        expiry=c.lastTradeDateOrContractMonth,
        side="LONG" if pos.position > 0 else "SHORT",
        quantity=abs(pos.position),
        avgCost=pos.avgCost,
        unrealizedPNL=pos.unrealizedPnL if hasattr(pos, 'unrealizedPnL') else None,
        realizedPNL=pos.realizedPnL if hasattr(pos, 'realizedPnL') else None,
        account=pos.account,
        conId=c.conId
    )


def get_positions(symbol=None, exchange=None, expiry=None):
    app_logger.debug(f"Récupération positions : {symbol} {exchange} {expiry}")
    positions = ib.positions()
    result = []
    for pos in positions:
        c = pos.contract
        if c.secType != "FUT":
            continue
        if symbol and c.symbol != symbol:
            continue
        if exchange and c.exchange != exchange:
            continue
        if expiry and not c.lastTradeDateOrContractMonth.startswith(expiry):
            continue
        result.append(_to_position(pos))
    app_logger.debug(f"{len(result)} position(s) trouvée(s)")
    return result


def get_all_positions():
    return get_positions()


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL INFO
# ─────────────────────────────────────────────────────────────────────────────

def get_symbol_info(symbol, exchange, expiry):
    app_logger.debug(f"Info contrat : {symbol} {exchange} {expiry}")
    try:
        contract = get_contract(symbol, exchange, expiry)
        details_list = ib.reqContractDetails(contract)
        if not details_list:
            app_logger.error("Aucun détail de contrat retourné")
            return None
        d = details_list[0]
        info = {
            "symbol":            d.contract.symbol,
            "exchange":          d.contract.exchange,
            "expiry":            d.contract.lastTradeDateOrContractMonth,
            "currency":          d.contract.currency,
            "conId":             d.contract.conId,
            "multiplier":        d.contract.multiplier,
            "min_tick":          d.minTick,
            "tick_value":        float(d.contract.multiplier or 1) * d.minTick,
            "volume_min":        1,
            "long_name":         d.longName,
            "trading_hours":     d.tradingHours,
            "liquid_hours":      d.liquidHours,
            "time_zone":         d.timeZoneId,
        }
        app_logger.debug(f"Info contrat : {info}")
        return info
    except Exception as e:
        app_logger.exception("Exception lors de la récupération des infos")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RATES / DONNÉES HISTORIQUES
# ─────────────────────────────────────────────────────────────────────────────

def _convert_to_bar(bar) -> Bar:
    return Bar(
        end_time=str(bar.date),
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=int(bar.volume)
    )


def _get_bars(symbol, exchange, expiry, time_frame, count):
    duration, bar_size = TIMEFRAME_MAP.get(str(time_frame), ("2 D", "1 min"))
    # Augmenter la durée si count est grand
    if count > 500:
        duration = "1 M"
    contract = get_contract(symbol, exchange, expiry)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    return bars


def get_last_rate(symbol, exchange, expiry, time_frame):
    app_logger.debug(f"Dernière barre : {symbol} {exchange} {expiry} TF={time_frame}")
    try:
        bars = _get_bars(symbol, exchange, expiry, time_frame, 1)
        if not bars:
            return None
        return _convert_to_bar(bars[-1])
    except Exception as e:
        app_logger.exception("Erreur get_last_rate")
        return None


def get_previous_rate(symbol, exchange, expiry, time_frame):
    app_logger.debug(f"Barre précédente : {symbol} {exchange} {expiry} TF={time_frame}")
    try:
        bars = _get_bars(symbol, exchange, expiry, time_frame, 2)
        if not bars or len(bars) < 2:
            return None
        return _convert_to_bar(bars[-2])
    except Exception as e:
        app_logger.exception("Erreur get_previous_rate")
        return None


def get_rates_list(symbol, exchange, expiry, time_frame, count):
    app_logger.debug(f"Liste barres : {symbol} {exchange} {expiry} TF={time_frame} count={count}")
    try:
        bars = _get_bars(symbol, exchange, expiry, time_frame, count)
        return [_convert_to_bar(b) for b in bars[-count:]]
    except Exception as e:
        app_logger.exception("Erreur get_rates_list")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_request_as_float(query_name, default=None):
    value = request.args.get(query_name)
    if value:
        try:
            return float(value)
        except ValueError:
            app_logger.warning(f"Paramètre invalide : {query_name}={value}, défaut={default}")
            return default
    return default


def get_request_as_int(query_name, default=None):
    value = request.args.get(query_name)
    if value:
        try:
            return int(value)
        except ValueError:
            return default
    return default


# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def log_response_info(response):
    return response


# --- Positions ---

@app.route("/api/positions/open/<symbol>/<exchange>/<expiry>/<order_type>", methods=['POST'])
def position_open(symbol, exchange, expiry, order_type):
    """
    Ouvre une position.
    Params query :
        size        (float, défaut=1)
        limit_price (float, optionnel) → ordre limit, sinon market
        stop_loss   (float, optionnel)
        take_profit (float, optionnel)
    """
    app_logger.info(f"API: Open {order_type} {symbol} {exchange} {expiry}")
    size        = get_request_as_float('size', 1)
    limit_price = get_request_as_float('limit_price', None)
    stop_loss   = get_request_as_float('stop_loss', None)
    take_profit = get_request_as_float('take_profit', None)
    result = open_position_with_retry(
        symbol, exchange, expiry, order_type.upper(),
        size, limit_price, stop_loss, take_profit
    )
    return jsonify(asdict(result))


@app.route("/api/positions/close/<symbol>/<exchange>/<expiry>", methods=['DELETE'])
def position_close(symbol, exchange, expiry):
    """Ferme la position sur le contrat donné."""
    app_logger.info(f"API: Close {symbol} {exchange} {expiry}")
    result = close_position(symbol, exchange, expiry)
    return jsonify(asdict(result))


@app.route("/api/positions/close-all/<symbol>/<exchange>/<expiry>", methods=['DELETE'])
def positions_close_all(symbol, exchange, expiry):
    """Ferme toutes les positions sur le contrat donné."""
    app_logger.info(f"API: Close all {symbol} {exchange} {expiry}")
    result = close_all_positions(symbol, exchange, expiry)
    return jsonify(asdict(result))


@app.route("/api/positions/stop-loss/<symbol>/<exchange>/<expiry>/<int:order_id>", methods=['PUT'])
def update_stop_loss(symbol, exchange, expiry, order_id):
    """
    Modifie le stop-loss d'un ordre existant.
    Param query : stop_loss (float)
    """
    app_logger.info(f"API: Update SL orderId={order_id}")
    stop_loss = get_request_as_float('stop_loss', None)
    result = modify_stop_loss(symbol, exchange, expiry, order_id, stop_loss)
    return jsonify(asdict(result))


@app.route("/api/positions/take-profit/<symbol>/<exchange>/<expiry>/<int:order_id>", methods=['PUT'])
def update_take_profit(symbol, exchange, expiry, order_id):
    """
    Modifie le take-profit d'un ordre existant.
    Param query : take_profit (float)
    """
    app_logger.info(f"API: Update TP orderId={order_id}")
    take_profit = get_request_as_float('take_profit', None)
    result = modify_take_profit(symbol, exchange, expiry, order_id, take_profit)
    return jsonify(asdict(result))


@app.route("/api/positions", methods=['GET'])
def positions_all_get():
    """Retourne toutes les positions futures ouvertes."""
    app_logger.info("API: Get all positions")
    return jsonify([asdict(p) for p in get_all_positions()])


@app.route("/api/positions/<symbol>/<exchange>/<expiry>", methods=['GET'])
def positions_get(symbol, exchange, expiry):
    """Retourne les positions pour un contrat spécifique."""
    app_logger.info(f"API: Get positions {symbol} {exchange} {expiry}")
    return jsonify([asdict(p) for p in get_positions(symbol, exchange, expiry)])


# --- Infos ---

@app.route("/api/infos/<symbol>/<exchange>/<expiry>", methods=['GET'])
def symbol_infos_get(symbol, exchange, expiry):
    """Retourne les caractéristiques du contrat (tick, multiplier, etc.)."""
    app_logger.info(f"API: Get infos {symbol} {exchange} {expiry}")
    return jsonify(get_symbol_info(symbol, exchange, expiry))


# --- Rates ---

@app.route("/api/rates/<symbol>/<exchange>/<expiry>/<time_frame>/current", methods=['GET'])
def current_rate_get(symbol, exchange, expiry, time_frame):
    """Retourne la dernière barre fermée."""
    bar = get_last_rate(symbol, exchange, expiry, time_frame)
    return jsonify(asdict(bar) if bar else None)


@app.route("/api/rates/<symbol>/<exchange>/<expiry>/<time_frame>/previous", methods=['GET'])
def previous_rate_get(symbol, exchange, expiry, time_frame):
    """Retourne l'avant-dernière barre fermée."""
    bar = get_previous_rate(symbol, exchange, expiry, time_frame)
    return jsonify(asdict(bar) if bar else None)


@app.route("/api/rates/<symbol>/<exchange>/<expiry>/<time_frame>", methods=['GET'])
def get_rates(symbol, exchange, expiry, time_frame):
    """
    Retourne N barres historiques.
    Param query : count (int, défaut=1)
    """
    count = get_request_as_int('count', 1)
    return jsonify([asdict(b) for b in get_rates_list(symbol, exchange, expiry, time_frame, count)])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    argv = sys.argv

    if len(argv) != 5:
        print(
            "Usage : python futures_api.py <server_port> <ib_host> <ib_port> <client_id>\n"
            "  server_port : port de cette API REST (ex: 5001)\n"
            "  ib_host     : IP de TWS/Gateway (ex: 127.0.0.1)\n"
            "  ib_port     : 7497 (TWS paper) | 7496 (TWS live) | 4002 (GW paper) | 4001 (GW live)\n"
            "  client_id   : identifiant client IB (ex: 1)\n\n"
            f"Reçu : {len(argv)} argument(s)"
        )
        sys.exit(1)

    server_port = int(argv[1])
    ib_host     = argv[2]
    ib_port     = int(argv[3])
    client_id   = int(argv[4])

    app_logger.info(f"Démarrage Futures API sur le port {server_port}")
    app_logger.info(f"Connexion IB : {ib_host}:{ib_port} (clientId={client_id})")

    try:
        connect_ib(ib_host, ib_port, client_id)
        app_logger.info("Connexion IB établie — démarrage du serveur HTTP")
        serve(app, host='0.0.0.0', port=server_port, threads=50)
    except Exception as e:
        app_logger.critical(f"Échec du démarrage : {e}")
        sys.exit(1)
