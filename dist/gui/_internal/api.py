import os
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades

load_dotenv()

class OandaAPI:
    def __init__(self):
        self.access_token = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.client = API(access_token=self.access_token)

    def execute_trade(self, instrument, side, units=1000):
        data = {
            "order": {
                "instrument": instrument,
                "units": str(units if side == "BUY" else -units),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        r = orders.OrderCreate(accountID=self.account_id, data=data)
        try:
            self.client.request(r)
            return True
        except Exception as e:
            print(f"Trade failed: {e}")
            return False

    def get_open_position(self, instrument):
        r = positions.OpenPositions(accountID=self.account_id)
        try:
            result = self.client.request(r)
            for pos in result.get("positions", []):
                if pos["instrument"] == instrument:
                    net = pos["long"]["units"] if float(pos["long"]["units"]) != 0 else pos["short"]["units"]
                    side = "BUY" if float(net) > 0 else "SELL"
                    return {"side": side, "units": abs(int(float(net)))}
            return None
        except Exception as e:
            print(f"Error getting position: {e}")
            return None

    def get_unrealized_pnl(self, instrument):
        r = positions.PositionDetails(accountID=self.account_id, instrument=instrument)
        try:
            result = self.client.request(r)
            return float(result["position"]["unrealizedPL"])
        except:
            return 0.0