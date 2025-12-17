import pandas as pd
import requests
from datetime import datetime, timezone

class RealTimeIngestor:
    def __init__(self):
        self.url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

    def fetch_data(self):
        print(f"[{datetime.now()}] Connecting to SWPC NOAA...")
        try:
            response = requests.get(self.url, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"ERROR: {e}")
            return None

        df = pd.DataFrame(data)

        df = df[df['energy'] == '0.1-0.8nm'].copy()

        df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
        df = df.set_index('time_tag').sort_index()

        df = df[['flux']].rename(columns={'flux': 'xl'})

        df = df[~df.index.duplicated(keep='last')]
        full_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')
        df = df.reindex(full_idx)

        df['xl'] = df['xl'].interpolate(method='linear', limit=10)
        df['xl'] = df['xl'].ffill().fillna(0)

        last_time = df.index[-1]
        latency = datetime.now(timezone.utc) - last_time
        print(f"Dados coletados até: {last_time} (Latência: {latency.total_seconds()/60:.1f} min)")

        print(df)
        return df