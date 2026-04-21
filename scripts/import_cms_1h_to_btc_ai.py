"""
One-shot import: crypto-market-state → btc_AI.
Converte schema ao importar. Roda UMA vez.
"""
import pandas as pd
from pathlib import Path

CMS_PATH = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet")
BTC_AI_PATH = Path("/Users/brown/Documents/MLGeral/btc_AI/data/01_raw/spot/btc_1h.parquet")


def main():
    print(f"Loading: {CMS_PATH}")
    df = pd.read_parquet(CMS_PATH)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Index: {df.index.name}")

    print("\nConverting schema to btc_AI...")
    df = df.reset_index()
    df = df.rename(columns={
        "trades":                  "num_trades",
        "taker_buy_volume":        "taker_buy_base_vol",
        "taker_buy_quote_volume":  "taker_buy_quote_vol",
    })

    if "quote_volume" in df.columns:
        df = df.drop(columns=["quote_volume"])

    df["source"] = "binance_spot"

    df = df[[
        "timestamp", "open", "high", "low", "close", "volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "source"
    ]]

    df["num_trades"] = df["num_trades"].astype(int)
    for col in ["open", "high", "low", "close", "volume",
                "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    assert df["timestamp"].is_monotonic_increasing, "timestamps não ordenados"
    assert not df["timestamp"].duplicated().any(), "timestamps duplicados"
    assert df["taker_buy_base_vol"].notna().all(), "NaN em taker_buy_base_vol"
    assert df["taker_buy_quote_vol"].notna().all(), "NaN em taker_buy_quote_vol"

    BTC_AI_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BTC_AI_PATH, index=False)

    print(f"\nSaved: {BTC_AI_PATH}")
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Dtypes:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
