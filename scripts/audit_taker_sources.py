"""
Auditoria Rápida — Fontes de Taker Data.

Verifica em 2 minutos:
  1. Quais arquivos parquet têm colunas taker
  2. Quais timeframes (1h, 4h) estão disponíveis
  3. Qual é a fonte (Binance? CoinGlass?)
  4. O pipeline usa qual?
"""
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_taker")


def scan_parquet_files():
    parquets = list(ROOT.rglob("*.parquet"))
    logger.info(f"Scanning {len(parquets)} parquet files...")

    taker_files = []
    for p in parquets:
        try:
            df = pd.read_parquet(p)
            taker_cols = [c for c in df.columns if "taker" in c.lower()]
            if not taker_cols:
                continue

            rel = p.relative_to(ROOT)
            name_lower = p.name.lower()
            path_lower = str(rel).lower()

            tf = ("1h" if "1h" in name_lower or "/1h" in path_lower
                  else "4h" if "4h" in name_lower or "/4h" in path_lower
                  else "1d" if "1d" in name_lower else "unknown")

            sources = []
            if "binance" in path_lower: sources.append("Binance")
            if "coinglass" in path_lower: sources.append("CoinGlass")
            if "spot" in path_lower: sources.append("spot")
            if "futures" in path_lower: sources.append("futures")

            info = {
                "path": str(rel),
                "timeframe": tf,
                "taker_columns": taker_cols,
                "rows": len(df),
                "source_hints": sources,
            }

            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"])
                info["date_range"] = f"{ts.min().date()} → {ts.max().date()} ({(ts.max()-ts.min()).days}d)"

            taker_files.append(info)
        except Exception:
            pass

    return taker_files


def search_code_references():
    results = []
    for base in [ROOT / "src", ROOT / "scripts", ROOT / "conf"]:
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            rel = py_file.relative_to(ROOT)
            for ln, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or "taker" not in stripped.lower():
                    continue
                results.append({"file": str(rel), "line": ln, "content": stripped[:110]})
    return results


def check_binance_1h():
    candidates = [
        ROOT / "data/01_raw/spot/btc_1h.parquet",
        ROOT / "data/01_raw/futures/btc_1h.parquet",
        ROOT / "data/01_raw/binance/btc_1h.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        tc = [c for c in df.columns if "taker" in c.lower()]
        print(f"\n  {p.relative_to(ROOT)}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Taker cols: {tc or 'NONE'}")
        if tc:
            return p, tc
    return None, []


def main():
    print("=" * 60)
    print("Auditoria Rápida — Fontes de Taker Data")
    print("=" * 60)

    taker_files = scan_parquet_files()
    print(f"\n[PARQUETS COM TAKER] {len(taker_files)} arquivos encontrados:\n")
    for info in taker_files:
        print(f"  📁 {info['path']}")
        print(f"     TF={info['timeframe']} | rows={info['rows']} | "
              f"cols={info['taker_columns']} | source={info['source_hints']}")
        if "date_range" in info:
            print(f"     Range: {info['date_range']}")

    refs = search_code_references()
    print(f"\n[CODE REFS] {len(refs)} linhas com 'taker':")
    for r in refs[:20]:
        print(f"  {r['file']}:L{r['line']}  {r['content']}")
    if len(refs) > 20:
        print(f"  ... e mais {len(refs)-20}")

    print("\n[BINANCE 1H CHECK]")
    b1h_path, b1h_cols = check_binance_1h()

    has_b1h = b1h_path is not None
    has_4h = any("4h" in f["timeframe"] for f in taker_files)

    print("\n" + "=" * 60)
    print("VEREDICTO")
    print("=" * 60)
    if has_b1h:
        print(f"\n✅ BINANCE 1H NATIVO: {b1h_path.relative_to(ROOT)}")
        print(f"   Colunas: {b1h_cols}")
        print("\n   → Podemos usar taker 1h com shift(1) — sem look-ahead")
        print("   → Lag_1h do estudo (Sharpe 4.71) pode ser sinal REAL")
        print("   → Próximo passo: refazer filter_validation com 1h shift(1)")
    elif has_4h:
        print("\n⚠️ SÓ COINGLASS 4H:")
        print("   → Usar prev_4h (já validado, Sharpe 3.61)")
        print("   → Lag_1h do estudo foi artefato de resample")
        print("   → Próximo passo: implementar prev_4h em shadow mode")
    else:
        print("\n❓ Sem fonte clara de taker — inspecionar manualmente")


if __name__ == "__main__":
    main()
