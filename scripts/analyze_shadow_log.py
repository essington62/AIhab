"""
Análise semanal do shadow log — valida filtro taker_z out-of-sample.

Rodar semanalmente:
  python scripts/analyze_shadow_log.py

Responde:
  1. Quantos trades o filtro teria bloqueado?
  2. Desses, ratio losers/winners (histórico: 1.43)
  3. 4h vs 1h: qual está capturando melhor?
  4. Decisão: ativar / continuar shadow / rejeitar

Output: prompts/shadow_analysis_report.md
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shadow_analysis")

ROOT = Path(__file__).resolve().parents[1]
SHADOW_LOG = ROOT / "data/08_shadow/taker_z_shadow_log.jsonl"
TRADES_PATH = ROOT / "data/05_output/trades.parquet"
REPORT_PATH = ROOT / "prompts/shadow_analysis_report.md"

HISTORIC_RATIO = 1.43   # losers/winners bloqueados — referência histórica
MIN_SAMPLE_ACTIVATE = 100


def load_shadow_log() -> pd.DataFrame:
    if not SHADOW_LOG.exists():
        raise FileNotFoundError(f"Shadow log não encontrado: {SHADOW_LOG}\nRode o paper_trader por alguns dias.")

    records = []
    with open(SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    return df


def load_trades() -> pd.DataFrame:
    if not TRADES_PATH.exists():
        logger.warning(f"trades.parquet não encontrado: {TRADES_PATH}")
        return pd.DataFrame()
    df = pd.read_parquet(TRADES_PATH)
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df


def filter_stats(df_merged: pd.DataFrame, col: str, threshold: float = -1.0) -> dict:
    """Estatísticas de filtro para trades com outcome conhecido."""
    valid = df_merged.dropna(subset=[col, "return_pct"])
    blocked = valid[valid[col] < threshold]
    kept = valid[valid[col] >= threshold]

    n_blocked = len(blocked)
    n_kept = len(kept)

    if n_blocked == 0:
        return {"n_blocked": 0, "n_losers": 0, "n_winners": 0, "ratio": None,
                "sharpe_kept": None, "n_kept": n_kept}

    n_losers = int((blocked["return_pct"] < 0).sum())
    n_winners = n_blocked - n_losers
    ratio = n_losers / max(n_winners, 1)

    # Sharpe dos kept
    r = kept["return_pct"].values
    sharpe = (r.mean() / r.std()) * (52 ** 0.5) if r.std() > 0 and len(r) > 1 else None

    return {
        "n_blocked": n_blocked,
        "n_losers": n_losers,
        "n_winners": n_winners,
        "ratio": round(ratio, 2),
        "sharpe_kept": round(sharpe, 2) if sharpe else None,
        "n_kept": n_kept,
    }


def main():
    logger.info("=" * 60)
    logger.info("Shadow Filter Analysis")
    logger.info("=" * 60)

    try:
        df_shadow = load_shadow_log()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Shadow log: {len(df_shadow)} entries (status=ok)")
    logger.info(f"Period: {df_shadow['entry_time'].min()} → {df_shadow['entry_time'].max()}")

    df_trades = load_trades()
    logger.info(f"Completed trades: {len(df_trades)}")

    # Merge por trade_id
    df_merged = df_shadow.copy()
    if len(df_trades) > 0 and "trade_id" in df_trades.columns and "trade_id" in df_merged.columns:
        df_merged = df_merged.merge(
            df_trades[["trade_id", "return_pct", "exit_reason"]],
            on="trade_id",
            how="left",
        )
        n_matched = df_merged["return_pct"].notna().sum()
        logger.info(f"Matched with outcome: {n_matched}/{len(df_merged)}")
    else:
        df_merged["return_pct"] = None
        df_merged["exit_reason"] = None

    # Stats
    stats_4h = filter_stats(df_merged, "taker_z_4h")
    stats_1h = filter_stats(df_merged, "taker_z_1h")

    n_total = len(df_shadow)
    n_would_block_4h = int(df_shadow["would_block_4h"].sum())
    n_would_block_1h = int(df_shadow["would_block_1h"].sum())
    n_agree = int((df_shadow["would_block_4h"] == df_shadow["would_block_1h"]).sum())
    n_disagree = n_total - n_agree

    # Decisão
    def decision_text(stats: dict, source: str) -> str:
        if stats["n_blocked"] == 0:
            return f"⏳ Sem bloqueios ainda — continuar acumulando"
        if stats["ratio"] is None:
            return f"⏳ Sem outcomes ainda — continuar acumulando"
        r = stats["ratio"]
        if n_total < 50:
            return f"⏳ ESPERAR — amostra pequena ({n_total} trades < 50)"
        elif n_total < MIN_SAMPLE_ACTIVATE:
            return f"⏳ CONTINUAR SHADOW — {n_total} trades (meta: {MIN_SAMPLE_ACTIVATE})"
        else:
            if r >= 1.3:
                return f"✅ PODE ATIVAR {source} — ratio {r:.2f} ≥ 1.30 (meta histórica {HISTORIC_RATIO})"
            elif r >= 1.0:
                return f"⚠️ MARGINAL {source} — ratio {r:.2f} (histórico {HISTORIC_RATIO})"
            else:
                return f"❌ NÃO ATIVAR {source} — ratio {r:.2f} degradado (histórico {HISTORIC_RATIO})"

    lines = [
        "# Shadow Filter Analysis Report",
        f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Period:** {df_shadow['entry_time'].min().date()} → {df_shadow['entry_time'].max().date()}",
        f"**Total trades observed:** {n_total}",
        f"**Trades com outcome:** {df_merged['return_pct'].notna().sum()}",
        "",
        "## 4h CoinGlass (taker_z < -1.0)\n",
        f"- Shadow bloquearia: {n_would_block_4h} / {n_total} ({n_would_block_4h/n_total*100:.1f}%)",
        f"- Trades com outcome bloqueados: {stats_4h['n_blocked']}",
    ]
    if stats_4h["n_blocked"] > 0:
        lines += [
            f"  - Losers: {stats_4h['n_losers']} | Winners: {stats_4h['n_winners']}",
            f"  - Ratio: {stats_4h['ratio']:.2f} (histórico: {HISTORIC_RATIO})",
        ]
        if stats_4h["sharpe_kept"] is not None:
            lines.append(f"  - Sharpe dos kept: {stats_4h['sharpe_kept']:.2f}")
    lines += [
        f"- **Decisão:** {decision_text(stats_4h, '4h')}",
        "",
        "## 1h Binance nativo (taker_z_1h < -1.0)\n",
        f"- Shadow bloquearia: {n_would_block_1h} / {n_total} ({n_would_block_1h/n_total*100:.1f}%)",
        f"- Trades com outcome bloqueados: {stats_1h['n_blocked']}",
    ]
    if stats_1h["n_blocked"] > 0:
        lines += [
            f"  - Losers: {stats_1h['n_losers']} | Winners: {stats_1h['n_winners']}",
            f"  - Ratio: {stats_1h['ratio']:.2f} (histórico: {HISTORIC_RATIO})",
        ]
        if stats_1h["sharpe_kept"] is not None:
            lines.append(f"  - Sharpe dos kept: {stats_1h['sharpe_kept']:.2f}")
    lines += [
        f"- **Decisão:** {decision_text(stats_1h, '1h')}",
        "",
        "## Agreement 4h vs 1h\n",
        f"- Concordaram: {n_agree} / {n_total} ({n_agree/n_total*100:.1f}%)",
        f"- Divergiram: {n_disagree} / {n_total} ({n_disagree/n_total*100:.1f}%)",
        "",
        "## Critérios para ativação\n",
        f"- ✅ Ativar: ratio ≥ 1.30 com {MIN_SAMPLE_ACTIVATE}+ trades",
        f"- ⚠️ Continuar shadow: ratio 1.0–1.3 ou amostra insuficiente",
        f"- ❌ Rejeitar: ratio < 1.0",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report: {REPORT_PATH}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
