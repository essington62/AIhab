"""
Auditoria SOL Bot 4 — Verifica conformidade com specs críticas.

Rodar: python scripts/audit_sol_bot4.py
Output: AUDIT_SOL_BOT4.md com findings
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL_BOT_PATH = ROOT / "src" / "trading" / "sol_bot4.py"
AUDIT_REPORT = ROOT / "AUDIT_SOL_BOT4.md"

issues = []


def check(condition, msg, severity="WARN"):
    status = "✅" if condition else "❌"
    issues.append({
        "status": status,
        "severity": severity if not condition else "OK",
        "message": msg,
    })
    print(f"{status} [{severity if not condition else 'OK':8s}] {msg}")


def extract_function(source, func_name):
    """Extract function body — handles multi-line bodies."""
    pattern = rf"def {func_name}\s*\([^)]*\).*?(?=\ndef |\Z)"
    match = re.search(pattern, source, re.DOTALL)
    return match.group(0) if match else None


def check_pd_to_datetime_utc(source):
    """Finds pd.to_datetime calls and checks if all have utc=True."""
    # Extract all pd.to_datetime(...) calls — handles nested parens
    calls = []
    for m in re.finditer(r"pd\.to_datetime\(", source):
        start = m.start()
        depth = 0
        i = m.start() + len("pd.to_datetime(") - 1
        while i < len(source):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(source[start:i + 1])
                    break
            i += 1
    without_utc = [c for c in calls if "utc=True" not in c]
    return calls, without_utc


def main():
    print("=" * 65)
    print("SOL Bot 4 — Auditoria de Conformidade (8 pontos)")
    print("=" * 65)

    if not SOL_BOT_PATH.exists():
        print(f"❌ CRÍTICO: {SOL_BOT_PATH} não existe!")
        sys.exit(1)

    source = SOL_BOT_PATH.read_text()

    # ========== PONTO 1: Anti look-ahead ==========
    print("\n─── PONTO 1: Anti look-ahead ─────────────────────────────────")

    entry_fn = extract_function(source, "check_entry_signal")
    compute_fn = extract_function(source, "compute_sol_features")

    # Check: features.get("taker_z_prev") used in check_entry_signal
    taker_prev_fetched = bool(entry_fn and 'get("taker_z_prev")' in entry_fn)
    taker_raw_fetched = bool(entry_fn and re.search(r'get\(["\']taker_z["\'](?!\w)', entry_fn))
    check(taker_prev_fetched and not taker_raw_fetched,
          "check_entry_signal busca 'taker_z_prev' (não taker_z raw) em features dict", "CRITICAL")

    eth_prev_fetched = bool(entry_fn and 'get("eth_ret_1h_prev")' in entry_fn)
    check(eth_prev_fetched, "check_entry_signal busca 'eth_ret_1h_prev' em features", "CRITICAL")

    oi_prev_fetched = bool(entry_fn and ("oi_z_24h_max_prev" in entry_fn or "oi_z_24h_block" in entry_fn))
    check(oi_prev_fetched, "check_entry_signal usa oi_z_24h_max_prev ou gate de bloco", "HIGH")

    shift_applied = bool(compute_fn and "shift(1)" in compute_fn)
    check(shift_applied, "compute_sol_features aplica shift(1) nas features preditoras", "CRITICAL")

    # ========== PONTO 2: Capital $10k separado ==========
    print("\n─── PONTO 2: Capital $10k separado de BTC/ETH ────────────────")

    uses_sol_portfolio = "portfolio_sol.json" in source
    check(uses_sol_portfolio, "Usa portfolio_sol.json (arquivo SOL-específico)", "CRITICAL")

    no_btc_state = "portfolio_state.json" not in source
    check(no_btc_state, "Não referencia portfolio_state.json (BTC portfolio)", "CRITICAL")

    has_10k = "10000" in source
    check(has_10k, "Capital inicial $10,000 presente no código", "HIGH")

    check("04_scoring/portfolio_sol" in source,
          "Portfolio path em data/04_scoring/ (isolado do BTC 05_output)", "MEDIUM")

    # ========== PONTO 3: Shadow scoring ==========
    print("\n─── PONTO 3: Shadow scoring logging ──────────────────────────")

    has_func = "def log_shadow_scoring" in source
    check(has_func, "Função log_shadow_scoring definida", "HIGH")

    has_path = "sol_scoring_shadow_log.jsonl" in source
    check(has_path, "Path sol_scoring_shadow_log.jsonl correto", "HIGH")

    # Chamada efetiva fora da definição
    call_matches = list(re.finditer(r"log_shadow_scoring\s*\(", source))
    check(len(call_matches) >= 2, "log_shadow_scoring chamada >= 2x (def + pelo menos 1 call)", "CRITICAL")

    check("score_total" in source and "scoring_would_enter" in source and "breakdown" in source,
          "Campos score_total, scoring_would_enter, breakdown no payload do log", "HIGH")

    # Chamada no ciclo (fora da def)
    if len(call_matches) >= 2:
        # Find call outside function definition
        def_pos = source.find("def log_shadow_scoring")
        calls_outside = [m for m in call_matches if m.start() > def_pos + 200]
        check(len(calls_outside) >= 1, "log_shadow_scoring chamada no ciclo principal (fora da def)", "CRITICAL")
    else:
        check(False, "log_shadow_scoring chamada no ciclo principal", "CRITICAL")

    # ========== PONTO 4: Early exit OI 24h ==========
    print("\n─── PONTO 4: Early exit OI 24h ───────────────────────────────")

    check("OI_EARLY_EXIT" in source, "Reason 'OI_EARLY_EXIT' implementado", "HIGH")
    check("def check_oi_early_exit" in source, "Função check_oi_early_exit separada", "MEDIUM")
    check("hours_held" in source and "12" in source, "Verifica hours_held >= 12", "MEDIUM")

    oi_threshold_ok = bool(
        re.search(r"oi_z_threshold.*2\.0|2\.0.*oi_z_threshold|get\(.*oi_z_threshold.*2\.0\)", source)
        or ("oi_z_threshold" in source and "2.0" in source)
    )
    check(oi_threshold_ok, "Threshold oi_z_threshold ≥ 2.0 configurável", "HIGH")

    oi_exit_called = bool(re.search(r"check_oi_early_exit\s*\(", source))
    check(oi_exit_called, "check_oi_early_exit chamada em ciclo de stops", "HIGH")

    # ========== PONTO 5: Cooldown 4h ==========
    print("\n─── PONTO 5: Cooldown 4h pós-exit ────────────────────────────")

    check("cooldown_until" in source, "cooldown_until armazenado no portfolio state", "HIGH")
    check("cooldown_hours" in source, "cooldown_hours lido de params", "MEDIUM")

    # cooldown_until deve ser salvo em execute_exit
    exit_fn = extract_function(source, "execute_exit")
    check(bool(exit_fn and "cooldown_until" in exit_fn),
          "execute_exit persiste cooldown_until após fechar posição", "HIGH")

    # cooldown deve ser checado na entrada
    check(bool(entry_fn and "cooldown_until" in entry_fn),
          "check_entry_signal respeita cooldown_until antes de entrar", "HIGH")

    # ========== PONTO 6: Logs separados ==========
    print("\n─── PONTO 6: Logs separados ───────────────────────────────────")

    check('getLogger("sol_bot4")' in source,
          "Logger nomeado 'sol_bot4' (namespace isolado)", "MEDIUM")

    sol_cycle_path = ROOT / "scripts" / "sol_hourly_cycle.py"
    if sol_cycle_path.exists():
        cyc = sol_cycle_path.read_text()
        check("%(asctime)s" in cyc and "sol_bot4" in cyc,
              "sol_hourly_cycle.py configura logging com format e importa sol_bot4", "LOW")
    else:
        check(False, "scripts/sol_hourly_cycle.py existe", "HIGH")

    # ========== PONTO 7: Timezones UTC ==========
    print("\n─── PONTO 7: Timezones UTC ────────────────────────────────────")

    all_calls, without_utc = check_pd_to_datetime_utc(source)
    check(len(without_utc) == 0,
          f"pd.to_datetime: {len(all_calls) - len(without_utc)}/{len(all_calls)} com utc=True", "HIGH")
    if without_utc:
        for c in without_utc:
            print(f"       missing utc=True: {c[:80]}...")

    dt_now_calls = re.findall(r"datetime\.now\([^)]*\)", source)
    utc_now = [c for c in dt_now_calls if "timezone.utc" in c]
    check(len(dt_now_calls) == len(utc_now),
          f"datetime.now: {len(utc_now)}/{len(dt_now_calls)} com timezone.utc", "HIGH")

    check(bool(re.search(r'Timestamp\.now\(["\']UTC["\']\)', source)),
          "pd.Timestamp.now('UTC') usado em comparações de stops", "MEDIUM")

    # ========== PONTO 8: MultiAssetManager ==========
    print("\n─── PONTO 8: MultiAssetManager integração ─────────────────────")

    cap_yml = ROOT / "conf" / "capital_manager.yml"
    if cap_yml.exists():
        cap_src = cap_yml.read_text()

        check("sol:" in cap_src and "sol_bot4" in cap_src,
              "Bucket 'sol' com bots_allowed=[sol_bot4] em capital_manager.yml", "HIGH")

        # Bucket names in YAML are btc/eth/sol; bots_allowed use bot_ prefix
        check("bot_1_reversal" in cap_src and "bot_2_momentum" in cap_src,
              "bots_allowed BTC preservados (bot_1_reversal, bot_2_momentum)", "CRITICAL")
        check("bot_3_volume" in cap_src,
              "bots_allowed ETH preservado (bot_3_volume)", "CRITICAL")

        check("initial_capital_usd: 10000" in cap_src or "initial_capital_usd: 10_000" in cap_src
              or ('"sol"' in cap_src or "sol:" in cap_src) and "10000" in cap_src,
              "capital_usd: 10000 no bucket sol de capital_manager.yml", "HIGH")
    else:
        check(False, "conf/capital_manager.yml existe", "CRITICAL")

    mam_path = ROOT / "src" / "trading" / "multi_asset_manager.py"
    if mam_path.exists():
        mam_src = mam_path.read_text()
        # MAM lê dinamicamente do YAML — verificar que lê capital_manager.yml
        check("capital_manager.yml" in mam_src,
              "MultiAssetManager lê conf/capital_manager.yml (dinâmico, suporta SOL automaticamente)", "MEDIUM")
    else:
        check(False, "src/trading/multi_asset_manager.py existe", "CRITICAL")

    # ========== RESUMO ==========
    print("\n" + "=" * 65)
    print("RESUMO")
    print("=" * 65)

    fail_critical = [i for i in issues if i["severity"] == "CRITICAL" and i["status"] == "❌"]
    fail_high = [i for i in issues if i["severity"] == "HIGH" and i["status"] == "❌"]
    fail_medium = [i for i in issues if i["severity"] == "MEDIUM" and i["status"] == "❌"]
    passed = [i for i in issues if i["status"] == "✅"]

    print(f"Total checks  : {len(issues)}")
    print(f"Passou        : {len(passed)}")
    print(f"❌ CRITICAL   : {len(fail_critical)}")
    print(f"❌ HIGH       : {len(fail_high)}")
    print(f"❌ MEDIUM     : {len(fail_medium)}")

    if fail_critical:
        print("\n🚨 AÇÃO IMEDIATA:")
        for i in fail_critical:
            print(f"  ❌ {i['message']}")
    if fail_high:
        print("\n⚠️  Corrigir em 24h:")
        for i in fail_high:
            print(f"  ❌ {i['message']}")

    print()
    if len(fail_critical) == 0 and len(fail_high) == 0:
        verdict = "🟢 GO — zero issues críticos/high. Bot pode continuar em paper."
    elif len(fail_critical) == 0 and len(fail_high) <= 2:
        verdict = "🟡 CORRIGIR-E-CONTINUAR — patches em 24h, bot pode rodar."
    elif len(fail_critical) <= 2:
        verdict = "🟡 CORRIGIR-E-CONTINUAR — patches urgentes < 4h."
    else:
        verdict = "🔴 PARAR BOT — 3+ critical issues."
    print(verdict)

    # ========== MARKDOWN ==========
    lines = [
        "# SOL Bot 4 — Auditoria de Conformidade",
        "",
        f"**Data:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Veredicto:** {verdict}",
        "",
        f"Total {len(issues)} checks | ❌ CRITICAL: {len(fail_critical)} | ❌ HIGH: {len(fail_high)} | ❌ MEDIUM: {len(fail_medium)}",
        "",
        "## Resultados",
        "",
    ]
    for i in issues:
        lines.append(f"- {i['status']} **[{i['severity']}]** {i['message']}")

    if fail_critical or fail_high:
        lines += ["", "## Ações"]
        if fail_critical:
            lines.append("### 🚨 AGORA")
            for i in fail_critical:
                lines.append(f"- {i['message']}")
        if fail_high:
            lines.append("")
            lines.append("### ⚠️ 24h")
            for i in fail_high:
                lines.append(f"- {i['message']}")

    AUDIT_REPORT.write_text("\n".join(lines))
    print(f"\nReport: {AUDIT_REPORT}")


if __name__ == "__main__":
    main()
