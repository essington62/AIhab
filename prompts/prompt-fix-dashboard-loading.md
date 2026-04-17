# Task: Fix dashboard loading — tela cinza no primeiro acesso

## Problema

O dashboard (Dash/Plotly) carrega com blocos cinzas sem formatação no primeiro acesso via browser. Só funciona após refresh manual (F5). Isso acontece tanto em acesso local quanto via IP público (54.232.162.161).

Sintoma: HTML renderiza mas CSS/JS não carregam na primeira requisição. Na segunda chamada (refresh) funciona porque assets estão em cache.

Spec completa do projeto em `CLAUDE.md` na raiz do repo btc_AI.

## Diagnóstico provável

1. **Dash serve_locally**: Se `app = Dash(__name__, serve_locally=True)` não está configurado, o Dash tenta carregar assets de CDN (plotly.js, dash.js). Se o CDN demora ou falha, o layout aparece sem estilo.

2. **Layout dinâmico pesado**: Se `app.layout` é uma função que faz queries pesadas (lê parquets, chama APIs), o primeiro request pode dar timeout antes dos assets carregarem.

3. **Suppress callback exceptions**: Se callbacks disparam antes dos componentes existirem no DOM, pode causar erro silencioso que trava o render.

4. **Loading state ausente**: Sem `dcc.Loading` wrapper, o browser mostra HTML parcial enquanto callbacks rodam.

## Correções

### 1. Forçar serve_locally=True

Em `src/dashboard/app.py`:

```python
app = Dash(
    __name__,
    serve_locally=True,           # NÃO depender de CDN
    suppress_callback_exceptions=True,
)
```

Se já está `serve_locally=True`, verificar se os assets estão sendo copiados corretamente no Docker build.

### 2. Layout estático + callbacks para dados

Separar o layout (estrutura HTML/CSS) dos dados. O layout deve ser estático e leve — os dados vêm via callbacks:

```python
# BOM — layout é uma estrutura estática, dados vêm de callbacks
app.layout = html.Div([
    dcc.Interval(id='interval', interval=60*1000),
    html.Div(id='dashboard-content', children=[
        # Estrutura com placeholders
        html.Div("Loading...", id='score-card'),
        html.Div("Loading...", id='regime-card'),
        # ...
    ])
])

@app.callback(Output('score-card', 'children'), Input('interval', 'n_intervals'))
def update_score(n):
    # Aqui sim faz a query pesada
    return calculate_score()
```

```python
# RUIM — layout faz queries pesadas no import
app.layout = build_heavy_layout()  # Bloqueia o primeiro render
```

Se o layout atual é uma função pesada, refatorar para separar estrutura de dados.

### 3. Adicionar Loading wrapper

Envolver o conteúdo principal com `dcc.Loading`:

```python
app.layout = html.Div([
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id='dashboard-content')
    )
])
```

### 4. Meta tags para forçar render

Adicionar meta tags que ajudam no primeiro render:

```python
app = Dash(
    __name__,
    serve_locally=True,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"http-equiv": "X-UA-Compatible", "content": "IE=edge"},
    ],
)
```

### 5. Health check endpoint

Adicionar endpoint `/health` que retorna 200 sem carregar o dashboard completo. Útil para verificar se o server está rodando:

```python
@app.server.route('/health')
def health():
    return 'ok'
```

### 6. Verificar Dockerfile

Garantir que no Dockerfile:
- Assets estáticos são copiados: `COPY assets/ /app/assets/`
- Port está exposta: `EXPOSE 8050`
- Server roda com host 0.0.0.0: `app.run(host='0.0.0.0', port=8050)`

Se usa gunicorn:
```dockerfile
CMD ["gunicorn", "src.dashboard.app:server", "-b", "0.0.0.0:8050", "--timeout", "120", "--workers", "1"]
```

O `--timeout 120` garante que requests pesados não dão timeout (default gunicorn é 30s).

## Testes

1. Abrir dashboard em aba anônima (sem cache) → deve carregar corretamente na primeira vez
2. Fazer hard refresh (Ctrl+Shift+R) → deve carregar corretamente
3. Verificar console do browser (F12) → sem erros de assets 404
4. Endpoint `/health` retorna 200

## Entregáveis

1. `src/dashboard/app.py` corrigido
2. Dockerfile atualizado se necessário
3. Commit + push:
   ```bash
   git add -A && git commit -m "fix: dashboard first-load rendering (serve_locally + loading state)"
   git push origin master:main
   ```

## Restrições

- **NÃO alterar** lógica de cálculo de scores, gates, ou regime
- **NÃO alterar** layout visual (cards, cores, ordem dos elementos)
- **NÃO remover** nenhuma seção do dashboard
- Só corrigir o problema de carregamento inicial
- Manter compatibilidade com Docker (serve_locally é obrigatório no container)
