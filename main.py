from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value
import traceback
import json

app = Flask(__name__)

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================

COLS_PCA_MIN = [
    'p_vp', 'p_vpa', 'volatilidade', 
    'tax_administracao', 'tax_gestao', 'tax_performance'
]

COLS_PCA_MAX = [
    'liquidez_diaria_r', 'patrimonio_liquido', 'num_cotistas', 'quant_ativos',
    'ultimo_dividendo', 'dividend_yield',
    'dy_3m_acumulado', 'dy_6m_acumulado', 'dy_12m_acumulado', 'dy_ano',
    'dy_3m_media', 'dy_6m_media', 'dy_12m_media', 'dy_patrimonial',
    'rentab_periodo', 'rentab_acumulada', 'rentab_patr_periodo', 'rentab_patr_acumulada',
    'variacao_patrimonial', 'variacao_preco'
]

ALL_PCA_COLS = COLS_PCA_MIN + COLS_PCA_MAX

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def categorizar_setor(s):
    s = str(s).strip()
    if s in ["Papéis", "Recebíveis Imobiliários", "Serviços Financeiros Diversos", "Indefinido"]: return "Papel"
    if s in ["Imóveis Industriais e Logísticos", "Logística"]: return "Tijolo - Logística"
    if s in ["Lajes Corporativas", "Agências de Bancos", "Educacional", "Hospitalar", 
             "Hotéis", "Imóveis Comerciais - Outros", "Exploração de Imóveis", 
             "Shoppings", "Varejo", "Imóveis Residenciais", "Misto", "Fundo de Desenvolvimento"]: return "Tijolo - Renda Urbana"
    return "Híbridos e Outros"

def get_robust_target(series, direction='max'):
    clean = series[series != 0].dropna()
    if clean.empty: return 0.0
    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    if direction == 'max':
        limit = Q3 + (1.5 * IQR)
        return min(limit, clean.max())
    else:
        limit = Q1 - (1.5 * IQR)
        return max(limit, clean.min())

def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None and val != "" else default
    except:
        return default

def safe_int(val, default=0):
    try:
        return int(float(val)) if val is not None and val != "" else default
    except:
        return default

def safe_json_val(val):
    if pd.isna(val) or np.isinf(val): return 0.0
    return float(val)

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Smart FII Solver Online", "version": "7.0 - Max Yield Weighted"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. PARSING ROBUSTO
        raw_content = request.get_json()
        if not raw_content: return jsonify({"error": "Payload vazio."}), 400

        content = raw_content
        if isinstance(raw_content, list):
            if len(raw_content) > 0: content = raw_content[0]
            else: return jsonify({"error": "Lista vazia."}), 400

        if isinstance(content, dict) and 'body' in content:
            if isinstance(content['body'], dict): content = content['body']
            elif isinstance(content['body'], str):
                try: content = json.loads(content['body'])
                except: pass

        dados_json = content.get('dados', [])
        if not dados_json: return jsonify({"error": "Lista 'dados' ausente."}), 400

        # Parâmetros
        AMOUNT = safe_float(content.get('amount'), 1000.0)
        PESOS_DEFAULT = {
            "Híbridos e Outros": 0.20, "Papel": 0.30,
            "Tijolo - Logística": 0.25, "Tijolo - Renda Urbana": 0.25
        }
        pesos_usuario = content.get('pesos', PESOS_DEFAULT)
        filtros = content.get('filtros', {})

        # Filtros
        MIN_LIQUIDEZ = safe_float(filtros.get('liquidez'), 0)
        MIN_PVP = safe_float(filtros.get('min_pvp'), 0)
        MAX_PVP = safe_float(filtros.get('max_pvp'), 999)
        MIN_ATIVOS = safe_int(filtros.get('min_ativos'), 0)
        MIN_DY = safe_float(filtros.get('min_dy'), 0)
        if 0 < MIN_DY < 1.0: MIN_DY *= 100
        
        MIN_PATRIMONIO = safe_float(filtros.get('patrimonio'), 0)
        MIN_COTISTAS = safe_int(filtros.get('cotistas'), 0)
        MIN_PRECO = safe_float(filtros.get('min_preco'), 0)
        MIN_VAR_PAT = safe_float(filtros.get('min_var_pat'), -999)

        # 2. DATAFRAME
        df = pd.DataFrame(dados_json)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={'fundos': 'ticker', 'ativo': 'ticker', 'codigo': 'ticker'})
        
        if 'ticker' not in df.columns: return jsonify({"error": "Coluna 'ticker' obrigatória."}), 400
        if 'setor' not in df.columns: df['setor'] = 'Indefinido'

        # Conversão Numérica
        cols_to_numeric = [c.lower() for c in ALL_PCA_COLS] + ['preco_atual_r', 'variacao_patrimonial']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # 3. FILTRAGEM
        df_filtered = df[
            (df['liquidez_diaria_r'] >= MIN_LIQUIDEZ) &
            (df['patrimonio_liquido'] >= MIN_PATRIMONIO) &
            (df['num_cotistas'] >= MIN_COTISTAS) &
            (df['p_vp'] >= MIN_PVP) & (df['p_vp'] <= MAX_PVP) &
            (df['dy_12m_acumulado'] >= MIN_DY) &
            (df['quant_ativos'] >= MIN_ATIVOS) &
            (df['preco_atual_r'] >= MIN_PRECO) &
            (df['variacao_patrimonial'] >= MIN_VAR_PAT)
        ].copy()

        if df_filtered.empty:
            return jsonify({"status": "aviso", "message": "Nenhum fundo passou nos filtros."}), 200

        # 4. PCA E SCORE
        df_filtered['macro_setor'] = df_filtered['setor'].apply(categorizar_setor)
        valid_cols = [c for c in df_filtered.columns if c in cols_to_numeric and df_filtered[c].std() > 0]
        
        if len(valid_cols) > 2:
            X = df_filtered[valid_cols].copy()
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            pca = PCA(n_components=0.95)
            
            X_sc = scaler.fit_transform(imputer.fit_transform(X))
            pca.fit(X_sc)
            X_pca = pca.transform(X_sc)
            eigenvalues = pca.explained_variance_ratio_
            df_filtered['coords'] = list(X_pca)

            targets = []
            for setor, grupo in df_filtered.groupby('macro_setor'):
                tgt = {'macro_setor': setor}
                for col in valid_cols:
                    direction = 'min' if col in [c.lower() for c in COLS_PCA_MIN] else 'max'
                    tgt[col] = get_robust_target(grupo[col], direction)
                targets.append(tgt)
            
            df_tgt = pd.DataFrame(targets)
            X_tgt_pca = pca.transform(scaler.transform(imputer.transform(df_tgt[valid_cols])))
            target_map = dict(zip(df_tgt['macro_setor'], X_tgt_pca))
            
            def calc_match(row):
                t = target_map.get(row['macro_setor'])
                if t is None: return 0.0
                return np.sqrt(np.sum(((np.array(row['coords']) - t)**2) * eigenvalues))

            df_filtered['dist'] = df_filtered.apply(calc_match, axis=1)
            
            df_filtered['max_dist'] = df_filtered.groupby('macro_setor')['dist'].transform('max')
            df_filtered['min_dist'] = df_filtered.groupby('macro_setor')['dist'].transform('min')
            
            df_filtered['match_score'] = 100 * (1 - (
                (df_filtered['dist'] - df_filtered['min_dist']) / 
                (df_filtered['max_dist'] - df_filtered['min_dist'] + 1e-9)
            ))
        else:
            df_filtered['match_score'] = 50.0

        candidatos = df_filtered[df_filtered['match_score'] > 20].copy()
        if candidatos.empty:
             return jsonify({"status": "aviso", "message": "Score de qualidade baixo."}), 200

        # ==============================================================================
        # 5. OTIMIZAÇÃO (SOLVER AJUSTADO PARA MAXIMIZAR INVESTIMENTO)
        # ==============================================================================
        carteira_final = []
        
        for setor, peso in pesos_usuario.items():
            budget = AMOUNT * peso
            pool = candidatos[candidatos['macro_setor'] == setor].copy()
            
            # Se não tem candidatos ou o budget não paga nem a cota mais barata
            if pool.empty or (budget < pool['preco_atual_r'].min()):
                continue
            
            prob = LpProblem(f"Opt_{setor}", LpMaximize)
            tickers = pool['ticker'].tolist()
            precos = pool.set_index('ticker')['preco_atual_r'].to_dict()
            yields = pool.set_index('ticker')['dy_12m_acumulado'].to_dict()
            scores = pool.set_index('ticker')['match_score'].to_dict()
            
            x = LpVariable.dicts("Qtd", tickers, lowBound=0, cat='Integer')
            
            # --- FUNÇÃO OBJETIVO OTIMIZADA PARA CAIXA ZERO ---
            # Maximiza: Total de Dividendos Ponderados pelo Score
            # Adiciona um "Bonus de Aporte" (0.0001 * Investido) para forçar o gasto do caixa residual
            prob += lpSum([
                (x[t] * precos[t] * (yields[t]/100.0) * (scores[t]/100.0)) + # Dividendos Ponderados
                (x[t] * precos[t] * 0.0001) # Tie-Breaker para gastar o caixa
                for t in tickers
            ])
            
            # Restrição do Budget
            prob += lpSum([x[t] * precos[t] for t in tickers]) <= budget
            
            # --- TRAVA DE CONCENTRAÇÃO DINÂMICA ---
            # Garante "pelo menos 1 ativo" relaxando a trava se houver poucos candidatos
            n_ativos_disponiveis = len(tickers)
            
            if n_ativos_disponiveis == 1:
                limite_ativo = budget # 100% (All-in se for o único disponível)
            elif n_ativos_disponiveis == 2:
                limite_ativo = budget * 0.60 # 60% (Garante compra dos dois)
            else:
                limite_ativo = budget * 0.35 # 35% (Padrão de segurança)
                
            for t in tickers: 
                prob += x[t] * precos[t] <= limite_ativo
            
            prob.solve(PULP_CBC_CMD(msg=False))
            
            for t in tickers:
                q = value(x[t])
                if q and q > 0:
                    row = pool[pool['ticker'] == t].iloc[0]
                    carteira_final.append({
                        "fundos": t,
                        "qtd": int(q),
                        "preco": float(row['preco_atual_r']),
                        "total": round(q * row['preco_atual_r'], 2),
                        "setor": setor,
                        "dy": float(row.get('dy_12m_acumulado', 0)),
                        "p_vp": safe_json_val(row.get('p_vp', 0)),
                        "match_score": round(safe_json_val(row.get('match_score', 0)),1)
                    })

        df_res = pd.DataFrame(carteira_final)
        
        if df_res.empty: return jsonify({"status": "aviso", "message": "Orçamento insuficiente."}), 200

        total_inv = df_res['total'].sum()
        dy_pond = (df_res['dy'] * df_res['total']).sum() / total_inv if total_inv > 0 else 0

        # Reordenação personalizada: Papel -> Logística -> Renda Urbana -> Híbridos
        ordem_setores = ["Papel", "Tijolo - Logística", "Tijolo - Renda Urbana", "Híbridos e Outros"]
        df_res['setor'] = pd.Categorical(df_res['setor'], categories=ordem_setores, ordered=True)
        df_res = df_res.sort_values('setor')

        return jsonify({
            "status": "sucesso",
            "resumo": {
                "total_investido": round(total_inv, 2),
                "sobra": round(AMOUNT - total_inv, 2),
                "dy_medio_carteira": round(dy_pond, 2),
                "qtd_ativos": len(df_res)
            },
            "carteira": df_res.to_dict(orient='records')
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Erro interno.", "detalhes": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
