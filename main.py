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
        if val is None or val == "": return default
        return float(val)
    except:
        return default

def safe_int(val, default=0):
    try:
        if val is None or val == "": return default
        return int(float(val))
    except:
        return default

# Função nova para garantir que valores NaN/Inf não quebrem o JSON
def safe_json_val(val):
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return val

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Smart FII Solver Online", "version": "6.0 - Blindada"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. PARSING ROBUSTO (Correção para estrutura n8n e listas)
        raw_content = request.get_json()
        
        if not raw_content:
            return jsonify({"error": "Payload JSON vazio."}), 400

        content = raw_content

        # Se for lista (padrão n8n), pega o primeiro item
        if isinstance(content, list):
            if len(content) > 0:
                content = content[0]
            else:
                return jsonify({"error": "Lista de entrada vazia."}), 400

        # Se tiver encapsulado em 'body' (Webhook n8n), desenrola
        if isinstance(content, dict) and 'body' in content:
            if isinstance(content['body'], dict):
                content = content['body']
            elif isinstance(content['body'], str):
                try:
                    content = json.loads(content['body'])
                except:
                    pass 

        # Extração de Dados
        dados_json = content.get('dados', [])
        if not dados_json:
            return jsonify({"error": "Lista 'dados' não encontrada no JSON."}), 400

        # Parâmetros
        AMOUNT = safe_float(content.get('amount'), 1000.0)
        PESOS_DEFAULT = {
            "Híbridos e Outros": 0.20, "Papel": 0.30,
            "Tijolo - Logística": 0.25, "Tijolo - Renda Urbana": 0.25
        }
        pesos_usuario = content.get('pesos', PESOS_DEFAULT)
        filtros = content.get('filtros', {})

        # 2. DATAFRAME E LIMPEZA
        df = pd.DataFrame(dados_json)
        
        # Normaliza nomes das colunas (tudo minúsculo, sem espaço)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Garante coluna identificadora
        rename_map = {'fundos': 'ticker', 'ativo': 'ticker', 'codigo': 'ticker'}
        df = df.rename(columns=rename_map)
        
        if 'ticker' not in df.columns:
            return jsonify({"error": "Coluna 'ticker'/'fundos' obrigatória."}), 400
        
        # Garante coluna setor
        if 'setor' not in df.columns:
            df['setor'] = 'Indefinido'

        # Conversão Numérica
        cols_to_numeric = [c.lower() for c in ALL_PCA_COLS] + ['preco_atual_r', 'variacao_patrimonial']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # 3. FILTRAGEM
        MIN_LIQUIDEZ = safe_float(filtros.get('liquidez'), 0)
        MIN_PVP = safe_float(filtros.get('min_pvp'), 0)
        MAX_PVP = safe_float(filtros.get('max_pvp'), 999)
        MIN_DY = safe_float(filtros.get('min_dy'), 0)
        if 0 < MIN_DY < 1.0: MIN_DY *= 100 # Ajuste percentual
        
        MIN_PATRIMONIO = safe_float(filtros.get('patrimonio'), 0)
        MIN_COTISTAS = safe_int(filtros.get('cotistas'), 0)
        MIN_PRECO = safe_float(filtros.get('min_preco'), 0)
        MIN_VAR_PAT = safe_float(filtros.get('min_var_pat'), -999)

        df_filtered = df[
            (df['liquidez_diaria_r'] >= MIN_LIQUIDEZ) &
            (df['patrimonio_liquido'] >= MIN_PATRIMONIO) &
            (df['num_cotistas'] >= MIN_COTISTAS) &
            (df['p_vp'] >= MIN_PVP) & (df['p_vp'] <= MAX_PVP) &
            (df['dy_12m_acumulado'] >= MIN_DY) &
            (df['preco_atual_r'] >= MIN_PRECO) &
            (df['variacao_patrimonial'] >= MIN_VAR_PAT)
        ].copy()

        if df_filtered.empty:
            return jsonify({"status": "aviso", "message": "Nenhum fundo passou nos filtros."}), 200

        # 4. PROCESSAMENTO (Categorização e PCA)
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

            # Targets
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
            
            # Score
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

        # 5. SOLVER
        carteira_final = []
        
        for setor, peso in pesos_usuario.items():
            budget = AMOUNT * peso
            pool = candidatos[candidatos['macro_setor'] == setor].copy()
            
            if pool.empty or budget < 10: continue
            
            prob = LpProblem(f"Opt_{setor}", LpMaximize)
            tickers = pool['ticker'].tolist()
            precos = pool.set_index('ticker')['preco_atual_r'].to_dict()
            yields = pool.set_index('ticker')['dy_12m_acumulado'].to_dict()
            scores = pool.set_index('ticker')['match_score'].to_dict()
            
            x = LpVariable.dicts("Qtd", tickers, lowBound=0, cat='Integer')
            
            # Maximizar (Yield * Score)
            prob += lpSum([x[t] * (yields.get(t,0) * (scores.get(t,50)/100.0)) for t in tickers])
            
            # Budget
            prob += lpSum([x[t] * precos.get(t,0) for t in tickers]) <= budget
            
            if len(tickers) >= 3:
                limit = budget * 0.35
                for t in tickers: prob += x[t] * precos.get(t,0) <= limit
            
            prob.solve(PULP_CBC_CMD(msg=False))
            
            for t in tickers:
                q = value(x[t])
                if q and q > 0:
                    row = pool[pool['ticker'] == t].iloc[0]
                    # AQUI ESTAVA O SEU PROBLEMA: Adicionar variáveis de forma segura
                    carteira_final.append({
                        "fundos": t,
                        "qtd_cotas": int(q),
                        "preco_atual_r": round(q * row['preco_atual_r'], 2),
                        "macro_setor": setor,
                        "dy_12m_acumulado": float(row.get('dy_12m_acumulado', 0)),
                        "p_vp": safe_json_val(row.get('p_vp', 0)), # Protegido
                        "match_score": safe_json_val(row.get('match_score', 0)) # Protegido
                    })

        df_res = pd.DataFrame(carteira_final)
        
        if df_res.empty:
            return jsonify({"status": "aviso", "message": "Orçamento insuficiente."}), 200

        # Totais
        total_inv = df_res['total'].sum()
        dy_pond = (df_res['dy'] * df_res['total']).sum() / total_inv if total_inv > 0 else 0

        return jsonify({
            "status": "sucesso",
            "resumo": {
                "total_investido": round(total_inv, 2),
                "sobra": round(AMOUNT - total_inv, 2),
                "dy_medio": round(dy_pond, 2),
                "qtd_ativos": len(df_res)
            },
            "carteira": df_res.to_dict(orient='records')
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Erro interno no servidor.", "detalhes": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
