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

# Definição das variáveis para o modelo PCA (Análise de Qualidade)
# GRUPO MIN: Quanto MENOR, melhor (Risco, Custo, Preço)
COLS_PCA_MIN = [
    'p_vp', 'p_vpa', 'volatilidade', 
    'tax_administracao', 'tax_gestao', 'tax_performance'
]

# GRUPO MAX: Quanto MAIOR, melhor (Retorno, Liquidez, Tamanho)
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
# FUNÇÕES AUXILIARES (Lógica de Negócio)
# ==============================================================================

def categorizar_setor(s):
    """Normaliza os nomes dos setores para os 4 macro-grupos definidos."""
    s = str(s).strip()
    if s in ["Papéis", "Recebíveis Imobiliários", "Serviços Financeiros Diversos", "Indefinido"]: return "Papel"
    if s in ["Imóveis Industriais e Logísticos", "Logística"]: return "Tijolo - Logística"
    if s in ["Lajes Corporativas", "Agências de Bancos", "Educacional", "Hospitalar", 
             "Hotéis", "Imóveis Comerciais - Outros", "Exploração de Imóveis", 
             "Shoppings", "Varejo", "Imóveis Residenciais", "Misto", "Fundo de Desenvolvimento"]: return "Tijolo - Renda Urbana"
    return "Híbridos e Outros"

def get_robust_target(series, direction='max'):
    """
    Define o alvo ideal usando lógica de Boxplot (IQR) para ignorar outliers.
    Isso evita que um fundo com yield de 500% distorça a nota dos outros.
    """
    clean = series[series != 0].dropna()
    if clean.empty: return 0.0
    
    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    
    if direction == 'max':
        limit = Q3 + (1.5 * IQR)
        return min(limit, clean.max()) # Teto racional
    else:
        limit = Q1 - (1.5 * IQR)
        return max(limit, clean.min()) # Piso racional

def safe_float(val, default=0.0):
    try:
        if val is None or val == "": return default
        return float(val)
    except:
        return default

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Smart FII Solver Online", "version": "5.0 - Robust PCA"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. PARSING E VALIDAÇÃO
        content = request.get_json()
        
        if not content:
            return jsonify({"error": "Payload JSON vazio."}), 400

        # Tratamento flexível para receber dados (suporta lista direta ou objeto com chave 'dados')
        dados_json = content.get('dados', [])
        if not dados_json and isinstance(content, list):
            dados_json = content
        
        if not dados_json:
            return jsonify({"error": "Lista de fundos ('dados') não fornecida."}), 400

        # Extração de Parâmetros
        AMOUNT = safe_float(content.get('amount'), 1000.0)
        
        # Pesos Padrão (se não vier no JSON)
        PESOS_DEFAULT = {
            "Híbridos e Outros": 0.20,
            "Papel": 0.30,
            "Tijolo - Logística": 0.25,
            "Tijolo - Renda Urbana": 0.25
        }
        pesos_usuario = content.get('pesos', PESOS_DEFAULT)
        
        # Filtros
        filtros = content.get('filtros', {})
        MIN_LIQUIDEZ = safe_float(filtros.get('liquidez'), 200000.0)
        MIN_DY = safe_float(filtros.get('min_dy'), 6.0)
        MIN_PVP = safe_float(filtros.get('min_pvp'), 0.80)
        MAX_PVP = safe_float(filtros.get('max_pvp'), 1.20)
        MIN_ATIVOS = safe_float(filtros.get('min_ativos'), 3)

        # 2. PREPARAÇÃO DO DATAFRAME
        df = pd.DataFrame(dados_json)
        
        # Normalização de nomes de colunas
        if 'ticker' not in df.columns and 'fundos' in df.columns:
            df = df.rename(columns={'fundos': 'ticker'})
            
        if 'ticker' not in df.columns:
            return jsonify({"error": "Coluna 'ticker' ou 'fundos' obrigatória."}), 400

        # Conversão Numérica Robusta
        cols_to_numeric = ALL_PCA_COLS + ['preco_atual_r']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # Categorização
        df['macro_setor'] = df['setor'].apply(categorizar_setor)

        # 3. FILTRAGEM HARD (Corte Inicial)
        df_filtered = df[
            (df['liquidez_diaria_r'] >= MIN_LIQUIDEZ) &
            (df['p_vp'] >= MIN_PVP) &
            (df['p_vp'] <= MAX_PVP) &
            (df['dy_12m_acumulado'] >= MIN_DY) &
            (df['quant_ativos'] >= MIN_ATIVOS) &
            (df['preco_atual_r'] > 0)
        ].copy()

        if df_filtered.empty:
            return jsonify({"status": "aviso", "message": "Nenhum fundo passou nos filtros iniciais."}), 200

        # 4. PCA E SCORE DE QUALIDADE (Latente & Robusto)
        
        # Identificar colunas com dados válidos (variância > 0)
        valid_cols = [c for c in ALL_PCA_COLS if df_filtered[c].std() > 0]
        
        if len(valid_cols) < 2:
            return jsonify({"error": "Dados insuficientes para cálculo estatístico (colunas zeradas)."}), 400

        # A) TREINO APENAS NOS DADOS REAIS
        X_real = df_filtered[valid_cols].copy()
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        pca = PCA(n_components=0.95) # Explica 95% da variância
        
        X_imp = imputer.fit_transform(X_real)
        X_sc = scaler.fit_transform(X_imp)
        pca.fit(X_sc)
        
        # Coordenadas dos Reais
        X_pca_real = pca.transform(X_sc)
        eigenvalues = pca.explained_variance_ratio_
        df_filtered['coords'] = list(X_pca_real)

        # B) CRIAÇÃO DOS TARGETS (IDEAIS) VIA BOXPLOT
        targets_data = []
        for setor, grupo in df_filtered.groupby('macro_setor'):
            tgt = {'macro_setor': setor}
            for col in valid_cols:
                if col in COLS_PCA_MIN:
                    tgt[col] = get_robust_target(grupo[col], 'min')
                else:
                    tgt[col] = get_robust_target(grupo[col], 'max')
            targets_data.append(tgt)
            
        df_tgt = pd.DataFrame(targets_data)
        
        # Projeção dos Targets (Latentes)
        X_tgt = df_tgt[valid_cols]
        X_tgt_imp = imputer.transform(X_tgt)
        X_tgt_sc = scaler.transform(X_tgt_imp)
        X_tgt_pca = pca.transform(X_tgt_sc)
        
        target_map = dict(zip(df_tgt['macro_setor'], X_tgt_pca))

        # C) CÁLCULO DO SCORE
        def calc_match(row):
            t_vec = target_map.get(row['macro_setor'])
            if t_vec is None: return 0.0
            r_vec = np.array(row['coords'])
            # Distância Euclidiana Ponderada
            dist = np.sqrt(np.sum(((r_vec - t_vec)**2) * eigenvalues))
            return dist

        df_filtered['dist'] = df_filtered.apply(calc_match, axis=1)
        
        # Normalização (0 a 100) por setor
        df_filtered['max_dist'] = df_filtered.groupby('macro_setor')['dist'].transform('max')
        df_filtered['min_dist'] = df_filtered.groupby('macro_setor')['dist'].transform('min')
        
        # Score = 100 se distância for mínima, 0 se for máxima (dentro do setor)
        # Adicionamos small epsilon para evitar divisão por zero
        df_filtered['match_score'] = 100 * (1 - (
            (df_filtered['dist'] - df_filtered['min_dist']) / 
            (df_filtered['max_dist'] - df_filtered['min_dist'] + 1e-9)
        ))
        
        # Filtro de qualidade mínima para o Solver (Score > 30)
        candidatos = df_filtered[df_filtered['match_score'] > 30].copy()

        if candidatos.empty:
             return jsonify({"status": "aviso", "message": "Nenhum fundo atingiu o score mínimo de qualidade."}), 200

        # 5. OTIMIZAÇÃO (SOLVER)
        carteira_final = []
        
        for setor, peso in pesos_usuario.items():
            budget = AMOUNT * peso
            pool = candidatos[candidatos['macro_setor'] == setor].copy()
            
            if pool.empty or budget < 10:
                continue
                
            # Configuração do Problema
            prob = LpProblem(f"Otimizacao_{setor}", LpMaximize)
            tickers = pool['ticker'].tolist()
            
            # Mapas de dados para acesso rápido
            precos = pool.set_index('ticker')['preco_atual_r'].to_dict()
            yields = pool.set_index('ticker')['dy_12m_acumulado'].to_dict()
            scores = pool.set_index('ticker')['match_score'].to_dict()
            
            # Variável de Decisão: Quantidade Inteira
            x = LpVariable.dicts("Qtd", tickers, lowBound=0, cat='Integer')
            
            # OBJETIVO: Maximizar Yield Ponderado pela Qualidade
            # Um fundo com Score 100 entrega 100% da sua utilidade (Yield)
            # Um fundo com Score 50 entrega 50% da utilidade
            prob += lpSum([x[t] * (yields[t] * (scores[t]/100.0)) for t in tickers])
            
            # RESTRIÇÃO 1: Orçamento
            prob += lpSum([x[t] * precos[t] for t in tickers]) <= budget
            
            # RESTRIÇÃO 2: Concentração (Max 35% do budget em um ativo, se houver variedade)
            if len(tickers) >= 3:
                limite_ativo = budget * 0.35
                for t in tickers:
                    prob += x[t] * precos[t] <= limite_ativo
            
            # Resolver
            prob.solve(PULP_CBC_CMD(msg=False))
            
            # Coletar
            for t in tickers:
                qtd = value(x[t])
                if qtd and qtd > 0:
                    row = pool[pool['ticker'] == t].iloc[0]
                    carteira_final.append({
                        "ticker": t,
                        "setor": setor,
                        "qtd_cotas": int(qtd),
                        "preco_medio": row['preco_atual_r'],
                        "total_investido": round(qtd * row['preco_atual_r'], 2),
                        "match_score": round(row['match_score'], 1),
                        "dy_12m": row['dy_12m_acumulado']
                    })

        # 6. MONTAGEM DA RESPOSTA
        df_res = pd.DataFrame(carteira_final)
        
        if df_res.empty:
            return jsonify({"status": "aviso", "message": "Não foi possível alocar capital (budget insuficiente para cotas)."}), 200
            
        df_res = df_res.sort_values(['setor', 'total_investido'], ascending=[True, False])
        
        total_inv = df_res['total_investido'].sum()
        sobra = AMOUNT - total_inv
        
        # Yield Médio da Carteira
        dy_ponderado = (df_res['dy_12m'] * df_res['total_investido']).sum() / total_inv

        response_payload = {
            "status": "sucesso",
            "resumo": {
                "aporte_inicial": AMOUNT,
                "total_investido": round(total_inv, 2),
                "sobra_caixa": round(sobra, 2),
                "dy_medio_anual": round(dy_ponderado, 2),
                "qtd_ativos": len(df_res)
            },
            "carteira": df_res.to_dict(orient='records')
        }
        
        return jsonify(response_payload)

    except Exception as e:
        # Em produção, use logging ao invés de print
        tracebox = traceback.format_exc()
        print(tracebox)
        return jsonify({"error": "Erro interno no servidor.", "detalhes": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
