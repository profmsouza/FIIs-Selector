from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# BIBLIOTECA DE PESQUISA OPERACIONAL
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value
import traceback
import json

app = Flask(__name__)

# --- CONFIGURAÇÃO FIXA ---
TOP_N_CANDIDATES = 10 

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API FIIs Online", "versao": "3.5 - Fix NaN Mask Error"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. RECEBIMENTO E PARSING
        content = request.get_json()
        
        if not content:
            return jsonify({"error": "Nenhum JSON recebido"}), 400

        # Verifica se 'dados' existe
        dados_json = content.get('dados', [])
        
        if isinstance(dados_json, str):
            try:
                dados_json = json.loads(dados_json)
            except:
                dados_json = []

        if not dados_json and isinstance(content, list):
             dados_json = content 

        if not dados_json:
            return jsonify({"error": "O campo 'dados' está vazio ou inválido."}), 400

        # Funções de Segurança
        def safe_float(val, default):
            try:
                if val is None or val == "": return default
                return float(val)
            except:
                return default

        def safe_int(val, default):
            try:
                if val is None or val == "": return default
                return int(float(val))
            except:
                return default

        # --- PARÂMETROS ---
        AMOUNT = safe_float(content.get('amount'), 5200.0)
        pesos_usuario = content.get('pesos', {})
        filtros_user = content.get('filtros', {})

        # --- FILTROS ---
        CORTE_LIQUIDEZ = safe_float(filtros_user.get('liquidez'), 200000.0)
        CORTE_PATRIMONIO = safe_float(filtros_user.get('patrimonio'), 250000000.0)
        CORTE_COTISTAS = safe_int(filtros_user.get('cotistas'), 10000)
        MIN_PVP = safe_float(filtros_user.get('min_pvp'), 0.70)
        MAX_PVP = safe_float(filtros_user.get('max_pvp'), 1.20)
        MIN_ATIVOS = safe_int(filtros_user.get('min_ativos'), 3)
        MIN_VAR_PAT = safe_float(filtros_user.get('min_var_pat'), -10.0)
        MIN_PRECO = safe_float(filtros_user.get('min_preco'), 60.00)

        input_dy = safe_float(filtros_user.get('min_dy'), 6.0)
        if input_dy < 1.0 and input_dy > 0: 
            MIN_DY_12M = input_dy * 100 
        else:
            MIN_DY_12M = input_dy

        # 2. TRATAMENTO DE DADOS
        fii = pd.DataFrame(dados_json)

        # Padronização de Colunas (Fundos -> Ticker)
        if 'ticker' not in fii.columns and 'fundos' in fii.columns:
            fii = fii.rename(columns={'fundos': 'ticker'})
        
        if 'ticker' not in fii.columns:
             return jsonify({"error": "A coluna 'fundos' ou 'ticker' não foi encontrada nos dados."}), 400
        
        # Garante que ticker é string e remove vazios
        fii['ticker'] = fii['ticker'].astype(str).str.strip()

        PESOS_PADRAO = {
            "Híbridos e Outros": 0.20, "Papel": 0.25,
            "Tijolo - Logística": 0.30, "Tijolo - Renda Urbana": 0.25
        }
        PESOS_SETORIAIS = {**PESOS_PADRAO, **pesos_usuario}

        fii = fii.fillna(0)
        cols_numericas = ['liquidez_diaria_r', 'patrimonio_liquido', 'num_cotistas', 
                          'p_vp', 'dy_12m_acumulado', 'quant_ativos', 
                          'variacao_patrimonial', 'preco_atual_r', 'ultimo_dividendo']
        
        for col in cols_numericas:
            if col in fii.columns:
                fii[col] = pd.to_numeric(fii[col], errors='coerce').fillna(0)

        if 'ultimo_dividendo' not in fii.columns or fii['ultimo_dividendo'].sum() == 0:
             fii['ultimo_dividendo'] = (fii['preco_atual_r'] * (fii['dy_12m_acumulado'] / 100)) / 12

        # 3. APLICAÇÃO DOS FILTROS
        fii = fii[
            (fii['liquidez_diaria_r'] >= CORTE_LIQUIDEZ) &
            (fii['patrimonio_liquido'] >= CORTE_PATRIMONIO) &
            (fii['num_cotistas'] >= CORTE_COTISTAS) &
            (fii['p_vp'] >= MIN_PVP) & (fii['p_vp'] <= MAX_PVP) &
            (fii['dy_12m_acumulado'] >= MIN_DY_12M) &
            (fii['quant_ativos'] >= MIN_ATIVOS) &
            (fii['variacao_patrimonial'] > MIN_VAR_PAT) &
            (fii['preco_atual_r'] >= MIN_PRECO)
        ].copy()

        if fii.empty:
            return jsonify({"aviso": "Nenhum fundo passou nos filtros selecionados."}), 200

        # 4, 5, 6. CATEGORIZAÇÃO E PCA
        def categorizar_setor(s):
            s = str(s)
            if s in ["Papéis", "Serviços Financeiros Diversos"]: return "Papel"
            if s in ["Imóveis Industriais e Logísticos", "Logística"]: return "Tijolo - Logística"
            if s in ["Lajes Corporativas", "Agências de Bancos", "Educacional", "Hospitalar", 
                     "Hotéis", "Imóveis Comerciais - Outros", "Exploração de Imóveis", 
                     "Shoppings", "Varejo", "Tecidos. Vestuário e Calçados", 
                     "Imóveis Residenciais", "Incorporações"]: return "Tijolo - Renda Urbana"
            return "Híbridos e Outros"

        fii['macro_setor'] = fii['setor'].apply(categorizar_setor)
        
        cols_min = ['preco_atual_r', 'p_vp', 'p_vpa', 'variacao_preco', 'volatilidade', 'tax_gestao', 'tax_performance', 'tax_administracao']
        cols_max = ['liquidez_diaria_r', 'ultimo_dividendo', 'dividend_yield', 'dy_3m_acumulado', 'dy_6m_acumulado', 'dy_12m_acumulado', 'dy_3m_media', 'dy_6m_media', 'dy_12m_media', 'dy_ano', 'dy_patrimonial', 'rentab_periodo', 'rentab_acumulada', 'variacao_patrimonial', 'rentab_patr_periodo', 'rentab_patr_acumulada', 'patrimonio_liquido', 'vpa', 'quant_ativos', 'num_cotistas']

        agg_dict = {c: 'min' for c in cols_min if c in fii.columns}
        agg_dict.update({c: 'max' for c in cols_max if c in fii.columns})

        fii_ref_setorial = fii.groupby('macro_setor').agg(agg_dict).reset_index()
        # [CORREÇÃO AQUI]: Usamos 'ticker' para o target também, evitando colunas duplicadas com NaN
        fii_ref_setorial['ticker'] = "TGT_" + fii_ref_setorial['macro_setor']
        fii_ref_setorial['setor'] = fii_ref_setorial['macro_setor']
        
        fii_combined = pd.concat([fii, fii_ref_setorial], ignore_index=True)
        numeric_cols = list(agg_dict.keys())
        X = fii_combined[numeric_cols]
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        pca = PCA(n_components=None)
        X_pca = pca.fit_transform(X_scaled)
        eigenvalues_weights = pca.explained_variance_ratio_
        fii_combined['pca_coords'] = list(X_pca)

        # [CORREÇÃO AQUI]: Garantimos que não há NaNs na coluna ticker antes do startswith
        fii_combined['ticker'] = fii_combined['ticker'].fillna('')
        
        targets_coords = fii_combined[fii_combined['ticker'].str.startswith('TGT_')].set_index('macro_setor')['pca_coords']
        fii_real = fii_combined[~fii_combined['ticker'].str.startswith('TGT_')].copy()

        def calc_weighted_dist(row):
            target_vec = targets_coords[row['macro_setor']]
            fund_vec = row['pca_coords']
            sq_diff = (fund_vec - target_vec) ** 2
            weighted_sq_diff = sq_diff * eigenvalues_weights 
            return np.sqrt(np.sum(weighted_sq_diff))

        fii_real['dist'] = fii_real.apply(calc_weighted_dist, axis=1)

        # 7. ALOCAÇÃO
        carteira_final = []

        for setor, grupo in fii_real.groupby('macro_setor'):
            
            peso_alvo = PESOS_SETORIAIS.get(setor, 0)
            budget_disponivel = AMOUNT * peso_alvo
            
            # Shortlist PCA
            pool_candidatos = grupo.sort_values('dist').head(TOP_N_CANDIDATES).copy()
            pool_candidatos = pool_candidatos[pool_candidatos['dy_12m_acumulado'] > 0]

            if budget_disponivel <= 0 or pool_candidatos.empty:
                continue

            # Solver Knapsack
            prob = LpProblem(f"Knapsack_{setor}", LpMaximize)
            
            tickers = pool_candidatos['ticker'].tolist()
            precos = pool_candidatos.set_index('ticker')['preco_atual_r'].to_dict()
            
            retorno_em_reais = {}
            for t, row in pool_candidatos.set_index('ticker').iterrows():
                retorno_em_reais[t] = row['preco_atual_r'] * (row['dy_12m_acumulado'] / 100)

            qtd_vars = LpVariable.dicts("Qtd", tickers, lowBound=0, cat='Integer')

            prob += lpSum([qtd_vars[t] * retorno_em_reais[t] for t in tickers])
            prob += lpSum([qtd_vars[t] * precos[t] for t in tickers]) <= budget_disponivel

            prob.solve(PULP_CBC_CMD(msg=False))

            max_dist_setor = pool_candidatos['dist'].max()
            
            for t in tickers:
                qtd_otima = int(value(qtd_vars[t]))
                
                if qtd_otima > 0:
                    row = pool_candidatos[pool_candidatos['ticker'] == t].iloc[0]
                    total_alocado = qtd_otima * row['preco_atual_r']
                    
                    match_score = 100.0
                    if max_dist_setor > 0:
                         match_score = (100 * (1 - (row['dist'] / max_dist_setor))).round(1)

                    res = {
                        'fundos': t,
                        'macro_setor': setor,
                        'preco_atual_r': row['preco_atual_r'],
                        'dy_12m_acumulado': row['dy_12m_acumulado'],
                        'dist_pca': row['dist'],
                        'qtd_cotas': qtd_otima,
                        'total_investido': total_alocado,
                        'match_score': match_score
                    }
                    carteira_final.append(res)

        if not carteira_final:
             return jsonify({"aviso": "Não foi possível alocar capital."}), 200

        df_final = pd.DataFrame(carteira_final).sort_values(['macro_setor', 'total_investido'], ascending=[True, False])
        df_final = df_final.replace([np.inf, -np.inf], 0).fillna(0)

        INVESTIMENTO = round(float(df_final['total_investido'].sum()), 2)
        SOBRA = round(AMOUNT - INVESTIMENTO, 2)
        
        dy_ponderado = 0
        if INVESTIMENTO > 0:
            dy_ponderado = (df_final['dy_12m_acumulado'] * df_final['total_investido']).sum() / INVESTIMENTO

        return jsonify({
            "carteira": df_final.to_dict(orient='records'),
            "resumo": {
                "aporte_inicial": AMOUNT,
                "total_investido": INVESTIMENTO,
                "sobra_caixa": SOBRA,
                "dy_medio_carteira": round(dy_ponderado, 2),
                "metodo": "Híbrido: Top 10 PCA + Solver Knapsack",
                "filtros_utilizados": filtros_user
            }
        })

    except Exception as e:
        print(traceback.format_exc()) 
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
