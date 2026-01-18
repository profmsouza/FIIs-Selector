from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API FIIs Online", "versao": "2.3 - Gamification Mode"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. RECEBIMENTO E PARSING
        content = request.get_json()
        
        if not content:
            return jsonify({"error": "Nenhum JSON recebido"}), 400

        # Verifica se 'dados' existe (lista de FIIs vinda do n8n)
        dados_json = content.get('dados', [])
        if not dados_json and isinstance(content, list):
             dados_json = content # Fallback se vier lista direta

        if not dados_json:
            return jsonify({"error": "O campo 'dados' está vazio ou inválido"}), 400

        # --- PARÂMETROS DO USUÁRIO ---
        AMOUNT = float(content.get('amount', 5200))
        raw_top_n = content.get('top_n', 3)
        if not raw_top_n: raw_top_n = 3
        pesos_usuario = content.get('pesos', {})

        # --- FILTROS DINÂMICOS (GAMIFICAÇÃO) ---
        # Busca 'filtros' no JSON, se não achar, usa dicionário vazio e cai no default do get
        filtros_user = content.get('filtros', {})

        CORTE_LIQUIDEZ = float(filtros_user.get('liquidez', 200000))
        CORTE_PATRIMONIO = float(filtros_user.get('patrimonio', 250000000))
        CORTE_COTISTAS = int(filtros_user.get('cotistas', 10000))
        MIN_PVP = float(filtros_user.get('min_pvp', 0.70))
        MAX_PVP = float(filtros_user.get('max_pvp', 1.20))
        MIN_ATIVOS = int(filtros_user.get('min_ativos', 3))
        MIN_DY_12M = float(filtros_user.get('min_dy', 6.0))
        MIN_VAR_PAT = float(filtros_user.get('min_var_pat', -10.0))
        CORTE_PRECO = float(filtros_user.get('max_preco', 60.00)) # Atenção: mudei lógica para MAX preço ou MIN?
        # No seu original era >= CORTE_PRECO, vou manter a lógica original:
        MIN_PRECO = float(filtros_user.get('min_preco', 60.00)) 

        fii = pd.DataFrame(dados_json)

        # ==========================================================
        # 2. DEFINIÇÃO DINÂMICA DE PESOS
        # ==========================================================
        PESOS_PADRAO = {
            "Híbridos e Outros": 0.20,
            "Papel": 0.25,
            "Tijolo - Logística": 0.30,
            "Tijolo - Renda Urbana": 0.25
        }
        PESOS_SETORIAIS = {**PESOS_PADRAO, **pesos_usuario}

        # Tratamento de NAs e Tipos
        fii = fii.fillna(0)
        cols_numericas = ['liquidez_diaria_r', 'patrimonio_liquido', 'num_cotistas', 
                          'p_vp', 'dy_12m_acumulado', 'quant_ativos', 
                          'variacao_patrimonial', 'preco_atual_r']
        for col in cols_numericas:
            if col in fii.columns:
                fii[col] = pd.to_numeric(fii[col], errors='coerce').fillna(0)

        # ==========================================================
        # 3. APLICAÇÃO DOS FILTROS (AGORA DINÂMICOS)
        # ==========================================================
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
            return jsonify({"aviso": "Nenhum fundo passou nos filtros selecionados. Tente relaxar as restrições."}), 200

        # ... (Mantém a lógica de Categorização, Target e PCA inalterada até o cálculo da distância) ...
        # [CÓDIGO DE CATEGORIZAÇÃO E PCA AQUI - IGUAL AO ANTERIOR]
        # Vou resumir para não estourar o limite, mas assuma que o bloco 4 e 5 estão aqui.
        
        # Recriando o bloco necessário para o contexto
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
        fii_ref_setorial['fundos'] = "TGT_" + fii_ref_setorial['macro_setor']
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

        targets_coords = fii_combined[fii_combined['fundos'].str.startswith('TGT_')].set_index('macro_setor')['pca_coords']
        fii_real = fii_combined[~fii_combined['fundos'].str.startswith('TGT_')].copy()

        def calc_weighted_dist(row):
            target_vec = targets_coords[row['macro_setor']]
            fund_vec = row['pca_coords']
            sq_diff = (fund_vec - target_vec) ** 2
            weighted_sq_diff = sq_diff * eigenvalues_weights 
            return np.sqrt(np.sum(weighted_sq_diff))

        fii_real['dist'] = fii_real.apply(calc_weighted_dist, axis=1)

        # ==========================================================
        # 7. ALOCAÇÃO E MATCH SCORE
        # ==========================================================
        carteira_final = []

        for setor, grupo in fii_real.groupby('macro_setor'):
            n_para_este_setor = int(raw_top_n.get(setor, 3)) if isinstance(raw_top_n, dict) else int(raw_top_n)
            
            melhores_do_setor = grupo.sort_values('dist').head(n_para_este_setor)

            peso_alvo = PESOS_SETORIAIS.get(setor, 0)
            budget = AMOUNT * peso_alvo
            
            if budget > 0 and not melhores_do_setor.empty:
                scores = 1 / (melhores_do_setor['dist'] + 1e-6)
                pesos_rel = scores / scores.sum()
                
                alocacao = budget * pesos_rel
                qtd = np.floor(alocacao / melhores_do_setor['preco_atual_r'])
                total = qtd * melhores_do_setor['preco_atual_r']
                
                # --- CALCULO DO MATCH SCORE (0 a 100%) ---
                # A lógica: Se dist=0, score=100. Conforme dist aumenta, score cai.
                # Ajuste o 'fator_sensibilidade' se quiser que o score caia mais rápido ou devagar.
                fator_sensibilidade = 1 
                match_score_series = 100 * (1 / (1 + (melhores_do_setor['dist'] * fator_sensibilidade)))

                res = melhores_do_setor[['fundos', 'macro_setor', 'preco_atual_r', 'dist', 'dy_12m_acumulado', 'p_vp']].copy()
                res['qtd_cotas'] = qtd
                res['total_investido'] = total
                res['match_score'] = match_score_series.round(1) # Ex: 95.5%

                carteira_final.append(res)

        if not carteira_final:
             return jsonify({"aviso": "Não foi possível alocar capital."}), 200

        df_final = pd.concat(carteira_final).sort_values(['macro_setor', 'total_investido'], ascending=[True, False])

        return jsonify({
            "carteira": df_final.to_dict(orient='records'),
            "resumo": {
                "aporte_inicial": AMOUNT,
                "total_investido": round(df_final['total_investido'].sum(), 2),
                "filtros_utilizados": filtros_user # Retorna para feedback visual
            }
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
