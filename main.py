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
    return jsonify({"status": "API FIIs Online", "versao": "2.0 Dinamica"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        # 1. RECEBIMENTO E PARSING
        content = request.get_json()
        
        # Garante que temos dados
        if not content:
            return jsonify({"error": "Nenhum JSON recebido"}), 400

        # Lógica para extrair 'dados', 'amount' e 'pesos'
        # Suporta o formato antigo (lista direta) ou novo (objeto completo)
        if isinstance(content, dict) and 'dados' in content:
            dados_json = content['dados']
            AMOUNT = float(content.get('amount', 5200)) # Padrão 5200 se não vier
            pesos_usuario = content.get('pesos', {})    # Padrão vazio se não vier
        else:
            # Fallback para compatibilidade se enviar a lista pura
            dados_json = content
            AMOUNT = 5200
            pesos_usuario = {}

        if not dados_json:
            return jsonify({"error": "O campo 'dados' está vazio"}), 400

        fii = pd.DataFrame(dados_json)

        # ==========================================================
        # 2. DEFINIÇÃO DINÂMICA DE PESOS
        # ==========================================================
        
        # Pesos padrão (Caso o usuário não envie nada ou envie parcial)
        PESOS_PADRAO = {
            "Híbridos e Outros": 0.20,
            "Papel": 0.25,
            "Tijolo - Logística": 0.30,
            "Tijolo - Renda Urbana": 0.25
        }

        # Atualiza os pesos padrão com o que veio da requisição
        # Ex: Se o usuário mandar só {"Papel": 0.50}, o resto continua o padrão.
        PESOS_SETORIAIS = {**PESOS_PADRAO, **pesos_usuario}

        # Opcional: Normalizar pesos para garantir que a soma dê 1 (100%)
        total_pesos = sum(PESOS_SETORIAIS.values())
        if total_pesos > 0:
            PESOS_SETORIAIS = {k: v/total_pesos for k, v in PESOS_SETORIAIS.items()}

        # ==========================================================
        # 3. CONFIGURAÇÃO DE FILTROS E VARIÁVEIS
        # ==========================================================
        CORTE_LIQUIDEZ = 200000
        CORTE_PATRIMONIO = 250000000
        CORTE_COTISTAS = 10000
        MIN_PVP = 0.70
        MAX_PVP = 1.20
        MIN_ATIVOS = 3
        MIN_DY_12M = 6.0 
        MIN_VAR_PAT = -10.0
        CORTE_PRECO = 60.00

        # ==========================================================
        # 4. TRATAMENTO DE DADOS (Pre-processing)
        # ==========================================================
        fii = fii.fillna(0)
        
        # Converte colunas numéricas críticas para float (segurança extra)
        cols_numericas = ['liquidez_diaria_r', 'patrimonio_liquido', 'num_cotistas', 
                          'p_vp', 'dy_12m_acumulado', 'quant_ativos', 
                          'variacao_patrimonial', 'preco_atual_r']
        
        for col in cols_numericas:
            if col in fii.columns:
                fii[col] = pd.to_numeric(fii[col], errors='coerce').fillna(0)

        # Aplicação dos Filtros
        fii = fii[
            (fii['liquidez_diaria_r'] >= CORTE_LIQUIDEZ) &
            (fii['patrimonio_liquido'] >= CORTE_PATRIMONIO) &
            (fii['num_cotistas'] >= CORTE_COTISTAS) &
            (fii['p_vp'] >= MIN_PVP) & (fii['p_vp'] <= MAX_PVP) &
            (fii['dy_12m_acumulado'] >= MIN_DY_12M) &
            (fii['quant_ativos'] >= MIN_ATIVOS) &
            (fii['variacao_patrimonial'] > MIN_VAR_PAT) &
            (fii['preco_atual_r'] >= CORTE_PRECO)
        ].copy()

        if fii.empty:
            return jsonify({"aviso": "Nenhum fundo passou nos filtros de segurança."}), 200

        # Categorização de Setores
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

        # ==========================================================
        # 5. DEFINIÇÃO DO TARGET (Fundo Ideal)
        # ==========================================================
        
        cols_min = ['preco_atual_r', 'p_vp', 'p_vpa', 'variacao_preco', 'volatilidade', 
                    'tax_gestao', 'tax_performance', 'tax_administracao']
        
        cols_max = ['liquidez_diaria_r', 'ultimo_dividendo', 'dividend_yield', 
                    'dy_3m_acumulado', 'dy_6m_acumulado', 'dy_12m_acumulado',
                    'dy_3m_media', 'dy_6m_media', 'dy_12m_media', 'dy_ano',
                    'dy_patrimonial', 'rentab_periodo', 'rentab_acumulada',
                    'variacao_patrimonial', 'rentab_patr_periodo', 'rentab_patr_acumulada',
                    'patrimonio_liquido', 'vpa', 'quant_ativos', 'num_cotistas']

        agg_dict = {c: 'min' for c in cols_min if c in fii.columns}
        agg_dict.update({c: 'max' for c in cols_max if c in fii.columns})

        fii_ref_setorial = fii.groupby('macro_setor').agg(agg_dict).reset_index()
        fii_ref_setorial['fundos'] = "TGT_" + fii_ref_setorial['macro_setor']
        fii_ref_setorial['setor'] = fii_ref_setorial['macro_setor']

        fii_combined = pd.concat([fii, fii_ref_setorial], ignore_index=True)

        # ==========================================================
        # 6. PCA E DISTÂNCIA (Cérebro do Algoritmo)
        # ==========================================================
        
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
        # 7. ALOCAÇÃO FINANCEIRA (Usando os Pesos Dinâmicos)
        # ==========================================================
        
        finalistas = fii_real.sort_values('dist').groupby('macro_setor').head(3)
        carteira_final = []

        for setor, grupo in finalistas.groupby('macro_setor'):
            # AQUI: Usa o peso vindo do JSON ou o padrão
            peso_alvo = PESOS_SETORIAIS.get(setor, 0) 
            budget = AMOUNT * peso_alvo
            
            if budget > 0: # Só aloca se tiver orçamento para o setor
                scores = 1 / (grupo['dist'] + 1e-6)
                pesos_rel = scores / scores.sum()
                
                alocacao = budget * pesos_rel
                qtd = np.floor(alocacao / grupo['preco_atual_r'])
                total = qtd * grupo['preco_atual_r']
                
                res = grupo[['fundos', 'macro_setor', 'preco_atual_r', 'dist', 'dy_12m_acumulado', 'p_vp']].copy()
                res['qtd_cotas'] = qtd
                res['total_investido'] = total
                carteira_final.append(res)

        if not carteira_final:
             return jsonify({"aviso": "Não foi possível gerar carteira (verifique os pesos ou filtros)."}), 200

        df_final = pd.concat(carteira_final).sort_values(['macro_setor', 'total_investido'], ascending=[True, False])

        return jsonify({
            "carteira": df_final.to_dict(orient='records'),
            "resumo": {
                "aporte_inicial": AMOUNT,
                "pesos_utilizados": PESOS_SETORIAIS,
                "total_investido": round(df_final['total_investido'].sum(), 2),
                "sobra_caixa": round(AMOUNT - df_final['total_investido'].sum(), 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
