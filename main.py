from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API de Carteira FIIs Online!"})

@app.route('/processar_carteira', methods=['POST'])
def processar_carteira():
    try:
        content = request.get_json()
        
        # Lógica para suportar formato direto ou {dados: [], amount: 0}
        if isinstance(content, dict) and 'dados' in content:
            dados_json = content['dados']
            AMOUNT = float(content.get('amount', 5200))
        else:
            dados_json = content
            AMOUNT = 5200

        if not dados_json:
            return jsonify({"error": "Nenhum dado recebido"}), 400

        df = pd.DataFrame(dados_json)

        # --- CONFIGURAÇÕES ---
        CORTE_LIQUIDEZ = 200000       
        CORTE_PATRIMONIO = 250000000  
        MIN_PVP = 0.70
        MAX_PVP = 1.20
        MIN_DY_12M = 6.0              
        CORTE_PRECO = 60.00           

        PESOS_SETORIAIS = {
            "Híbridos e Outros": 0.20,
            "Papel": 0.25,
            "Tijolo - Logística": 0.30,
            "Tijolo - Renda Urbana": 0.25
        }

        # --- LIMPEZA ---
        cols_num = ['preco', 'liquidez', 'pvp', 'dy_mensal', 'dy_ano', 'ultimo_div', 'patrimonio']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- FILTROS ---
        df_filtrado = df[
            (df['liquidez'] >= CORTE_LIQUIDEZ) &
            (df['patrimonio'] >= CORTE_PATRIMONIO) &
            (df['pvp'] >= MIN_PVP) & (df['pvp'] <= MAX_PVP) &
            (df['dy_ano'] >= MIN_DY_12M) &
            (df['preco'] >= CORTE_PRECO)
        ].copy()

        if df_filtrado.empty:
            return jsonify({"aviso": "Nenhum fundo passou nos filtros."}), 200

        # --- MACRO SETORES ---
        def definir_macro_setor(setor):
            setor = str(setor)
            if setor in ["Papéis", "Serviços Financeiros Diversos"]:
                return "Papel"
            elif setor in ["Imóveis Industriais e Logísticos", "Logística"]:
                return "Tijolo - Logística"
            elif setor in ["Lajes Corporativas", "Agências de Bancos", "Educacional", 
                           "Hospitalar", "Hotéis", "Imóveis Comerciais - Outros",
                           "Exploração de Imóveis", "Shoppings", "Varejo", 
                           "Tecidos. Vestuário e Calçados", "Imóveis Residenciais", 
                           "Incorporações"]:
                return "Tijolo - Renda Urbana"
            else:
                return "Híbridos e Outros"

        df_filtrado['macro_setor'] = df_filtrado['setor'].apply(definir_macro_setor)

        # --- TARGETS ---
        targets = df_filtrado.groupby('macro_setor').agg({
            'preco': 'min',       
            'pvp': 'min',         
            'liquidez': 'max',    
            'ultimo_div': 'max',
            'dy_mensal': 'max',
            'dy_ano': 'max',
            'patrimonio': 'max'
        }).reset_index()

        targets['ticker'] = "TGT_" + targets['macro_setor']
        
        # --- PCA ---
        colunas_pca = ['preco', 'liquidez', 'pvp', 'dy_mensal', 'dy_ano', 'ultimo_div', 'patrimonio']
        df_pca_input = pd.concat([df_filtrado, targets], ignore_index=True).fillna(0)

        X = df_pca_input[colunas_pca]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=None)
        X_pca = pca.fit_transform(X_scaled)
        weights = pca.explained_variance_ratio_

        df_pca_input['pca_coords'] = list(X_pca)

        # --- DISTÂNCIA ---
        df_targets = df_pca_input[df_pca_input['ticker'].str.startswith('TGT_')].set_index('macro_setor')
        df_funds = df_pca_input[~df_pca_input['ticker'].str.startswith('TGT_')].copy()

        def calcular_distancia(row):
            target_coords = df_targets.loc[row['macro_setor'], 'pca_coords']
            my_coords = row['pca_coords']
            sq_diff = (my_coords - target_coords) ** 2
            weighted_dist = np.sqrt(np.sum(sq_diff * weights))
            return round(weighted_dist, 2)

        df_funds['dist'] = df_funds.apply(calcular_distancia, axis=1)

        # --- ALOCAÇÃO ---
        finalistas = df_funds.sort_values('dist').groupby('macro_setor').head(3)
        carteira_final = []

        for setor, grupo in finalistas.groupby('macro_setor'):
            peso_setor = PESOS_SETORIAIS.get(setor, 0)
            budget_setor = AMOUNT * peso_setor
            scores = 1 / (grupo['dist'] + 0.0001)
            pesos_relativos = scores / scores.sum()
            alocacao_reais = budget_setor * pesos_relativos
            qtd_cotas = np.floor(alocacao_reais / grupo['preco'])
            total_investido = qtd_cotas * grupo['preco']
            
            resultado = grupo[['ticker', 'macro_setor', 'preco', 'dy_ano', 'pvp', 'dist']].copy()
            resultado['qtd_cotas'] = qtd_cotas
            resultado['total_investido'] = total_investido
            carteira_final.append(resultado)

        if len(carteira_final) > 0:
            df_carteira = pd.concat(carteira_final).sort_values(['macro_setor', 'total_investido'], ascending=[True, False])
            return jsonify({
                "carteira": df_carteira.to_dict(orient='records'),
                "resumo": {
                    "aporte": AMOUNT,
                    "total_investido": round(df_carteira['total_investido'].sum(), 2)
                }
            })
        else:
             return jsonify({"aviso": "Erro ao gerar carteira"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
