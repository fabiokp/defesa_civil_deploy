"""
🚨 SIMULADOR DE ALERTAS DE DESASTRES - DEFESA CIVIL
Dashboard Interativo com Streamlit

Execução:
    streamlit run app_simulador_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from pathlib import Path
import json

# ========================================
# CONFIGURAÇÃO DE CAMINHOS (RELATIVOS)
# ========================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================
st.set_page_config(
    page_title="Simulador de Alertas - Defesa Civil",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# FUNÇÕES AUXILIARES (do notebook)
# ========================================

@st.cache_data
def carregar_base():
    """Carrega base de dados (com cache) - CAMINHO RELATIVO"""
    csv_path = DATA_DIR / "df_defesa_civil_categorizado.csv"
    
    if not csv_path.exists():
        st.error(f"❌ Arquivo não encontrado: {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def carregar_modelos():
    """Carrega modelos treinados (com cache) - CAMINHO RELATIVO"""
    loaded_models = {}
    
    if not MODELS_DIR.exists():
        st.error(f"❌ Diretório de modelos não encontrado: {MODELS_DIR}")
        st.stop()
    
    model_files = list(MODELS_DIR.glob("*.pkl"))
    
    if not model_files:
        st.error("❌ Nenhum modelo encontrado!")
        st.stop()
    
    for model_path in model_files:
        parts = model_path.stem.split('_')
        if len(parts) >= 2:
            target_name = '_'.join(parts[:2])
        else:
            target_name = parts[0]
        
        try:
            loaded_models[target_name] = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Erro ao carregar {model_path.name}: {e}")
    
    return loaded_models

@st.cache_data
def carregar_dados_documentacao():
    """Carrega dados do JSON de documentação técnica - CAMINHO RELATIVO"""
    json_path = DOCS_DIR / "dados_relatorio_tecnico.json"
    
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return None

def get_dados_municipio(df, cod_municipio):
    """Extrai dados do município"""
    mun_data = df[df['Cod_IBGE_Mun'] == cod_municipio].iloc[0]
    
    return {
        'nome': mun_data['Nome_Municipio'],
        'uf': mun_data['Sigla_UF'],
        'regiao': mun_data['regiao'],
        'populacao': int(mun_data['populacao']) if pd.notna(mun_data['populacao']) else 0,
        'pib_pc': float(mun_data['pib_pc']) if pd.notna(mun_data['pib_pc']) else 0,
        'hierarquia_urbana': mun_data['hierarquia_urbana'],
        'semiarido': mun_data['semiarido'],
        'cobertura_saude': float(mun_data['proporcao_cobertura_total_atencao_basica']) if pd.notna(mun_data['proporcao_cobertura_total_atencao_basica']) else 0,
        'cod_ibge': cod_municipio
    }

def criar_input_predicao(dados_mun, tipo_desastre, feature_cols):
    """Cria DataFrame de input para predição"""
    input_data = {
        'regiao': dados_mun['regiao'],
        'grupo_de_desastre': tipo_desastre,
        'pib_pc': dados_mun['pib_pc'],
        'populacao': dados_mun['populacao'],
        'hierarquia_urbana': dados_mun['hierarquia_urbana'],
        'semiarido': dados_mun['semiarido'],
        'proporcao_cobertura_total_atencao_basica': dados_mun['cobertura_saude']
    }
    
    input_data = {k: v for k, v in input_data.items() if k in feature_cols}
    return pd.DataFrame([input_data])

def fazer_predicoes(input_df, models_dict, df_base):
    """Faz predições com todos os modelos"""
    predicoes = {}
    
    for target_name, model_obj in models_dict.items():
        try:
            pipeline = model_obj['pipeline']
            class_mapping = model_obj['class_mapping']
            feature_cols = model_obj['feature_cols']
            
            has_preprocessor = 'preprocessor' in pipeline.named_steps
            
            if has_preprocessor:
                input_to_predict = input_df.copy()
            else:
                categorical_features = [f for f in ['regiao', 'grupo_de_desastre', 'hierarquia_urbana', 'semiarido'] if f in feature_cols]
                numeric_features = [f for f in ['pib_pc', 'populacao', 'proporcao_cobertura_total_atencao_basica'] if f in feature_cols]
                
                transformers = []
                if categorical_features:
                    transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))
                if numeric_features:
                    transformers.append(('num', StandardScaler(), numeric_features))
                
                preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
                preprocessor.fit(df_base[feature_cols].dropna())
                input_to_predict = preprocessor.transform(input_df)
            
            pred_encoded = pipeline.predict(input_to_predict)[0]
            pred_proba = pipeline.predict_proba(input_to_predict)[0] if hasattr(pipeline, 'predict_proba') else None
            pred_label = class_mapping[pred_encoded]
            
            predicoes[target_name] = {
                'predicao': pred_label,
                'probabilidades': dict(zip(class_mapping.values(), pred_proba)) if pred_proba is not None else None,
                'confianca': max(pred_proba) if pred_proba is not None else None
            }
            
        except Exception as e:
            st.error(f"Erro na predição de {target_name}: {e}")
            predicoes[target_name] = {'predicao': 'Erro', 'probabilidades': None, 'confianca': None}
    
    return predicoes

def plotar_kpis_streamlit(predicoes, dados_mun, tipo_desastre):
    """Plota KPIs usando Matplotlib para Streamlit"""
    
    labels_descritivos = {
        'DH_mortos': 'Danos Humanos\nMortes/Feridos',
        'DH_total': 'Danos Humanos Totais\nEnfermos, Desalojados, Desabrigados, Desaparecidos',
        'DM_total': 'Danos Materiais\nInstalações Destruídas',
        'PEPL_total': 'Prejuízos Econômicos\nSetor Público',
        'PEPR_total': 'Prejuízos Econômicos\nSetor Privado'
    }
    
    cores_severidade = {
        'Nenhum': '#2ecc71',
        'Nenhum Dano': '#2ecc71',
        'Com Dano': '#e74c3c',
        'Baixo/Médio': '#f39c12',
        'Alto': '#e74c3c'
    }
    
    n_predicoes = len(predicoes)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    titulo = f'ALERTA DE DESASTRE - {dados_mun["nome"]}/{dados_mun["uf"]}\nTipo de Desastre: {tipo_desastre}'
    fig.suptitle(titulo, fontsize=18, weight='bold', y=0.98)
    
    for idx, (target_name, pred_info) in enumerate(predicoes.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        target_label = labels_descritivos.get(target_name, target_name.replace('_', ' ').title())
        predicao = pred_info['predicao']
        confianca = pred_info['confianca']
        cor = cores_severidade.get(predicao, '#95a5a6')
        
        ax.barh([0], [1], color='#ecf0f1', height=0.5)
        
        if confianca:
            ax.barh([0], [confianca], color=cor, height=0.5, alpha=0.8)
        
        ax.text(0.5, 0, f'{predicao}', 
                ha='center', va='center', fontsize=14, weight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=cor, edgecolor='black', linewidth=2))
        
        if confianca:
            ax.text(0.5, -0.8, f'Confiança: {confianca*100:.1f}%', 
                    ha='center', va='center', fontsize=10, style='italic')
        
        ax.set_title(target_label, fontsize=10, weight='bold', pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
    
    for idx in range(n_predicoes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

# ========================================
# INTERFACE STREAMLIT
# ========================================

def main():
    # ========================================
    # SISTEMA DE ABAS
    # ========================================
    tab1, tab2 = st.tabs(["🚨 Simulador de Alertas", "📋 Metodologia"])
    
    # ========================================
    # ABA 1: SIMULADOR
    # ========================================
    with tab1:
        # Título principal
        st.title("🚨 Simulador de Alertas de Desastres")
        st.markdown("**Sistema de Predição de Impactos - Defesa Civil**")
        st.markdown("---")
        
        # Carregar dados (com spinner)
        with st.spinner("Carregando base de dados..."):
            df_base = carregar_base()
        
        with st.spinner("Carregando modelos de Machine Learning..."):
            loaded_models = carregar_modelos()
        
        if not loaded_models:
            st.error("❌ Nenhum modelo encontrado! Verifique o diretório de modelos.")
            st.stop()
        
        st.success(f"✅ {len(loaded_models)} modelos carregados com sucesso!")
        
        # SIDEBAR
        st.sidebar.header("⚙️ Configurações da Simulação")
        
        municipios_unicos = df_base[['Cod_IBGE_Mun', 'Nome_Municipio', 'Sigla_UF']].drop_duplicates()
        municipios_unicos['label'] = municipios_unicos['Nome_Municipio'] + ' - ' + municipios_unicos['Sigla_UF']
        municipios_unicos = municipios_unicos.sort_values('label')
        
        municipio_selecionado = st.sidebar.selectbox(
            "🏙️ Selecione o Município",
            options=municipios_unicos['Cod_IBGE_Mun'].tolist(),
            format_func=lambda x: municipios_unicos[municipios_unicos['Cod_IBGE_Mun'] == x]['label'].iloc[0],
            index=0
        )
        
        dados_municipio = get_dados_municipio(df_base, municipio_selecionado)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Dados do Município:**")
        st.sidebar.write(f"**População:** {dados_municipio['populacao']:,} hab")
        st.sidebar.write(f"**PIB per capita:** R$ {dados_municipio['pib_pc']:,.2f}")
        st.sidebar.write(f"**Região:** {dados_municipio['regiao']}")
        
        desastres_historicos = df_base[df_base['Cod_IBGE_Mun'] == municipio_selecionado]['grupo_de_desastre'].unique()
        desastres_historicos = sorted(desastres_historicos)
        
        st.sidebar.markdown("---")
        tipo_desastre = st.sidebar.selectbox(
            "🌪️ Selecione o Tipo de Desastre",
            options=desastres_historicos if len(desastres_historicos) > 0 else ["Hidrológico", "Geológico", "Meteorológico"],
            index=0
        )
        
        if tipo_desastre not in desastres_historicos:
            st.sidebar.warning("⚠️ Este desastre não consta no histórico deste município.")
        
        st.sidebar.markdown("---")
        simular = st.sidebar.button("🚀 SIMULAR ALERTA", type="primary", use_container_width=True)
        
        if simular:
            with st.spinner("🤖 Executando predições..."):
                feature_cols = list(loaded_models.values())[0]['feature_cols']
                input_predicao = criar_input_predicao(dados_municipio, tipo_desastre, feature_cols)
                predicoes = fazer_predicoes(input_predicao, loaded_models, df_base)
                st.success("✅ Predições concluídas!")
            
            st.markdown("## 📊 Dashboard de Alertas")
            fig = plotar_kpis_streamlit(predicoes, dados_municipio, tipo_desastre)
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### 📋 Resumo das Predições")
            
            resumo_data = []
            for target, pred_info in predicoes.items():
                resumo_data.append({
                    'Tipo de Dano': target.replace('_', ' ').title(),
                    'Predição': pred_info['predicao'],
                    'Confiança': f"{pred_info['confianca']*100:.1f}%" if pred_info['confianca'] else "N/A"
                })
            
            df_resumo = pd.DataFrame(resumo_data)
            st.dataframe(df_resumo, use_container_width=True)
            
            alertas_altos = sum(1 for p in predicoes.values() 
                                if 'Alto' in p['predicao'] or 'Com Dano' in p['predicao'])
            
            if alertas_altos >= 3:
                nivel = "🔴 CRÍTICO"
                cor_fundo = "#ffebee"
            elif alertas_altos >= 1:
                nivel = "🟠 ELEVADO"
                cor_fundo = "#fff3e0"
            else:
                nivel = "🟢 BAIXO"
                cor_fundo = "#e8f5e9"
            
            st.markdown("---")
            st.markdown(f"""
            <div style="background-color: {cor_fundo}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2>🚨 NÍVEL DE ALERTA: {nivel}</h2>
                <p><strong>Município:</strong> {dados_municipio['nome']}/{dados_municipio['uf']}</p>
                <p><strong>Desastre:</strong> {tipo_desastre}</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("👈 Configure os parâmetros na barra lateral e clique em **SIMULAR ALERTA**.")
            st.markdown("---")
            st.markdown("### ℹ️ Sobre o Sistema")
            st.markdown("""
            Este sistema utiliza **Machine Learning** (XGBoost, Random Forest) para prever impactos de desastres.
            
            **Predições disponíveis:**
            - 🚑 Danos Humanos (Mortos/Feridos)
            - 🏘️ Danos Materiais
            - 💰 Prejuízos Econômicos
            """)
    
    # ========================================
    # ABA 2: METODOLGIA
    # ========================================
    with tab2:
        st.markdown("# 📋 Documentação Técnica do Projeto")
        
        dados_doc = carregar_dados_documentacao()
        
        if dados_doc:
            # Seção 1: Visão Geral
            st.markdown("## 1. Visão Geral do Projeto")
            st.markdown("""
            Este projeto desenvolve um sistema preditivo baseado em algoritmos de **Machine Learning** 
            para auxiliar a **Defesa Civil Brasileira** na antecipação e planejamento de respostas 
            a desastres naturais.
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registros", f"{dados_doc['base_dados']['geral']['total_registros']:,}")
            with col2:
                st.metric("Municípios", f"{dados_doc['base_dados']['geral']['municipios_unicos']:,}")
            with col3:
                st.metric("Modelos", dados_doc['modelos']['total_modelos_treinados'])
            with col4:
                st.metric("F1-Score Médio", f"{dados_doc['resumo_desempenho']['media_f1']:.3f}")
            
            # Seção 1.5: Atlas Brasileiro de Desastres
            st.markdown("---")
            st.markdown("## 1.5. Fonte Primária: Atlas Brasileiro de Desastres")
            
            with st.expander("📖 Sobre o Atlas Brasileiro de Desastres", expanded=True):
                st.markdown("""
                O **Atlas Brasileiro de Desastres** constitui a principal fonte de dados deste projeto. 
                Trata-se de uma ferramenta oficial do governo federal, gerida pela **Secretaria Nacional de 
                Proteção e Defesa Civil (Sedec/MIDR)** em parceria com o **CEPED/UFSC**.
                
                **Características:**
                - **Fonte:** Sistema Integrado de Informações sobre Desastres (S2iD)
                - **Período:** 1991 até o presente (atualizações anuais)
                - **Cobertura:** Todos os 5.570 municípios brasileiros
                - **Dados:** Danos humanos, materiais e prejuízos econômicos
                """)
                
                st.markdown("**Limitações Metodológicas:**")
                st.warning("""
                - **Subnotificação:** Municípios com menor capacidade administrativa tendem a subnotificar eventos
                - **Heterogeneidade:** Padronização dos formulários evoluiu ao longo das décadas
                - **Incentivo à superestimativa:** Busca por recursos federais pode inflar estimativas
                - **Variabilidade temporal:** Dados pós-2013 (informatização) possuem maior qualidade
                """)
            
            # Seção 2: Base de Dados
            st.markdown("---")
            st.markdown("## 2. Arquitetura dos Dados")
            
            st.markdown(f"**Período coberto:** {dados_doc['base_dados']['geral']['periodo']}")
            
            st.markdown("### Distribuição Geográfica")
            df_regional = pd.DataFrame(list(dados_doc['base_dados']['distribuicao_regional'].items()), 
                                      columns=['Região', 'Registros'])
            df_regional['Percentual'] = (df_regional['Registros'] / df_regional['Registros'].sum() * 100).round(1)
            st.dataframe(df_regional, use_container_width=True)
            
            st.markdown("### Tipos de Desastres")
            df_desastres = pd.DataFrame(list(dados_doc['base_dados']['distribuicao_desastres'].items()),
                                       columns=['Tipo', 'Ocorrências'])
            df_desastres['Percentual'] = (df_desastres['Ocorrências'] / df_desastres['Ocorrências'].sum() * 100).round(1)
            st.dataframe(df_desastres, use_container_width=True)
            
            # Seção 3: Targets
            st.markdown("---")
            st.markdown("## 3. Variáveis Alvo (Targets)")
            
            for target_name, target_info in dados_doc['targets'].items():
                with st.expander(f"📊 {target_name.replace('_', ' ').title()}", expanded=False):
                    df_dist = pd.DataFrame(list(target_info['distribuicao'].items()),
                                          columns=['Categoria', 'Frequência'])
                    df_dist['Proporção (%)'] = [f"{v:.1f}%" for v in target_info['distribuicao_pct'].values()]
                    st.dataframe(df_dist, use_container_width=True)
                    st.info(f"**Imbalance Ratio:** {target_info['imbalance_ratio']:.2f} (ideal = 1.0)")
            
            # Seção 4: Desafios
            st.markdown("---")
            st.markdown("## 4. Desafios Metodológicos")
            
            with st.expander("⚠️ Desbalanceamento Extremo de Classes"):
                st.markdown("""
                70-95% dos registros apresentavam valor zero (sem danos), criando viés sistemático.
                
                **Abordagem Adotada:**
                - Aplicação de SMOTE (ratio 1:5) para targets binários
                - `class_weight='balanced'` em todos os estimadores
                - Otimização de recall da classe minoritária
                """)
            
            with st.expander("⚠️ Outliers e Inconsistências"):
                st.markdown("""
                Registros com danos humanos superiores à população municipal.
                
                **Abordagem Adotada:**
                - Criação de ratio (danos_humanos/população)
                - Remoção de outliers acima do P99.9 ou ratio > 1.0
                - Eliminação de 0.1-1% dos dados mais extremos
                """)
            
            # Seção 5: Metodologia
            st.markdown("---")
            st.markdown("## 5. Metodologia de Machine Learning")
            
            st.markdown("### Pipeline de Treinamento")
            st.code("""
1. Preprocessamento: OneHotEncoder + StandardScaler
2. Rebalanceamento: SMOTE (targets binários)
3. Split Estratificado: 80/20 treino-teste
4. Validação Cruzada: StratifiedKFold (k=3)
5. Tuning: GridSearchCV
6. Avaliação: Métricas no conjunto de teste
            """, language="text")
            
            st.markdown("### Algoritmos Utilizados")
            df_algos = pd.DataFrame({
                'Algoritmo': ['Logistic Regression', 'Random Forest', 'XGBoost'],
                'Hiperparâmetros': ['C, penalty, solver', 'n_estimators, max_depth, min_samples', 'n_estimators, max_depth, learning_rate'],
                'Características': ['Baseline linear', 'Ensemble com feature importance', 'Gradient boosting otimizado']
            })
            st.dataframe(df_algos, use_container_width=True)
            
            # Seção 6: Resultados
            st.markdown("---")
            st.markdown("## 6. Resultados e Desempenho")
            
            df_melhores = pd.DataFrame([
                {
                    'Target': target.replace('_', ' ').title(),
                    'Algoritmo': info['modelo'],
                    'F1-Score': f"{info['f1_score']:.4f}",
                    'Accuracy': f"{info['accuracy']:.4f}"
                }
                for target, info in dados_doc['modelos']['melhores_modelos'].items()
            ])
            st.dataframe(df_melhores, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F1-Score Médio", f"{dados_doc['resumo_desempenho']['media_f1']:.3f}")
            with col2:
                st.metric("Accuracy Médio", f"{dados_doc['resumo_desempenho']['media_accuracy']:.3f}")
            with col3:
                st.metric("Melhor F1", f"{dados_doc['resumo_desempenho']['melhor_f1_geral']:.3f}")
            
            # Seção 7: Conclusões
            st.markdown("---")
            st.markdown("## 7. Conclusões e Trabalhos Futuros")
            
            st.success("""
            **Principais Conquistas:**
            - ✅ Sistema funcional de predição com dashboard interativo
            - ✅ Tratamento avançado de desbalanceamento (SMOTE + class_weight)
            - ✅ Múltiplas predições simultâneas (5 tipos de impacto)
            - ✅ Otimização de hiperparâmetros via GridSearchCV
            """)
            
            st.info("""
            **Melhorias Futuras:**
            - 🔄 Incorporação de features temporais e sazonalidade
            - 🗺️ Uso de coordenadas geográficas e topografia
            - 🤖 Ensemble avançado (stacking de modelos)
            - 🌐 Deploy em produção com API REST
            - 📡 Integração com dados meteorológicos em tempo real
            """)
            
        else:
            st.error("❌ Arquivo de documentação não encontrado!")
            st.info("""
            **Para gerar a documentação:**
            
            1. Execute: `python gerar_relatorio_tecnico.py`
            2. Execute: `python gerar_html_documentacao.py`
            3. Recarregue o dashboard
            """)

# ========================================
# EXECUTAR APLICAÇÃO
# ========================================
if __name__ == "__main__":
    main()
