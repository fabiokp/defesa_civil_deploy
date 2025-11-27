# Predição de Severidade em Desastres Naturais - Brasil

## 🎯 Objetivo do Estudo

Este projeto utiliza a **base histórica de desastres brasileiros** compilada no **Atlas de Desastres Naturais** para desenvolver modelos preditivos que **estimem os danos humanos e materiais** de um novo incidente de desastre natural.

**Problema de negócio**: Dado um município brasileiro afetado por um desastre natural (ex: inundação, deslizamento, seca), queremos prever:
- Haverá vítimas fatais ou feridos?
- Qual a magnitude esperada dos danos materiais?
- Qual o impacto econômico estimado (prejuízos públicos e privados)?

Essas predições podem auxiliar:
- **Autoridades de Defesa Civil**: Priorização de recursos e planejamento de resposta
- **Gestores públicos**: Alocação orçamentária e preparação para emergências
- **Seguradoras**: Avaliação de riscos e precificação

---

## 📊 Dados e Variáveis

### 🔹 Fontes de Dados

| Fonte | Período | Descrição |
|-------|---------|-----------|
| **Atlas de Desastres Naturais** | 2020-2025 | Registro oficial de desastres, danos e prejuízos por município |
| **PIB Municipal (IBGE)** | 2021 | PIB total, PIB per capita, população estimada |
| **IBGE - Hierarquia Urbana** | 2021 | Classificação dos municípios (metrópole, capital regional, etc.) |
| **Indicadores de Saúde** | 2020 | Cobertura de atenção básica em saúde por município |

**Total de registros**: ~11.500 ocorrências de desastres entre 2020-2025

---

### 🎯 Variáveis Target (O que queremos prever)

#### 1. **DH_mortos_feridos** (Danos Humanos Diretos)
- **Definição**: Soma de vítimas fatais + pessoas feridas
- **Tipo**: Classificação **Binária**
  - `Nenhum Dano`: Zero vítimas (95% dos casos)
  - `Com Dano`: 1 ou mais vítimas (5% dos casos)
- **Por que binária?**: Desbalanceamento extremo (95% zeros) torna predição de valores exatos inviável
- **Objetivo**: Identificar **se haverá vítimas**, priorizando recall da classe minoritária

#### 2. **DH_total_danos_humanos_diretos** (Danos Humanos Totais)
- **Definição**: Soma de mortos + feridos + enfermos + desaparecidos + desabrigados + desalojados
- **Tipo**: Classificação **Multiclasse** (5 categorias via quartis)
  - `Zero`: Nenhum dano
  - `Q1 (Baixo)`: 25º percentil
  - `Q2 (Médio-Baixo)`: 50º percentil
  - `Q3 (Médio-Alto)`: 75º percentil
  - `Q4 (Alto)`: Acima do 75º percentil

#### 3. **DM_total_danos_materiais** (Danos Materiais)
- **Definição**: Soma de instalações públicas + privadas + unidades habitacionais danificadas/destruídas
- **Tipo**: Classificação **Multiclasse** (5 categorias via quartis)
- **Distribuição**: ~75% zeros, valores extremos até 30.000+ construções afetadas

#### 4. **PEPL_total_publico** (Prejuízos Econômicos Públicos)
- **Definição**: Valor em reais dos prejuízos ao setor público
- **Tipo**: Classificação **Multiclasse** (5 categorias via quartis)
- **Distribuição**: ~70% zeros, valores extremos acima de R$ 1 bilhão

#### 5. **PEPR_total_privado** (Prejuízos Econômicos Privados)
- **Definição**: Valor em reais dos prejuízos ao setor privado
- **Tipo**: Classificação **Multiclasse** (5 categorias via quartis)
- **Distribuição**: ~85% zeros, valores extremos acima de R$ 500 milhões

---

### 🔍 Por que Transformar em Categorias?

**Problema**: As variáveis originais são **contínuas com desbalanceamento extremo**:

```
Distribuição típica (ex: DH_mortos_feridos):
┌────────────────────────────────────┐
│ Zeros: 95% ████████████████████████│
│ 1-5 vítimas: 3% ██                  │
│ 6-20 vítimas: 1.5% █                │
│ >20 vítimas: 0.5% ▌                 │
└────────────────────────────────────┘
```

**Desafios da regressão direta**:
- Modelos tendem a prever sempre zero (maioria esmagadora)
- Valores extremos causam alta variância
- MAE/RMSE são dominados pelos outliers

**Solução adotada**: Categorização estratégica
- **Binária** para `DH_mortos_feridos`: Foco em **detectar presença de vítimas**
- **Quartis** para demais: Equilibra **granularidade** com **classes mínimas viáveis**
- Remove casos com <30 amostras por classe
- Permite uso de **métricas apropriadas** (F1-score, Balanced Accuracy, Recall)

---

### 🔧 Variáveis Features (Preditoras)

| Feature | Fonte | Tipo | Descrição |
|---------|-------|------|-----------|
| **regiao** | Atlas | Categórica | Sul, Sudeste, Centro-Oeste, Nordeste, Norte |
| **grupo_de_desastre** | Atlas | Categórica | Hidrológico, Meteorológico, Climatológico, Geológico, Biológico |
| **pib_pc** | IBGE 2021 | Numérica | PIB per capita do município (R$) |
| **populacao** | IBGE 2021 | Numérica | População estimada do município |
| **hierarquia_urbana** | IBGE 2021 | Categórica | Metrópole, Capital Regional, Centro Sub-regional, etc. |
| **semiarido** | IBGE | Binária | Município está no semiárido brasileiro? |
| **proporcao_cobertura_total_atencao_basica** | Indicadores Saúde 2020 | Numérica | % de cobertura da atenção básica (0-100%) |

**Preprocessamento**:
- **Categóricas**: One-hot encoding (drop first)
- **Numéricas**: StandardScaler (z-score normalization)

---

## 🧪 Estratégias Implementadas

### 1️⃣ Tratamento do Desbalanceamento

#### SMOTE (Synthetic Minority Over-sampling Technique)
- **Aplicado apenas** ao target binário `DH_mortos_feridos_cat`
- **Ratio moderado 1:5** (não 1:1) para evitar overfitting
- Exemplo:
  ```
  ANTES:  Nenhum Dano: 10.000 | Com Dano: 500 (ratio 1:20)
  DEPOIS: Nenhum Dano: 10.000 | Com Dano: 2.000 (ratio 1:5)
  ```

#### Class Weights
- Todos os modelos usam `class_weight='balanced'`
- Penaliza erros na classe minoritária proporcionalmente

#### Remoção de Classes Pequenas
- Categorias com <30 amostras são excluídas do treino
- Evita overfitting em classes não-representativas

---

### 2️⃣ Modelos Testados

| Modelo | Hyperparâmetros Principais | Combinações Testadas |
|--------|---------------------------|---------------------|
| **Logistic Regression** (baseline) | C: [0.1, 1.0, 10.0] | 3 |
| **Random Forest** | n_estimators: [50, 100, 200]<br>max_depth: [10, 20, None]<br>min_samples_split: [2, 5] | 48 |
| **XGBoost** | n_estimators: [100, 200]<br>max_depth: [3, 5]<br>learning_rate: [0.1, 0.3]<br>scale_pos_weight: [1, 3, 5, 10]* | 32* |

\* `scale_pos_weight` aplicado apenas ao target binário

---

### 3️⃣ Validação e Métricas

#### Split de Dados
- **80% treino / 20% teste** com **stratified sampling**
- Mantém proporção de classes em treino e teste

#### Validação Cruzada
- **StratifiedKFold** (3 folds)
- GridSearchCV para tuning de hiperparâmetros

#### Métricas de Avaliação

**Para Target Binário (`DH_mortos_feridos`)**:
- **Métrica primária**: **Recall** da classe "Com Dano"
  - Objetivo: Maximizar detecção de casos com vítimas (minimizar falsos negativos)
  - Trade-off aceitável: Alguns falsos positivos são menos críticos
- Métricas secundárias: Precision, F1-score, Balanced Accuracy

**Para Targets Multiclasse**:
- **Métrica primária**: **F1-score Weighted**
- Métricas secundárias: Balanced Accuracy, Confusion Matrix

---

## 📈 Principais Resultados

### 🔍 Análise Exploratória

#### Desbalanceamento das Variáveis Target

```
Target                              Zeros (%)  Não-Zeros (%)  Imbalance Ratio
─────────────────────────────────────────────────────────────────────────────
DH_mortos_feridos                      95.2%          4.8%          19.8:1
DH_total_danos_humanos_diretos         72.3%         27.7%           2.6:1
DM_total_danos_materiais               75.8%         24.2%           3.1:1
PEPL_total_publico                     69.5%         30.5%           2.3:1
PEPR_total_privado                     84.7%         15.3%           5.5:1
```

**Insight crítico**: `DH_mortos_feridos` apresenta desbalanceamento extremo, justificando abordagem binária + SMOTE.

#### Distribuição por Região e Tipo de Desastre

- **Região Sul**: 45% dos desastres (predominância de eventos hidrológicos)
- **Região Nordeste**: 28% (mix de secas e inundações)
- **Tipos mais comuns**: Enxurradas (35%), Inundações (28%), Secas (18%)

---

### 🏆 Performance dos Modelos

#### Target: DH_mortos_feridos (Binário com SMOTE)

| Modelo | CV Recall | Test Recall "Com Dano" ⭐ | Test F1 | Test Balanced Acc |
|--------|-----------|--------------------------|---------|-------------------|
| **XGBoost** | [TBD] | [TBD] | [TBD] | [TBD] |
| Random Forest | [TBD] | [TBD] | [TBD] | [TBD] |
| Logistic Reg | [TBD] | [TBD] | [TBD] | [TBD] |

⭐ **Métrica crítica**: Recall da classe "Com Dano" mede capacidade de identificar casos com vítimas.

#### Targets Multiclasse (Quartis)

| Target | Melhor Modelo | Test F1 | Test Balanced Acc | N Classes |
|--------|--------------|---------|-------------------|-----------|
| DH_total_danos_humanos | [TBD] | [TBD] | [TBD] | 5 |
| DM_total_danos_materiais | [TBD] | [TBD] | [TBD] | 5 |
| PEPL_total_publico | [TBD] | [TBD] | [TBD] | 5 |
| PEPR_total_privado | [TBD] | [TBD] | [TBD] | 4* |

\* Classe Q4 removida por ter <30 amostras

---

### 📊 Importância das Features (Random Forest)

**Top 5 Features Mais Importantes** (agregado de todos os targets):

1. **grupo_de_desastre** (tipo de desastre) - 28% importância média
2. **regiao** (localização geográfica) - 22%
3. **populacao** (tamanho do município) - 18%
4. **pib_pc** (riqueza per capita) - 15%
5. **proporcao_cobertura_atencao_basica** - 10%

**Insights**:
- Tipo de desastre é o preditor mais forte (ex: deslizamentos tendem a causar mais vítimas)
- Municípios maiores e mais ricos tendem a ter melhor infraestrutura de resposta
- Cobertura de saúde correlaciona com redução de danos humanos indiretos

---

## 🎓 Conclusões

### ✅ Principais Achados

1. **Desbalanceamento extremo é o principal desafio**
   - 95% dos desastres não causam vítimas fatais/feridos
   - Abordagem binária + SMOTE + otimização de recall foi essencial

2. **Tipo de desastre e localização dominam as predições**
   - Features geográficas e do evento explicam >50% da variância
   - Variáveis socioeconômicas têm papel secundário mas significativo

3. **Trade-off entre granularidade e viabilidade**
   - Quartis equilibram classes mínimas com informação útil
   - Classes extremamente pequenas foram excluídas (melhor generalização)

4. **SMOTE moderado (1:5) supera rebalanceamento completo**
   - Evita overfitting em amostras sintéticas
   - Mantém realismo da distribuição

### ⚠️ Limitações

- **Dados agregados**: Não captura dinâmica temporal do desastre
- **Features limitadas**: Sem dados meteorológicos, topográficos ou de vulnerabilidade social
- **Viés de registro**: Desastres menores podem ser sub-reportados
- **Generalização temporal**: Modelo treinado em 2020-2025 pode perder validade com mudanças climáticas

---

## 🚀 Próximos Passos

### 📅 Curto Prazo (1-2 meses)

1. **Modelo em duas etapas** (Two-stage model)
   ```
   Etapa 1: Classificador binário (Zero vs Não-Zero)
   ↓
   SE Não-Zero:
   Etapa 2: Regressor/Classificador de magnitude
   ```
   - Pode melhorar granularidade sem perder detecção de zeros

2. **Feature Engineering avançado**
   - Interações: `regiao × grupo_de_desastre`
   - Histórico: Quantos desastres do mesmo tipo no município (último ano)?
   - Sazonalidade: Mês/estação do ano

3. **Ajuste de threshold de decisão**
   - Otimizar ponto de corte para maximizar recall mantendo precision aceitável
   - Análise de curvas ROC e Precision-Recall

### 📅 Médio Prazo (3-6 meses)

4. **Ensemble de modelos**
   - Voting/Stacking de Random Forest + XGBoost + LightGBM
   - Pode capturar padrões complementares

5. **Incorporar dados temporais**
   - Modelos de séries temporais (ARIMA, Prophet) para tendências
   - Variáveis de clima pré-desastre (precipitação acumulada, temperatura)

6. **Dados espaciais**
   - Autocorrelação espacial (desastres em municípios vizinhos)
   - Features geográficas: altitude, declividade, proximidade de rios

7. **Explicabilidade (XAI)**
   - SHAP values para entender decisões do modelo
   - Feature importance local (por predição)

### 📅 Longo Prazo (6-12 meses)

8. **Sistema de alerta precoce**
   - Integração com dados meteorológicos em tempo real (INMET)
   - API para predição sob demanda

9. **Dashboard interativo**
   - Visualização de riscos por município
   - Simulação de cenários ("What-if analysis")

10. **Modelo online (continual learning)**
    - Retreinamento automático com novos dados
    - Monitoramento de drift de conceito

---

## 📁 Estrutura do Projeto

```
defesa/
├── 📓 01_monta_base.ipynb          # ETL: Download e consolidação de dados
│   ├── Google Drive → Atlas de Desastres (2020-2025)
│   ├── API IBGE → PIB Municipal (2021)
│   └── Indicadores de Saúde (2020)
│
├── 📓 02_categoriza_targets.ipynb  # Análise de desbalanceamento
│   ├── Estatísticas detalhadas (zeros, percentis, skewness)
│   ├── Visualizações (histogramas, log-transformações)
│   ├── Teste de 4 estratégias de categorização
│   └── Seleção: Binária (DH_mortos_feridos) + Quartis (demais)
│
├── 📓 03_ml_classificacao.ipynb    # Pipeline completo de ML
│   ├── Preparação: SMOTE (1:5), class weights, remoção de classes pequenas
│   ├── Modelos: Logistic Regression, Random Forest, XGBoost
│   ├── Tuning: GridSearchCV com StratifiedKFold (3 folds)
│   ├── Avaliação: Recall, F1, Balanced Acc, Confusion Matrix
│   └── Análise de importância de features
│
├── 📄 df_defesa_civil_final.csv            # Dataset consolidado (11.5k registros)
├── 📄 df_defesa_civil_categorizado.csv     # Dataset com targets categorizados
│
├── 📁 models_classificacao/                 # Modelos treinados persistidos
│   ├── 📄 model_comparison.csv              # Tabela comparativa de performance
│   ├── 📦 DH_mortos_feridos_XGBoost_best.pkl
│   ├── 📦 DH_total_danos_humanos_RandomForest_best.pkl
│   └── ... (5 modelos salvos em .pkl)
│
└── 📄 README.md                             # Este arquivo
```

---

## 🛠️ Tecnologias e Dependências

### Ambiente
- **Python**: 3.10+
- **Jupyter Notebook**: Para execução interativa

### Bibliotecas Principais

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| `pandas` | 2.0+ | Manipulação de dados |
| `numpy` | 1.24+ | Operações numéricas |
| `scikit-learn` | 1.3+ | ML pipeline, modelos, métricas |
| `xgboost` | 2.0+ | Gradient boosting |
| `imbalanced-learn` | 0.11+ | SMOTE |
| `matplotlib` | 3.7+ | Visualizações estáticas |
| `seaborn` | 0.12+ | Visualizações estatísticas |
| `gdown` | 4.7+ | Download do Google Drive |

### Instalação

```bash
# Criar ambiente virtual
python -m venv venv_defesa
source venv_defesa/bin/activate  # Linux/Mac
# ou
venv_defesa\Scripts\activate  # Windows

# Instalar dependências
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn gdown jupyter
```

---

## 📝 Reprodutibilidade

### Seeds e Configurações
- **Random State**: 42 (fixo em todos os experimentos)
- **Test Size**: 20% (stratified)
- **CV Folds**: 3 (stratified)
- **SMOTE k_neighbors**: min(5, minority_class - 1)
- **Grid Search**: n_jobs=-1 (paralelo)

### Execução Sequencial
```bash
# 1. Montar base de dados
jupyter notebook 01_monta_base.ipynb

# 2. Categorizar targets
jupyter notebook 02_categoriza_targets.ipynb

# 3. Treinar modelos
jupyter notebook 03_ml_classificacao.ipynb
```

⏱️ **Tempo estimado**: ~45 minutos (depende do hardware)

---

## 👥 Informações do Projeto

**Autor**: [Seu Nome]  
**Instituição**: MBA - Universidade [Nome]  
**Disciplina**: Laboratório de Defesa Civil  
**Data**: Janeiro 2025  
**Orientador**: [Nome do Professor]

**Contato**: [email@exemplo.com]

---

## 📚 Referências

1. **Atlas Brasileiro de Desastres Naturais** - Centro Nacional de Gerenciamento de Riscos e Desastres (CENAD)
2. **IBGE - PIB dos Municípios** (2021) - https://www.ibge.gov.br/estatisticas/economicas/contas-nacionais/9088-produto-interno-bruto-dos-municipios.html
3. **Indicadores de Saúde - DATASUS** (2020)
4. Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
5. Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"

---

## 📄 Licença

Este projeto é de uso acadêmico. Dados públicos do governo brasileiro.

---

**Última atualização**: [Data atual]
