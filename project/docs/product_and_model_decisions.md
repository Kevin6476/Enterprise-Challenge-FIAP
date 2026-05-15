# FIDC Analytics - Decisoes de Produto e Modelo

Este documento consolida as definicoes que devem ser mantidas no codigo, no dashboard, no PPT e no pitch.

## Problema final

Gestores e analistas de FIDC precisam identificar, de forma antecipada e priorizada, quais pagadores, cedentes e segmentos apresentam maior risco de inadimplencia ou deterioracao operacional, reduzindo dependencia de analises manuais e fragmentadas.

## Persona principal

Persona prioritaria do MVP:

- Gestor ou analista de risco de FIDC.

Usuarios secundarios:

- Area de produtos/dados da Nuclea.
- Consultorias e asset managers que acompanham carteiras de recebiveis.
- Times de compliance e governanca.

## Decisao que o MVP apoia

O MVP ajuda o usuario a responder:

- Onde esta o maior risco da carteira?
- Quais pagadores ou cedentes devem ser priorizados para analise?
- Quais regioes, CNAEs ou faixas de score concentram risco?
- O modelo consegue separar melhor boletos de maior e menor risco?

## Regra oficial de inadimplencia

Para o projeto, um boleto e considerado inadimplente quando:

```text
is_defaulted = 1 se dt_pagamento esta vazio
               ou se payment_delay_days > 0
```

Ou seja, qualquer pagamento feito apos o vencimento e tratado como evento de risco. Essa e uma regra conservadora e adequada para demonstrar governanca de risco em FIDC.

Observacao: o notebook de EDA possui uma classificacao preliminar com atraso maior que 5 dias. Para a entrega, vale a regra oficial implementada em `config/business_rules.py`.

## Escala oficial do score

A escala oficial do MVP e:

```text
0 a 100
```

Interpretacao:

| Faixa | Categoria | Uso operacional |
|---|---|---|
| 0 a 29.9 | low | Monitoramento normal |
| 30 a 59.9 | medium | Acompanhar sinais de deterioracao |
| 60 a 100 | high | Priorizar revisao da exposicao |

O pipeline armazena o score em escala 0 a 100. A interface visual do MVP apresenta esse mesmo score em escala 0 a 1000, multiplicando o valor por 10, para ficar fiel aos prototipos do produto. Portanto, `risk_score = 72.4` no codigo aparece como `724/1000` na tela.

## Threshold operacional de alerta

Alertas do MVP:

- Alto: `risk_score >= 60`
- Medio: `30 <= risk_score < 60`
- Baixo: `risk_score < 30`

Fila de priorizacao:

1. entidades em risco alto;
2. maior `avg_risk_score`;
3. maior exposicao financeira (`total_value`);
4. maior quantidade de boletos.

## Modelo escolhido

O modelo com melhor desempenho registrado foi o XGBoost:

| Modelo | AUC-ROC | AUC-PR | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8892 | 0.8149 | 0.7301 |
| Random Forest | 0.8851 | 0.8129 | 0.7241 |
| Logistic Regression | 0.7068 | 0.5491 | 0.4196 |

Justificativa:

- Melhor AUC-ROC entre os modelos testados.
- Melhor AUC-PR, importante para problema com classe desbalanceada.
- Melhor F1, indicando equilibrio entre precisao e recall.
- Boa aderencia a dados tabulares com relacoes nao lineares.

## Mitigacao de leakage

O projeto separa explicitamente colunas seguras e colunas com risco de leakage em `config/feature_catalog.py`.

Colunas excluidas do modelo incluem:

- resultado do pagamento (`dt_pagamento`, `vlr_baixa`, `tipo_baixa`);
- variaveis derivadas diretamente do atraso;
- agregacoes que poderiam usar informacao futura do mesmo lote.

Essa decisao torna o modelo mais defensavel para uma banca tecnica.

## Limitacoes honestas do MVP

- Base pequena: 7.118 boletos.
- Vencimentos concentrados em maio/2024.
- Split atual e aleatorio; em producao, o ideal e split temporal.
- Score atual e batch, nao tempo real.
- Dashboard atual e estatico/local; ainda falta publicacao com link publico.
- As fontes externas adicionais ainda nao foram integradas alem dos dados auxiliares disponiveis.

## Proximos passos recomendados

1. Validar pipeline completo em ambiente com todas as dependencias instaladas.
2. Publicar o dashboard em GitHub Pages, Streamlit Community Cloud ou ambiente similar.
3. Criar split temporal para simulacao mais realista.
4. Adicionar calibracao de probabilidades.
5. Criar alertas baseados em variacao de score ao longo do tempo.
6. Integrar novas fontes externas publicas, se houver tempo.
