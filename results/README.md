# Pasta de Resultados

Esta pasta armazena os resultados das simulações.

## Conteúdo

Os resultados das simulações são salvos aqui em diferentes formatos:
- CSV - Histórico de dados (velocidade, posição, tempo)
- JSON - Resumos de simulação
- Excel - Relatórios completos
- PNG/PDF - Gráficos e visualizações

## Organização

Recomenda-se organizar os resultados por:
- Data da simulação
- Tipo de pista
- Configuração do veículo

## Exemplo de Estrutura

```
results/
├── README.md
├── 2024-01-15_pista_interlagos/
│   ├── historico.csv
│   ├── resumo.json
│   └── graficos/
├── 2024-01-16_pista_curitiba/
│   └── ...
```

## Nota

Por padrão, arquivos de resultados não são versionados no Git (veja .gitignore).
