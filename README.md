# BR Fire Risk Dashboard

## About
Project for UFPE's Data Visualization Classes - 2024.2
- **Author**: Bruna Zamith Santos
- **Professor**: Nivan Roberto Ferreira Júnior
- **Department**: CIn UFPE

## Motivation
**TBD**

## Goals
**TBD**

## Notes
Please note that while the repository information and code will be in English, the dashboard will be in Brazilian Portuguese, as our primary focus is on Brazilian audience.

## Technologies
- Python 3
- SQL
- Tableau

## Related Work
[B. Z. Santos, B. M. Araujo Soriano, M. G. Narciso, D. F. Silva and R. Cerri, "A New Time Series Framework for Forest Fire Risk Forecasting and Classification," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-8, doi: 10.1109/IJCNN54540.2023.10191502](https://ieeexplore.ieee.org/document/10191502)

## Data
We employ 4 different data sources:

### 1) INMET's Climate Data
INMET, Brazil's National Institute of Meteorology (from Portuguese: *Instituto Nacional de Meteorologia*), provides daily climate data from various meteorological stations across the country.

The data used in this project was extracted from their [official website](https://portal.inmet.gov.br/dadoshistoricos) and covers the period from January 2001 to August 2024.

### 2) INPE's Fire Spot Data
INPE, Brazil's National Institute for Space Research (from Portuguese: *Instituto Nacional de Pesquisas Espaciais*), provides daily and monthly hotspot detection data based on satellite images.

The data used in this project was extracted from their official website: [daily](https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/) and [monthly](https://terrabrasilis.dpi.inpe.br/queimadas/situacao-atual/estatisticas/estatisticas_estados/), covering the period from January 2023 to August 2024.

### 3) Ministry of the Environment
Brazil's Ministry of the Environment provides data on monthly financial investments in various environmental initiatives.

The data used in this project was extracted from their official website and covers the period from 2011 to 2024.

## Install
```
python3 -m virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip3 install -r requirements.txt
```


