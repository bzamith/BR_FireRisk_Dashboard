# BR Fire Risk Dashboard

## About
Project for UFPE's Data Visualization Classes - 2024.2
- **Authors**: Arthur de Andrade Almeida, Bruna Zamith Santos, Sheyla Lima Leal de Souza
- **Professor**: Nivan Roberto Ferreira Júnior
- **Department**: CIn UFPE

## Motivation
The Brazilian territory, with its vast expanse and diverse biomes, faces serious threats from wildfires, closely linked to meteorological conditions. Biomes such as the Amazon, Cerrado, Caatinga, Atlantic Forest, Pampas, and Pantanal are frequently impacted by fires that can have devastating effects on biodiversity and natural resources. One of the main threats to the preservation of Brazilian biomes is the occurrence of wildfires. These fires, often exacerbated by a lack of rainfall during certain times of the year, are a recurring concern.

Although there are methodologies to calculate wildfire risk based on climate variables such as temperature, humidity, and precipitation, Brazil still lacks an integrated platform to centralize this information for effective forecasting and monitoring. Studies like that of Soriano, Daniel, and Santos (2015), which analyzed five indices for the Pantanal, including FMA, FMA+, Telicyn, Angström, and Nesterov, and that of Torres and Ribeiro (2008), which applied various indices in Juiz de Fora (MG), demonstrate the usefulness of these indices. Among them, the Nesterov and Angström indices, which consider humidity and temperature, are particularly effective for predicting fires in different regions of Brazil.

## Goals
The solution our group aims to develop is a data visualization platform for forecasting fire risk across the Brazilian territory. We will conduct an analysis of meteorological data to classify the risk of wildfire occurrence, contributing to fire prevention and control. The platform will utilize the Nesterov and Angström fire risk indices, which are widely used in wildfire risk analysis and have proven effective in studies applied in Brazil.

## Notes
Please note that while the repository information and code will be in English, the dashboard will be in Brazilian Portuguese, as our primary focus is on Brazilian audience.

## Technologies
- Python 3
- SQL
- Tableau

## Related Work
[B. Z. Santos, B. M. Araujo Soriano, M. G. Narciso, D. F. Silva and R. Cerri, "A New Time Series Framework for Forest Fire Risk Forecasting and Classification," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-8, doi: 10.1109/IJCNN54540.2023.10191502](https://ieeexplore.ieee.org/document/10191502)

## Data
We employ 2 different data sources:

### 1) INMET's Climate Data
INMET, Brazil's National Institute of Meteorology (from Portuguese: *Instituto Nacional de Meteorologia*), provides daily climate data from various meteorological stations across the country.

The data used in this project was extracted from their [official website](https://portal.inmet.gov.br/dadoshistoricos) and covers the period from January 2001 to August 2024.

### 2) INPE's Fire Spot Data
INPE, Brazil's National Institute for Space Research (from Portuguese: *Instituto Nacional de Pesquisas Espaciais*), provides daily and monthly hotspot detection data based on satellite images.

The data used in this project was extracted from their official website: [daily](https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/) and [monthly](https://terrabrasilis.dpi.inpe.br/queimadas/situacao-atual/estatisticas/estatisticas_estados/), covering the period from January 2023 to August 2024.

## Install
```
python3 -m virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip3 install -r requirements.txt
```


