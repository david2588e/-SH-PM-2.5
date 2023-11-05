# Shanghai -PM2.5 study by SHAP and DoWhy 
Using machine learning algorithm (SHAP：SHapley Additive explanations ), the changing characteristics of air pollution and its meteorological interaction effects in Shanghai from 2014 to 2020 were analyzed based on daily data of six air pollutants and air quality.
The results indicate that: 
1) From 2014 to 2020, both the air quality index and pollutant averages showed improvement, suggesting that the environmental improvement measures in Shanghai have been effective and industrial adjustments are playing their role.
2) The level of PM2.5 is influenced by a complex interaction between anthropogenic pollutant emissions and meteorological conditions. Through big data analysis, the influencing factors in descending order are: mintempC (minimum temperature) > SO2 (sulfur dioxide level) > NO2 (nitrogen dioxide level) > WinddirDegree (wind direction) > O3 (ozone level) > clouldcover (cloud cover). In addition to the commonly known relationship between low temperature and haze with high PM2.5 levels, there are also complex relationships between certain wind directions and PM2.5.
3)  Using the causal analysis software "dowhy", the relationship between anthropogenic factors such as NO2 and PM2.5 was studied, and based on the available data and algorithms, it is likely that NO2 has a correlation rather than a causal relationship with PM2.5.
4) Based on historical data and algorithms, optimal levels of anthropogenic factors for controlling PM2.5 were proposed.

The conclusion is based on existing data from Shanghai area and final conclusion is highly subject tp geographical position.

This paves a new way to understand the theory to generation of the PM2.5 area by area.
