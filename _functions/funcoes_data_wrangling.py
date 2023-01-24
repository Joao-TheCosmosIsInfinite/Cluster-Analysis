import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

def data_bin_cut(df, col, bins_number, color_bar="lightgreen"):
    """
    Função que gera uma tabela com a distribuição dos dados de uma variável numérica, e uma variável categórica através de faixas de valor
    
    :param df: Pandas dataframe
    :param col: Coluna do dataframe
    :param bins_number: Número de faixas
    
    :return: None
    """
    ## Tamanho do Dataframe
    tam = len(df)
    
    ## Criando os bins
    df_aux = pd.DataFrame(pd.cut(df[col], bins=bins_number, right=True).value_counts().sort_index()).reset_index()
    
    ## Renomeando a Coluna
    df_aux = df_aux.rename(columns={'index': 'data_bins'})
    
    ## Criando a coluna referente ao percentual
    df_aux["percent(%)"] = (df_aux[col]/ tam) * 100
    
    ## Criando um objeto styled e retornando
    return df_aux.style.hide_index().bar(subset=["percent(%)"], color='lightgreen')
    
def frequency_distribution(p_df_dataframe, p_column) -> pd.DataFrame():
    """
    Função que computa o as distribuições de frequencia absoluta e relativa de um dado dataframe e coluna

    Keyword arguments:
    :param p_df_dataframe: O pandas dataframe
    :param p_column: Coluna do dataframe

    :return: Um pandas dataframe com os dados das frequencias
    
    """
    ## Dataframe com os dados agrupados
    df_freq_dist = pd.DataFrame(p_df_dataframe \
                                   .groupby(p_column)[p_column] \
                                   .agg('count'))
    ## Renomeando a coluna para corresponder a frequencia absoluta
    df_freq_dist = df_freq_dist \
                        .rename(columns = {p_column : 'Frequencia Absoluta'}) \
                        .sort_values(by = 'Frequencia Absoluta', ascending = False)
    
    ## Somatorio das frequencias absolutas
    sum_columns = df_freq_dist['Frequencia Absoluta'].sum()
    
    ## Criação da frequencia relativa
    df_freq_dist['Frequencia Relativa'] = df_freq_dist \
                                                .groupby(level = 0) \
                                                .apply(lambda x : 100 * x/ sum_columns)
    
    return df_freq_dist