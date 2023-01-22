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