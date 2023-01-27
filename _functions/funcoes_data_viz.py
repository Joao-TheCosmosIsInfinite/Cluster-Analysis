import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')


def boxplot(df_dataframe, ax, column, palette='viridis', title='', title_size=14, title_color='dimgrey') -> None:
    """
    Função que gera um boxplot de acordo com os dados passados
    :param df_dataframe: Pandas dataframe
    :param ax: Matplotlib axes
    :param column: Coluna do dataframe
    :param p_pallete: Paleta de cores
    :return: None
    """
    ## Dataframe com as estatisticas descritivas
    df_summary = pd.DataFrame(df_dataframe[column].describe())
    
    ## Setando variáveis
    df_summary.loc["skewness"] = df_dataframe[column].skew()    
    df_summary.loc["kurtosis"] = df_dataframe[column].kurt()
    
    ## Objeto referente ao boxplot
    sns.boxplot(data=df_dataframe[column].values,
                palette=palette,
                orient='h',
                ax=ax)
    
    ## deixando os x_ticks com valor em branco
    ax.set(xticklabels = [])
    ax.set(ylabel = None)
    
    ## Titulo do boxplot com o nome da coluna
    ax.set_title(title, size=title_size, color=title_color)
    
    ## Tabela que será gerada junto com o gráfico, onde tera as estatisticas
    statistics_table = plt.table( cellText = df_summary.values,
                                    rowLabels = df_summary.index,
                                    colLabels =  ' ',
                                    cellLoc = 'left', 
                                    rowLoc = 'center',            
                                    loc ='bottom')
    
    ## Tamanho da fonte da tabela
    statistics_table.set_fontsize(14)
    
    ## Escala da tabela
    statistics_table.scale(1, 1.4)
    
    ## Colocar a tabela debaixo do boxplot
    plt.subplots_adjust(left = 0.2, bottom = .1)
    
    ## Exibir figura
    plt.show()
    
def bar_count_plot(df, ax, x=None, y=None, hue=None, order=None, palette='husl', 
                   title='', title_size=14, title_color='dimgrey'):
    """
    Função que gera um gráfico em barras a partir da volumetria de uma unica variável categórica
    :param df: Pandas dataframe
    :param ax: Matplotlib axes
    :param x: Coluna do dataframe no eixo x
    :param y: Coluna do dataframe no eixo y
    :param hue: Variavel opcional para incluir no plot
    :param pallete: Paleta de cores
    :param title: Titulo do gráfico
    :param title_size: Tamanho do título
    :param title_color: Cor do titulo
    :return: None
    """

    ## Verificando plotagem por quebra de alguma variável categórica
    ncount = len(df)
    
    ## Validando demais argumentos e plotando gráfico
    if order is None:
        sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, hue=hue)
    else:
        if x:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, hue=hue, order = df[x].value_counts().index)  
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, hue=hue, order = df[y].value_counts().index)
    
     
    ## Formatando eixos, removendo borda e afins
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')
    
    ## Titulo do grafico
    ax.set_title(title+'\n', size=title_size, color=title_color, loc='center')
    
    ## Inserindo os rotulos referentes aos números e aos percentuais    
    if x:
        ## Retirar os labels referentes ao eixo y
        ax.set_ylabel('')
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{}\n{:.2f}%'.format(int(y), 100. * y / ncount), 
                        (x.mean(), y), 
                        ha='center', 
                        va='bottom', 
                        color='black')
    else:
        ## Retirar os labels referentes ao eixo y
        ax.set_xlabel('')
        for p in ax.patches:
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            ax.annotate(' {} ( {:.2f}% )'.format(int(x), 100. * x / ncount), 
                        (x, y.mean()), 
                        va='center', 
                        color='black')
    
def pie_plot(df, col, ax, label_names=None, text='',colors=['#ff9999','#66b3ff'], circle_radius=0.72, 
               title='',title_size=14,title_color='dimgrey', label_size=18):
    """
    Função que gera um gráfico de rosca de um variavel de um dado dataframe/ dataset
    :param df: Pandas dataframe
    :param col: Coluna do dataframe
    :param ax: Matplotlib axes
    :param label_names: Nome das labels referentes a cada fatia
    :param text: Texto que sera inserido no centro do grafico
    :param colors: Lista com as cores de cada fatia do grafico
    :param title: Titulo do gráfico
    :param title_size: Tamanho do título
    :param title_color: Cor do titulo
    :return: None
    """
    ## Função interna que gera uma string com os dados referentes ao valor 
    ## numerico e percentual em relação ao total, de cada fatia do plot
    def autopct_values_and_percents(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))

        return '{:.2f}%\n({})'.format(pct, val)
    
    ## Retorno dos valores e definição da figura
    values = df[col].value_counts().values
    label_names = label_names if label_names is not None else df[col].value_counts().index

    ## Plotando gráfico de rosca
    center_circle = plt.Circle((0, 0), circle_radius, color='white')
    ## Realizando um explode nas fatias plotadas
    explode = [0.05] * len(values)
    ## Ajustando configuraçoes do gráfico
    ax.pie(values, 
           labels=label_names, 
           colors=colors, 
           autopct=lambda pct: autopct_values_and_percents(pct),
           startangle=90,
           pctdistance=0.85,
           explode=explode,
           textprops={'fontsize': label_size})    
    ax.add_artist(center_circle)
    ## Titulo do grafico
    ax.set_title(title+'\n', size=title_size, color=title_color)
    
    ## Incluindo um texto dentro do plot
    kwargs = dict(size=(16*circle_radius), va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    
    ## Ajustando gráfico
    ax.axis('equal')  
    plt.tight_layout()
    
def bar_plot(df, ax, x=None, y=None, hue=None, palette='husl',
             orient_horizontal=True, title='', title_size=14, 
             title_color='dimgrey'):
    """
    Função que gera um gráfico em barras a partir dos dados passados
    
    :param df: Pandas dataframe
    :param ax: Matplotlib axes
    :param x: Coluna do dataframe no eixo x
    :param y: Coluna do dataframe no eixo y
    :param hue: Variavel opcional para incluir no plot
    :param pallete: Paleta de cores
    :param title: Titulo do gráfico
    :param title_size: Tamanho do título
    :param title_color: Cor do titulo
    
    :return: None
    """

    ## Verificando plotagem por quebra de alguma variável categórica
    if pd.api.types.is_string_dtype(df[y].dtype):
        ncount = df[y].count()
    else:
        ncount = df[y].sum()
    
    ## Validando demais argumentos e plotando gráfico
    plot = sns.barplot(x=x, y=y, data=df, palette=palette, ax=ax)
    
     
    ## Formatando eixos, removendo borda e afins
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')
    
    ## Titulo do grafico
    ax.set_title(title+'\n', size=title_size, color=title_color, loc='center')
    
    ## Rotacionando o eixo x
    ax.set_xticklabels(ax.get_xticklabels(), 
                   ha = 'right',
                   rotation = 35,
                   fontsize = 10)
                   
    ## Inserindo os rotulos referentes aos números
    if orient_horizontal:
        for p in plot.patches:        
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.2f}\n{:.2f}%'.format(int(y), 100. * y / ncount), 
                        (x.mean(), y), 
                        ha='center', 
                        va='bottom', 
                        color='black')
    else:
        ncount=df[x].sum()
        for p in plot.patches:        
                x = p.get_bbox().get_points()[1, 0]
                y = p.get_bbox().get_points()[:, 1]
                ax.annotate('{:.2f} - {:.2f}%'.format(int(x), 100. * x / ncount), 
                            (x, y.mean()),   
                            va='center', 
                            color='black')

def heatmap_plot(df_corr, ax, palette='coolwarm', title='', title_size=14, title_color='dimgrey'):
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    ## Ajustando mascara
    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:,:-1].copy()
    ## Paleta de Cor
    cmap = palette
    ## Mapa de Calos
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                linewidths=5, 
                cmap=cmap, 
                vmin=-1, 
                vmax=1, 
                cbar_kws={"shrink": .8}, 
                square=True)
    ## Objetos do gráfico
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks)
    ax.set_xticklabels(ax.get_xticklabels(), 
                       ha = 'right',
                       rotation = 35,
                       fontsize = 11)
    ## Titulo do grafico
    ax.set_title(title+'\n', size=title_size, color=title_color, loc='center')
