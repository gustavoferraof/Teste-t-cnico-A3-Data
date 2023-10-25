#Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Leitura de dados
df_data = pd.read_csv("antenas.txt", sep=" ", header=None)
df_data.drop(columns=[2,6],inplace=True)
df_data = df_data.set_axis(['id_veiculo', 't_inicial','x', 'y', 't_p'],axis=1)

#Cálculo do tempo total da rota e da quantidade de veículos
df_t_veiculo = df_data.groupby(['id_veiculo']).agg({'t_p':'sum'}).reset_index()
df_t_veiculo.rename(columns={'t_p':'t_v_total'},inplace=True)
qtd_veiculos = df_t_veiculo.shape[0]

#Cálculo do percentual de tempo do veículo em uma cédula
df_t_veiculo_posicao = df_data.groupby(['id_veiculo','x','y']).agg({'t_p':'sum'}).reset_index()
df_t_veiculo_posicao = df_t_veiculo_posicao.merge(df_t_veiculo, how='inner', on='id_veiculo')
df_t_veiculo_posicao['t_p_porcent'] = df_t_veiculo_posicao['t_p'] / df_t_veiculo_posicao['t_v_total']

#Determinação da posição p
df_t_veiculo_posicao['p'] = (df_t_veiculo_posicao['x'] * 100) + df_t_veiculo_posicao['y']

##########
#Parâmetros
##########
parametros_a = []
parametros_b = []
qtd_antenas = []
antenas = []
veiculos_atend = []

#Funções uteis
#Proucura a melhor posição para iserir um novo RSU
def search_new_location(df_t_veiculo_posicao, pos_sel, li_v_atend):
    #Filtra posições vazias
    df_base_not_sel = df_t_veiculo_posicao.loc[~(df_t_veiculo_posicao['id_veiculo'].isin(li_v_atend)) &
                                               ~(df_t_veiculo_posicao['p'].isin(pos_sel))]

    # Tempo restante do veiculo
    df_t_veiculo = df_base_not_sel.groupby(['id_veiculo']).agg({'t_p': 'sum'}).reset_index()
    df_t_veiculo.rename(columns={'t_p': 't_v_total'}, inplace=True)

    df_base_not_sel.drop(columns=['t_v_total'], inplace=True)
    df_base_not_sel = df_base_not_sel.merge(df_t_veiculo, how='inner', on='id_veiculo')
    df_base_not_sel['t_p_porcent'] = df_base_not_sel['t_p'] / df_base_not_sel['t_v_total']

    #Métricas por posição
    df_p_t = df_base_not_sel.groupby(['p']).agg({'t_p': 'sum', 'id_veiculo': 'nunique',
                                                 't_p_porcent': 'sum'}).reset_index()
    #Ordena e escolhe a melhor posição
    df_p_t = df_p_t.sort_values(['t_p_porcent'], ascending=[False])
    pos_sel.append(df_p_t['p'].iloc[0])

#Atualiza a lista de veículos que atendem o requisto de tempo mínimo de comunicação
def update_atending_list(df_t_veiculo_posicao, pos_sel):
    #Filtra as posições selecionadas
    df_base_sel = df_t_veiculo_posicao.loc[df_t_veiculo_posicao['p'].isin(pos_sel)]
    df_base_sel = df_base_sel.groupby('id_veiculo').agg({'t_p_porcent': 'sum'}).reset_index()
    #Seleciona os veículos que atendem o tempo mínimo de atendimento pelas seleções posicionadas
    li_v_atend = df_base_sel.loc[df_base_sel['t_p_porcent'] >= param_a]['id_veiculo'].to_list()
    return li_v_atend

#Busca local para tirar antenas não necessaria no atendimento dos requisitos mínimos
def busca_local_posicao_desnecessaria(df_t_veiculo_posicao, df_p_t_sel, pos_sel, li_v_atend):
    posicao = []
    veiculos_restantes = []

    #llop nas posições
    for pos in pos_sel:
        posicoes_restantes = pos_sel.copy()
        posicoes_restantes.remove(pos)
        #Verificação se a lista continuaria com o minimo necessário
        li_atend_rest = update_atending_list(df_t_veiculo_posicao, posicoes_restantes)

        posicao.append(pos)
        veiculos_restantes.append(len(li_atend_rest))

    #Escolha da antena na posicção que menos afeta a solução
    df_restantes = pd.DataFrame(data={'p': posicao,
                                      'veiculos_restantes': veiculos_restantes})

    df_restantes = df_restantes.merge(df_p_t_sel, how='left', on='p')
    df_restantes.sort_values(['veiculos_restantes', 't_p_porcent', 'id_veiculo', 't_p'],
                             ascending=[False, True, True, True], inplace=True)
    # Se alguma solução continuar atendendo os requisitos, atualizar
    if df_restantes['veiculos_restantes'].iloc[0] >= qtd_min_veiculos:
        pos_sel.remove(df_restantes['p'].iloc[0])
        li_v_atend = update_atending_list(df_t_veiculo_posicao, pos_sel)
        return pos_sel, li_v_atend, True
    else:
        return pos_sel, li_v_atend, False


#loop pelos parâmetros
for a in [0.3,0.5,0.7]:
    for b in [0.3,0.5,0.7]:
        param_a = a
        param_b = b


        qtd_min_veiculos = qtd_veiculos*b
        pos_sel = []
        li_v_atend = []

        #Loop até que a quantida mínima de veículos atenda os requisitos solicitados
        while(len(li_v_atend) < qtd_min_veiculos):
            search_new_location(df_t_veiculo_posicao, pos_sel, li_v_atend)
            li_v_atend = update_atending_list(df_t_veiculo_posicao, pos_sel)


        ############## Busca local
        df_base_sel = df_t_veiculo_posicao.loc[df_t_veiculo_posicao['p'].isin(pos_sel)]
        df_p_t_sel = df_base_sel.groupby(['p']).agg({'t_p': 'sum', 'id_veiculo': 'nunique',
                                                     't_p_porcent': 'sum'}).reset_index()

        improvement=True
        while(improvement == True):
            pos_sel, li_v_atend, improvement =  busca_local_posicao_desnecessaria(df_t_veiculo_posicao,
                                                                                 df_p_t_sel, pos_sel, li_v_atend)


        #Salvar os resultados
        parametros_a.append(param_a)
        parametros_b.append(param_b)
        qtd_antenas.append((len(pos_sel)))
        antenas.append(pos_sel)
        veiculos_atend.append(li_v_atend)


df_resultados = pd.DataFrame(data={'param_a':parametros_a,
                                   'param_b':parametros_b,
                                   'qtd_antenas':qtd_antenas,
                                   'antenas':antenas,
                                   'veiculos_atend':veiculos_atend,})

df_antenas = pd.DataFrame(data={'p':antenas[0]})

df_antenas['y'] = (df_antenas['p'] % 100)
df_antenas['x'] = (df_antenas['p'] - df_antenas['y'])/100


### Visualizações
df_data_vis = df_data.groupby(['x','y']).agg({'t_p':'sum'}).reset_index()

# Crie uma matriz vazia para armazenar os valores no grid
grid = np.zeros((100, 100))

# Preencha a matriz com os valores
for _, row in df_data_vis.iterrows():
    grid[int(row['x'])][int(row['y'])] += row['t_p']

# Cria um mapa de calor usando o matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(grid, cmap='viridis', interpolation='none', origin='lower', extent=[0, 100, 0, 100])
plt.colorbar(label='Tempo na célula (x,y)')
plt.title('Quantidade de tempo total dos veículos monitarados')
plt.xlabel('Posição X')
plt.ylabel('Posição Y')

# Plote antenas
plt.scatter(df_antenas['x'], df_antenas['y'], marker='v', c='red', s=5, label='RSU')

plt.legend()

plt.show()


