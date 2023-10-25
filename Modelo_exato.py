#Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp

#Leitura de dados
df_data = pd.read_csv("antenas.txt", sep=" ", header=None)
df_data.drop(columns=[2,6],inplace=True)
df_data = df_data.set_axis(['id_veiculo', 't_inicial','x', 'y', 't_p'],axis=1)

#Cálculo do tempo total da rota
df_t_veiculo = df_data.groupby(['id_veiculo']).agg({'t_p':'sum'}).reset_index()
df_t_veiculo.rename(columns={'t_p':'t_v_total'},inplace=True)

#Cálculo do percentual de tempo do veículo em uma cédula
df_t_veiculo_posicao = df_data.groupby(['id_veiculo','x','y']).agg({'t_p':'sum'}).reset_index()
df_t_veiculo_posicao = df_t_veiculo_posicao.merge(df_t_veiculo, how='inner', on='id_veiculo')
df_t_veiculo_posicao['t_p_porcent'] = df_t_veiculo_posicao['t_p'] / df_t_veiculo_posicao['t_v_total']

#Determinação da posição p
df_t_veiculo_posicao['p'] = (df_t_veiculo_posicao['x'] * 100) + df_t_veiculo_posicao['y']

##########
#Parâmetros
##########
param_a = 0.5
param_b = 0.3

#Vepiculos
set_V = df_t_veiculo['id_veiculo'].to_numpy()

temp = df_t_veiculo_posicao.groupby(['p']).agg({'id_veiculo':'count'}).reset_index()
#Posições no grid
set_P = temp['p']

#Posições no grid que o veiculo ocupou
set_v_p = {}
for v in set_V:
    set_v_p[v] = [x for x in
                  df_t_veiculo_posicao.loc[df_t_veiculo_posicao['id_veiculo'] == v]['p'].to_list()]

df_t_veiculo_posicao['v_p'] = list(zip(df_t_veiculo_posicao['id_veiculo'],
                                       df_t_veiculo_posicao['p']))


temp = df_t_veiculo_posicao[['v_p','t_p_porcent']]

#Tempo que cada veiculo fica na posição
t_v_p = dict(temp.values)

##########
#Variáveis
##########
# Instalçao de antena na posição p
y_p_vars = {(p):
    pulp.LpVariable(cat='Integer',lowBound = 0,upBound = 1, name="y_v_{0}".format(p))
    for p in set_P}

#Se o veículo fica o tempo mínimo com comunicação
x_v_vars = {(v):
    pulp.LpVariable(cat='Integer', lowBound = 0,upBound = 1, name="x_v_{0}".format(v))
    for v in set_V}

##########
#Modelo
##########

### Create problem
opt_model = pulp.LpProblem(name="OPT_Linear_Model")

#minimiza a quantiddade instalada
objective = pulp.lpSum(y_p_vars[p]  for p in set_P)


### Constraints
# == constraints
# Verifica se o veiculo v fica com comunicação a
constraints_veiculo_conectado = {v: opt_model.addConstraint(
    pulp.LpConstraint(
        e=(x_v_vars[v] - pulp.lpSum(y_p_vars[p] * t_v_p[(v,p)] for p in set_v_p[v]) ) ,
        sense=pulp.LpConstraintLE,
        rhs=1 - param_a,
        name="const_conect_{0}".format(v)))
    for v in set_V}

#verifica  que pelo menos b por cento dos veículos tem comuniação
constraints_qtd_veiculo = { opt_model.addConstraint(
    pulp.LpConstraint(
        e=(pulp.lpSum(x_v_vars[v] for v in set_V)) ,
        sense=pulp.LpConstraintGE,
        rhs=param_b * len(set_V),
        name="const_quantidade"))
    }


# for minimization
opt_model.sense = pulp.LpMinimize
opt_model.setObjective(objective)

#Escreve o modelo
#opt_model.writeLP("modelo.lp")

#Parâmetros do solver e resolução do problema
#solver_list = pulp.listSolvers(onlyAvailable=True)
solver = pulp.get_solver('PULP_CBC_CMD',timeLimit=1500, gapRel=0.01)
opt_model.solve(solver)

#resultado das variáveis
name = []
values = []
for v in opt_model.variables():
    name.append(v.name)
    values.append(v.varValue)

df_var = pd.DataFrame(data={'var': name, 'value': values})
df_var['name'] = df_var['var'].str.split('_',n=0).str[0]
df_y = df_var.loc[(df_var['name'] == 'y') & (df_var['value'] > 0.5)]

df_y['p'] = df_y['var'].str.split('_',n=0).str[2]
df_y['p'] = pd.to_numeric(df_y['p'])
df_y['y'] = (df_y['p'] % 100)
df_y['x'] = (df_y['p'] - df_y['y'])/100


#Visualização dos dados
#Mapa de calor com tempo em cada posição no grid
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
plt.scatter(df_y['x'], df_y['y'], marker='v', c='red', s=5, label='RSU')

plt.legend()

plt.show()

