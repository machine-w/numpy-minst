import numpy as np

dimensions=[28*28,10]
distribution=[
    {'b':[0,0]},
    {'b':[0,0],'w':[-1,1]},
]

def init_parameters_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters():
    parameter=[]
    
    for i in range(len(distribution)):
        layer_parameter={}
        for j in distribution[i].keys():
            if j=='b':
                layer_parameter['b']=init_parameters_b(i)
                continue
            if j=='w':
                layer_parameter['w']=init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    print(parameter)
    return parameter

def init_parameters_simple():
     parameter=[]
     parameter.append({'b':np.zeros(28*28)})
     parameter.append({'b':np.zeros(10),'w':np.random.rand(28*28,10)})
     return parameter

if __name__ == '__main__':
    print(init_parameters_simple())
