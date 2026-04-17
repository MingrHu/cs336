import numpy as np

class MR_MLP:
    def __init__(self,input_dim:int,hidden_dim1:int,hidden_dim2:int,output_dim:int,learning_rate:float=0.01):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.lr = learning_rate

        self.w1 = np.random.randn(self.input_dim,self.hidden_dim1)
        self.b1 = np.zeros((1,self.hidden_dim1))
        self.w2 = np.random.randn(self.hidden_dim1,self.hidden_dim2)
        self.b2 = np.zeros((1,self.hidden_dim2))
        self.w3 = np.random.randn(self.hidden_dim2,self.output_dim)
        self.b3 = np.zeros((1,self.output_dim))


    def __relu__(self,x:np.ndarray):
        return np.maximum(x,0)
    
    def __sigmoid__(self,x:np.ndarray):
        return 1/(1+np.exp(-x))
    
    def __tanh__(self,x:np.ndarray):
        return np.tanh(x)
    
    def __softmax__(self,x:np.ndarray):
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

    # 输入x shape = (batch_size, input_dim)
    def forward(self,x:np.ndarray):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = self.__relu__(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.__relu__(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.ret = self.__softmax__(self.z3)
        return self.ret
    
    # 输入y_true shape = (batch_size, output_dim)
    def backward(self,x:np.ndarray,y_true:np.ndarray):
        dz3 = self.ret.copy()
        batch_size = x.shape[0]
        dz3[np.arange(batch_size),y_true] -= 1
        dz3 /= batch_size

        self.dw3 = self.a2.T @ dz3
        self.db3 = dz3.sum(axis=0,keepdims=True)


