import numpy as np
import pandas as pd

from ml_utils import(create_regression_data,r2_score,std_scaler_mat,re_std_scaler)
from typing import Any

class MR_MLP:
    def __init__(self,input_dim:int,hidden_dim1:int,hidden_dim2:int,output_dim:int,learning_rate:float=0.01):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.lr = learning_rate
        # Kaiming He 权重初始化
        self.w1 = np.random.randn(self.input_dim,self.hidden_dim1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1,self.hidden_dim1))
        self.w2 = np.random.randn(self.hidden_dim1,self.hidden_dim2)* np.sqrt(2.0 / hidden_dim1)
        self.b2 = np.zeros((1,self.hidden_dim2))
        self.w3 = np.random.randn(self.hidden_dim2,self.output_dim)* np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros((1,self.output_dim))


    def __relu__(self,x:np.ndarray):
        return np.maximum(x,0)
    
    def __grad_relu__(self,x:np.ndarray):
        return np.where(x > 0,1.0,0.0)
    
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
        # self.ret = self.__softmax__(self.z3)
        self.ret = self.z3
        return self.ret
    
    def loss(self,y_true:np.ndarray):
        return np.mean((y_true - self.ret)**2)
    
    # 输入y_true shape = (batch_size, output_dim)
    # 隐藏层w shape = (hidden_dim, output_dim)
    def backward(self,x:np.ndarray,y_true:np.ndarray):
        batch_size = x.shape[0]
        dz3 = (self.ret - y_true) * 2 / batch_size
        self.dw3 = self.a2.T @ dz3
        self.db3 = dz3.sum(axis=0,keepdims=True)

        dz2 = (dz3 @ self.w3.T) * self.__grad_relu__(self.z2)
        self.dw2 = self.a1.T @ dz2
        self.db2 = dz2.sum(axis = 0,keepdims = True)

        dz1 = (dz2 @ self.w2.T) * self.__grad_relu__(self.z1)
        self.dw1 = x.T @ dz1
        self.db1 = dz1.sum(axis = 0,keepdims = True)

        self.w3 -= self.lr * self.dw3
        self.b3 -= self.lr * self.db3
        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1


def train_model(model:MR_MLP,X:np.ndarray,Y:np.ndarray,epochs:int)->MR_MLP:
    print("训练开始...")
    
    x_scaled, _, _ = std_scaler_mat(X)
    y_scaled, y_std, y_mean = std_scaler_mat(Y)
    best_r2:float = 0
    tag_epoch = 0

    for epoch in range(epochs + 1):
        y_pred = model.forward(x_scaled)
        cur_loss = model.loss(y_scaled)
        
        y_true_ori = re_std_scaler(y_scaled, y_std, y_mean)
        y_pred_ori = re_std_scaler(y_pred, y_std, y_mean)
        cur_r2 = r2_score(y_true_ori, y_pred_ori)
        if cur_r2 > 0 and cur_r2 > best_r2:
            tag_epoch = epoch
            best_r2 = cur_r2
        
        if cur_r2 >0 and cur_r2 <= best_r2 and epoch - tag_epoch >= 100:
            break
        
        model.backward(x_scaled, y_scaled)

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {cur_loss:.4f} | r2:{cur_r2:.4f}")

    return model

def mlp_liner_test(input_dim = 5,hidden1 = 16,hidden2 = 8,output_dim = 1,lr = 0.01,epochs = 15000):
    X, Y = create_regression_data(samples=10000, input_dim=input_dim)
    y_sclaed,y_std,y_mean = std_scaler_mat(Y)

    org_model = MR_MLP(input_dim, hidden1, hidden2, output_dim, lr)
    ret_model = train_model(org_model,X,Y,epochs)

    print("\n==== 测试结果 ====")
    test_x = np.random.randn(1, input_dim)
    pred_y = re_std_scaler(ret_model.forward(test_x),y_std,y_mean)
    true_y = test_x @ np.array([[2.5], [-1.3], [0.8], [3.1], [-0.5]])
    print(f"输入: {test_x.round(3)}")
    print(f"预测输出: {pred_y[0,0]:.4f}")
    print(f"真实输出: {true_y[0,0]:.4f}")



def mlp_lpq_data_test():
    df = pd.read_excel("/Users/hmr/Desktop/AI/cs336/MachineLearning/lpq_data.xlsx")
    X = df.iloc[:, :5].values
    Y = df.iloc[:, 5:].values

    model = MR_MLP(
        input_dim=5,
        hidden_dim1=64,
        hidden_dim2=32,
        output_dim=8,
        learning_rate=0.001
    )
    
    model = train_model(model, X, Y, epochs=20000)
    return model

if __name__ == "__main__":
    mlp_lpq_data_test()

