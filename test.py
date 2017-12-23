import NeuralNetwork as NN;
import numpy as np;

obj = NN.NeuralNetwork(2,1,2,1);
inp=np.array([[0,0],[0,1],[1,0],[1,1]])

#comment following two lines if randomized weights are needed
obj.input_Weights=np.array([[-30,20,20],[10,-20,-20]])
obj.output_Weights=np.array([[-10,20,20]])

for i in range(4):
	obj.Get_Input_Layer(inp[i,:].reshape(2,1))
	obj.Feed_Forward()
	if((obj.output_Layer[:,0])<0.0001): 
		print("0")
	else:
		print("1")
