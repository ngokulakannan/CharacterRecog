import numpy as np;
import math as m;

class NeuralNetwork:
	
	def __init__(self,no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes):
		""" 
		Initialization of a network
		Parameters :(no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes)
		"""
		#Input nodes and layers
		self.no_Input_Nodes = no_input_nodes
		self.input_Layer=np.zeros((no_input_nodes,1))
		self.input_Weights=np.random.randint(-25,25,size=(no_hidden_nodes,no_input_nodes+1))
		
		#Hidden nodes and layers
		self.no_Hidden_layers=no_hidden_layers
		self.no_Hidden_nodes=no_hidden_nodes
		self.hidden_Layers= np.zeros((no_hidden_nodes,no_hidden_layers))
		if(no_hidden_layers>1):
			self.hidden_Weights=np.random.randint(-25,25,size=(no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1))
			
		#Output nodes and layers
		self.no_Output_Nodes=no_output_nodes
		self.output_Weights=np.random.randint(-25,25,size=(no_output_nodes,no_hidden_nodes+1))
		self.output_Layer=np.zeros((no_output_nodes,1))
		
		#cost function 
		self.Cost_Without_Regularization=0
		self.Cost=0
	
	
		
	def Get_Input_Layer(self,input_vector):
		"""
		Get input layer vector and stores it in input_Layer variable
		"""
		self.input_Layer=input_vector.copy()
	
	
	def Compute_Next_Layer(self,weights,nodes):
		"""
		Get weight matrix and node vector. Add bias node to input vector. 
		check dimensions are correct and get the z_vector by matrix multiplication
		return the sigmoid function's output
		"""
		#Adding bias node 
		nodes=np.vstack((1,nodes))
		assert(np.size(weights,1)==np.size(nodes,0)),"Number of rows in Weight and Number of columns in nodes doesn't match "
		
		z_vector=weights.dot(nodes)
		return self.Sigmoid_Function(z_vector)[:,0] 
	
	def Sigmoid_Function(self,z_vector):
		""" Activation function which uses the formula 1/(1+e^(-z)) """
		return (1/(1+np.exp(-z_vector)))
	
	
	
	def Feed_Forward(self,real_output_layer):
		"""
		computes the activation function of the next layer.
		"""
		
		#first hidden layer computation from input layer		
		self.hidden_Layers[:,0]=self.Compute_Next_Layer(self.input_Weights,self.input_Layer)
		
		
		#All hidden layer computation 
		if(self.no_Hidden_layers>1):
			for layer in range(self.no_Hidden_layers):
				if(layer!=0):
					#adding bias node to each hidden layer
					hidden_vector=self.hidden_Layers[:,layer-1].reshape(self.no_Hidden_nodes,1) #np.vstack((1,self.hidden_Layers[:,layer-1].reshape(self.no_Hidden_nodes,1)))
					self.hidden_Layers[:,layer]=self.Compute_Next_Layer(self.hidden_Weights[layer-1,:,:],hidden_vector)
		
		
		
		#Output layer computation
		self.output_Layer[:,0]=self.Compute_Next_Layer(self.output_Weights,self.hidden_Layers[:,self.no_Hidden_layers-1].reshape(self.no_Hidden_nodes,1))
		

		#compute first part of cost function	
		self.Cost_Function_Without_Regularization(real_output_layer)
	
	
	def Cost_Function_Without_Regularization(self,real_output_layer):
		"""
		This function computes the cost function without regularization.
		That is, y*log(h(x)) + (1-y)*(1-log(h(x)))
		"""
		assert(np.size(output_Layer,0)!=np.size(real_output_layer,0)),"Real and estimated output layers are not equal in size"
		for index in range(np.size(output_Layer,0)):
			self.Cost_Without_Regularization+=((real_Output_Layer[index] * m.log10(output_Layer[index])) + ((1-real_Output_Layer[index]) * m.log10(1-output_Layer[index])))
			
	def Regularization(self,lamda,no_of_input):
		"""
		This function computes the regularization part of the cost function of neural network
		Regularization to regularize the weights of all layers using lamda - the regularization term 
		lamda : regularization factor
		no_of_input : number of input to the neural network
		"""
		total_input_weight=np.sum(self.input_Weights**2)
		total_output_weight=np.sum(self.output_Weights**2)
		total_hidden_weight=0
		if(no_Hidden_layers>1):
		total_hidden_weight=np.sum(self.hidden_Weights**2)
		total_weight = total_hidden_weight + total_input_weight + total_output_weight
		Regularization_value=(lamda/(2*no_of_input))*total_weight
		return Regularization_value
		
	def Cost_Function(self,lamda,no_of_input):
		"""
		cost function to compute J{theta}
		lamda : regularization factor
		no_of_input : number of input to the neural network
		"""
		self.Cost= (-1/no_of_input) * self.Cost_Without_Regularization + Regularization(lamda,no_of_input) 
		
		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	