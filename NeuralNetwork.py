import numpy as np;

class NeuralNetwork:
	
	def __init__(self,no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes):
		""" 
		Initialization of a network
		Parameters :(no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes)
		"""
		#Input nodes and layers
		self.no_Input_Nodes = no_input_nodes
		self.input_Layer=np.zeros((no_input_nodes,1))
		self.input_Weights=np.zeros((no_hidden_nodes,no_input_nodes+1))
		
		#Hidden nodes and layers
		self.no_Hidden_layers=no_hidden_layers
		self.no_Hidden_nodes=no_hidden_nodes
		self.hidden_Layers= np.zeros((no_hidden_nodes,no_hidden_layers))
		if(no_hidden_layers>1):
			self.hidden_Weights=np.zeros((no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1))
		else:
			self.hidden_Weights=np.zeros((no_hidden_layers,no_hidden_nodes,no_hidden_nodes+1))
			
		#Output nodes and layers
		self.no_Output_Nodes=no_output_nodes
		self.output_Weights=np.zeros((no_output_nodes,no_hidden_nodes+1))
		self.output_Layer=np.zeros((no_output_nodes,1))
		
		
	
	
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
		nodes=np.vstack((1,nodes))
		assert(np.size(weights,0)!=np.size(nodes,1)),"Number of rows in Weight and Number of columns in nodes doesn't match "
		z_vector=weights.dot(nodes)
		return self.Sigmoid_Function(z_vector)
	
	def Sigmoid_Function(self,z_vector):
		""" Activation function which uses the formula 1/(1+e^(-z)) """
		return (1/(1+np.exp(-z_vector)))
	
	
	
	def Feed_Forward(self):
		"""
		computes the activation function of the next layer.
		"""
		#first hidden layer computation from input layer
		self.hidden_Layers[:,0]=Compute_Next_Layer(self.input_Weights,self.input_Layer)
		
		#All hidden layer computation 
		if(self.no_Hidden_layers>1):
			for layer in range(self.no_Hidden_layers):
				if(layer!=0):
					self.hidden_Layers[:,layer]=Compute_Next_Layer(self.hidden_Weights[layer-1,:,:],self.hidden_Layers[:,layer-1])
		
		#Output layer computation
		self.output_Layer[:,0]=Compute_Next_Layer(self.output_Weights,self.hidden_Layers[:,self.no_Hidden_layers-1])
					
		
	
	
	
	
	
	
	
	
	
	
	