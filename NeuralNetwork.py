import numpy as np;
import math as m;
import random as rd

class NeuralNetwork:
	
	def __init__(self,no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes):
		""" 
		Initialization of a network
		Parameters :(no_input_nodes,no_hidden_layers,no_hidden_nodes,no_output_nodes)
		"""
		#random values for intial weights
		rand_start=-0.01
		rand_end=0.01
		
		#input output data set
		#self.io_Data_Set=io_data_set
		
		#Input nodes and layers
		self.no_Input_Nodes = no_input_nodes
		self.input_Layer=np.zeros((no_input_nodes,1))
		self.input_Weights=np.random.uniform(rand_start,rand_end,(no_hidden_nodes,no_input_nodes+1))
		#self.input_Weights=np.random.rand(no_hidden_nodes,no_input_nodes+1)
		
		#Hidden nodes and layers
		self.no_Hidden_Layers=no_hidden_layers
		self.no_Hidden_Nodes=no_hidden_nodes
		self.hidden_Layers= np.zeros((no_hidden_nodes,no_hidden_layers))
		if(no_hidden_layers>1):
			self.hidden_Weights=np.random.uniform(rand_start,rand_end,(no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1))
			#self.hidden_Weights=np.random.rand(no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1)
			
		#Output nodes and layers
		self.no_Output_Nodes=no_output_nodes
		self.output_Weights=np.random.uniform(rand_start,rand_end,(no_output_nodes,no_hidden_nodes+1))
		#self.output_Weights=np.random.rand(no_output_nodes,no_hidden_nodes+1)
		self.output_Layer=np.zeros((no_output_nodes,1))
		
		#cost function 
		self.cost_Without_Regularization=0
		self.cost=0
		
		#error
		self.output_Error=np.zeros((no_output_nodes,1))
		self.hidden_Error=np.zeros((no_hidden_nodes+1,no_hidden_layers))
		
		#delta terms
		self.input_Delta=np.zeros((no_hidden_nodes,no_input_nodes+1))
		if(no_hidden_layers>1):
			self.hidden_Delta=np.zeros((no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1))
		self.output_Delta=np.zeros((no_output_nodes,no_hidden_nodes+1))
		
		#DELTA terms
		self.input_DELTA=np.zeros((no_hidden_nodes,no_input_nodes+1))
		if(no_hidden_layers>1):
			self.hidden_DELTA=np.zeros((no_hidden_layers-1,no_hidden_nodes,no_hidden_nodes+1))
		self.output_DELTA=np.zeros((no_output_nodes,no_hidden_nodes+1))


		
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
		if(self.no_Hidden_Layers>1):
			for layer in range(self.no_Hidden_Layers):
				if(layer!=0):
					#adding bias node to each hidden layer
					hidden_vector=self.hidden_Layers[:,layer-1].reshape(self.no_Hidden_Nodes,1) #np.vstack((1,self.hidden_Layers[:,layer-1].reshape(self.no_Hidden_Nodes,1)))
					self.hidden_Layers[:,layer]=self.Compute_Next_Layer(self.hidden_Weights[layer-1,:,:],hidden_vector)
		
		
		
		#Output layer computation
		self.output_Layer[:,0]=self.Compute_Next_Layer(self.output_Weights,self.hidden_Layers[:,self.no_Hidden_Layers-1].reshape(self.no_Hidden_Nodes,1))
		
		

		#compute first part of cost function	
		self.Cost_Function_Without_Regularization(real_output_layer)
	
	
	def Cost_Function_Without_Regularization(self,real_output_layer):
		"""
		This function computes the cost function without regularization.
		That is, y*log(h(x)) + (1-y)*(1-log(h(x)))
		"""
		assert(np.size(self.output_Layer,0)==np.size(real_output_layer,0)),"Real and estimated output layers are not equal in size"
		for index in range(np.size(self.output_Layer,0)):
			#print(real_output_layer[index]," ",self.output_Layer[index],":", m.log10(self.output_Layer[index]))
			self.cost_Without_Regularization+=((real_output_layer[index].dot( m.log10(self.output_Layer[index]+(1e-10)))) + ((1-real_output_layer[index]).dot( m.log10(1-self.output_Layer[index]+(1e-10)))))
			
	def Regularization(self,lamda,no_of_input):
		"""
		This function computes the regularization part of the cost function of neural network
		Regularization to regularize the weights of all layers using lamda - the regularization term 
		lamda : regularization factor
		no_of_input : number of input to the neural network
		"""
		total_input_weight=np.sum(self.input_Weights[:,1:]**2)
		total_output_weight=np.sum(self.output_Weights[:,1:]**2)
		total_hidden_weight=0
		if(self.no_Hidden_Layers>1):
			total_hidden_weight=np.sum(self.hidden_Weights[:,1:]**2)
		total_weight = total_hidden_weight + total_input_weight + total_output_weight
		regularization_value=(lamda/(2*no_of_input))*total_weight
		
		return regularization_value
		
	def Cost_Function(self,lamda,no_of_input):
		"""
		cost function to compute J{theta}
		lamda : regularization factor
		no_of_input : number of input to the neural network
		"""
		
		self.cost= (-1/no_of_input)*( self.cost_Without_Regularization) + self.Regularization(lamda,no_of_input) 
		
	def Find_Error(self,real_output_layer):
		"""
		Find errors in each layer.
		Formula: δ(l)=((Θ(l))T δ(l+1)) .∗ a(l) .* (1−a(l))
		"""
		self.output_Error=self.output_Layer-real_output_layer
		
		# Error for last hidden layer. ( removed as it is wrong!---> [1:] is to remove bias from weights!) vstack to add bias.
		#Beware (np.vstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1])))) 1 minus might come outside
		#print(self.output_Weights[:,:].T.shape,"  ",self.output_Error.shape,"  ",np.hstack((1,self.hidden_Layers[:,self.no_Hidden_Layers-1])).shape,"  ",(np.hstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1]))).shape,"  ",(self.output_Weights[:,:].T.dot( self.output_Error)).shape,"  ",(np.hstack((1,self.hidden_Layers[:,self.no_Hidden_Layers-1])) * (np.hstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1])))).shape,"    ",((self.output_Weights[:,:].T.dot( self.output_Error)) * (np.hstack((1,self.hidden_Layers[:,self.no_Hidden_Layers-1])) * (np.hstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1]))))).shape)
		self.hidden_Error[:,self.no_Hidden_Layers-1]=(self.output_Weights[:,:].T.dot( self.output_Error)).reshape(self.no_Hidden_Nodes+1,) * (np.hstack((1,self.hidden_Layers[:,self.no_Hidden_Layers-1])) * (np.hstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1]))))
		# Error for remaining layer. ( [1:] is to remove bias from weights!) hstack to add bias. 
		#Beware (np.hstack((1,1-self.hidden_Layers[:,self.no_Hidden_Layers-1])))) 1 minus might come outside
		if(self.no_Hidden_Layers>1):
			for layer in range(self.no_Hidden_Layers-2, -1, -1):
				self.hidden_Error[:,layer]=(self.hidden_Weights[layer,:,:].T.dot( self.hidden_Error[1:,layer+1])) * (np.hstack((1,self.hidden_Layers[:,layer])) * (np.hstack((1,1-self.hidden_Layers[:,layer]))))
		
			
	def Backpropogation(self,real_output_layer):
		"""
		Backpropogate through each layer to find the error values .
		compute delta values using formula: for j!=0 DELTA= 1/m*Delta+ lamda*theta.	for j=0 DELTA= 1/m*Delta
		"""
		#initialize temporary delta terms
		temp_input_Delta=np.zeros((self.no_Hidden_Nodes,self.no_Input_Nodes+1))
		if(self.no_Hidden_Layers>1):
			temp_hidden_Delta=np.zeros((self.no_Hidden_Layers-1,self.no_Hidden_Nodes,self.no_Hidden_Nodes+1))
		temp_output_Delta=np.zeros((self.no_Output_Nodes,self.no_Hidden_Nodes+1))
		
		#find errors in each layer
		self.Find_Error(real_output_layer);
		
		#replace the error of bias term with one 
		self.hidden_Error[0,:]=1
		
		#compute temporary delta terms
		temp_input_Delta[:,:]=self.hidden_Error[1:,0].reshape(self.no_Hidden_Nodes,1).dot(np.hstack((1,self.input_Layer[:,0].T)).reshape(1,self.no_Input_Nodes+1))
		if(self.no_Hidden_Layers>1):
			for layer in range(self.no_Hidden_Layers-1):
				#reshape(1,self.no_Hidden_Nodes) instead of reshape(self.no_Hidden_Nodes,1).T to reduce computation
				temp_hidden_Delta[layer,:,:]=(self.hidden_Error[:,layer+1].reshape(self.no_Hidden_Nodes+1,1).dot(self.hidden_Layers[:,layer].reshape(1,self.no_Hidden_Nodes))).T
		temp_output_Delta=self.output_Error.dot(np.hstack((1,self.hidden_Layers[:,self.no_Hidden_Layers-1])).reshape(self.no_Hidden_Nodes+1,1).T)
		
		#compute accumulated Delta value
		self.input_Delta=self.input_Delta+temp_input_Delta
		if(self.no_Hidden_Layers>1):
			self.hidden_Delta=self.hidden_Delta+temp_hidden_Delta
		self.output_Delta=self.output_Delta+temp_output_Delta
	
		
	def Train(self,input_vector,real_output_layer):
		
		self.Get_Input_Layer(input_vector)
		self.Feed_Forward(real_output_layer)
		self.Backpropogation(real_output_layer)
		
		
	
	def Compute_DELTA(self,lamda,no_of_input):
		for i in range(self.no_Hidden_Nodes):
			for j in range(self.no_Input_Nodes+1):
				if j!=0 :
					self.input_DELTA[i,j]=((1/no_of_input)*self.input_Delta[i,j])+(lamda*self.input_Weights[i,j])					
				else:
					self.input_DELTA[i,j]=(1/no_of_input)*self.input_Delta[i,j]
		if(self.no_Hidden_Layers>1):		
			for layer in range(self.no_Hidden_Layers-1):
				for i in range(self.no_Hidden_Nodes):
					for j in range(self.no_Hidden_Nodes+1):
						if j!=0 :
							self.hidden_DELTA[layer,i,j]=((1/no_of_input)*self.hidden_Delta[layer,i,j])+(lamda*self.hidden_Weights[layer,i,j])							
						else:
							self.hidden_DELTA[layer,i,j]=(1/no_of_input)*self.hidden_Delta[layer,i,j]
						
		for i in range(self.no_Output_Nodes):
			for j in range(self.no_Hidden_Nodes+1):
				if j!=0 :
					self.output_DELTA[i,j]=((1/no_of_input)*self.output_Delta[i,j])+(lamda*self.output_Weights[i,j])					
				else:
					self.output_DELTA[i,j]=(1/no_of_input)*self.output_Delta[i,j]
	
		
	def Stochastic_Gradient_Descent(self,alpha,lamda,no_of_input,io_list,no_iter):
	
		self.Cost_Function(lamda,no_of_input)
		cost=self.cost
		self.cost_Without_Regularization=0
		print("epoch 0 : ",cost)
		rd.shuffle(io_list)
		iter=0
		while 1:
			
			for m in range(no_of_input):
				theta=self.Vectorize_Weights()
				delta_vector=self.Vectorize_DELTA()	
				cost=self.cost
				for j in range(theta.size):
					theta[j]=theta[j]-(alpha*delta_vector[j])
				self.Convert_To_Weights(theta)
				input_vector=io_list[m].input_Vector.reshape(self.no_Input_Nodes,1)
				output_vector=io_list[m].output_Vector.reshape(self.no_Output_Nodes,1)
				self.Train(input_vector,output_vector)
				self.Compute_DELTA(lamda,no_of_input)
				
				self.Cost_Function(lamda,no_of_input)
				self.cost_Without_Regularization=0
				
				print("iter ",iter,"  epoch ",m,": ",self.cost)
				# if(cost<self.cost):
					# self.Convert_To_Weights(self.Vectorize_DELTA())
					# break;
				
			iter=iter+1
			if(iter==no_iter):
				self.Convert_To_Weights(self.Vectorize_DELTA())
				break;
	
	
	def Vectorize_Weights(self):
		if(self.no_Hidden_Layers>1):
			theta=np.append(np.append(self.input_Weights,self.hidden_Weights),self.output_Weights)
		else:
			theta=np.append(self.input_Weights,self.output_Weights)
		return theta

	def Vectorize_DELTA(self):
		if(self.no_Hidden_Layers>1):
			delta_vector=np.append(np.append(self.input_DELTA,self.hidden_DELTA),self.output_DELTA)
		else:
			delta_vector=np.append(self.input_DELTA,self.output_DELTA)
		return delta_vector
		
	def Convert_To_Weights(self,theta):
		self.input_Weights[:]=theta[:self.input_Weights.size].reshape(self.no_Hidden_Nodes,self.no_Input_Nodes+1)
		if(self.no_Hidden_Layers>1):
			self.hidden_Weights[:]=theta[self.input_Weights.size:(self.input_Weights.size+self.hidden_Weights.size)].reshape(self.no_Hidden_Layers-1,self.no_Hidden_Nodes,self.no_Hidden_Nodes+1)
			self.output_Weights[:]=theta[(self.input_Weights.size+self.hidden_Weights.size):].reshape(self.no_Output_Nodes,self.no_Hidden_Nodes+1)
		else:
			self.output_Weights[:]=theta[self.input_Weights.size:].reshape(self.no_Output_Nodes,self.no_Hidden_Nodes+1)

	######################## not completed ########################
	def Gradient_Checking(self,epsilon,lamda,no_of_input,io_list):
		
		theta=self.Vectorize_Weights()
		delta_vector=self.Vectorize_DELTA()	
		self.grad_Check=np.zeros((theta.size))
		
		pluscost=0
		minuscost=0
		for n in range(len(theta)):
			thetaplus=theta.copy()
			thetaminus=theta.copy()
			thetaplus[n]=thetaplus[n]+epsilon
			thetaminus[n]=thetaminus[n]-epsilon
			self.Convert_To_Weights(thetaplus)
			
			for m in range(no_of_input):
				self.Feed_Forward(io_list[m].output_Vector)
			self.Cost_Function(lamda,no_of_input)
			pluscost=self.cost
			self.cost_Without_Regularization=0
			self.Convert_To_Weights(thetaminus)
			for m in range(no_of_input):
				self.Feed_Forward(io_list[m].output_Vector)
			self.Cost_Function(lamda,no_of_input)
			minuscost=self.cost
			self.cost_Without_Regularization=0
			self.grad_Check[n]=(pluscost-minuscost)/(2*epsilon)
			
			print("theta: ",theta[n],"  grad_Check: ",self.grad_Check[n],"  DELTA: ",delta_vector[n],"pluscost: ",pluscost,"minuscost: ",minuscost)
		
			
			
				
				
class Input_Output:
    def __init__(self,input_vector,output_vector):
        self.input_Vector=input_vector
        self.output_Vector=output_vector
		

