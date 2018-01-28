import NeuralNetwork as NN;
import numpy as np;
import datetime as dt
import cv2
import glob
import sys

#For XOR gate
if 0 :
	obj = NN.NeuralNetwork(2,1,2,1);
	inp=np.array([[0,0],[0,1],[1,0],[1,1]])
	op=np.array([[1],[0],[0],[1]])
	#comment following two lines if randomized weights are needed
	#obj.input_Weights=np.array([[-30,20,20],[10,-20,-20]])
	#obj.output_Weights=np.array([[-10,20,20]])
	io_list=[]
	for i in range(4):
		obj.Get_Input_Layer(inp[i].reshape(2,1))
		obj.Feed_Forward(op[i].reshape(1,1))
		obj.Backpropogation(op[i].reshape(1,1))
		io=NN.Input_Output(inp[i].reshape(2,1),op[i].reshape(1,1))
		io_list.append(io)
		print(obj.output_Layer[:,0])

	obj.Compute_DELTA(0,4)
	obj.Gradient_Checking(0.0001,0.0,4,io_list)	
	obj.Stochastic_Gradient_Descent(0.01,0,4,io_list,5)
	print("cost: ",obj.cost)
	for i in range(4):
		obj.Get_Input_Layer(inp[i].reshape(2,1))
		obj.Feed_Forward(op[i].reshape(1,1))
		print(obj.output_Layer[:,0])
		
#For digit recognition	
if 1 :
	fo = open("time.txt", "w")
	fo.write( str(dt.datetime.now()))
	fo.write("\n")
	fo.close()
	count=0
	io_list=[]
	alpha=.01
	lamda=0.63
	no_iter=1
	epsilon=0.0001
	GC=1
	for num in range(1):
		count=0
		for i in glob.glob('E:\Projects\ML\CharacterRecog\TRAINING_DATA\TRAINING_DATA/NUMBERS/'+str(num)+'\*'):

			if count<1:
				h=cv2.imread(i)
				b,g,r=cv2.split(h)
				input_vector=np.array(r).reshape(7990,1)
				for g in range(len(input_vector)):
					if input_vector[g]==255:
						input_vector[g]=1
				output_vector=np.zeros(3).reshape(3,1)
				output_vector[num,0]=1
				io=NN.Input_Output(input_vector,output_vector)
				io_list.append(io)
			count=count+1




	obj = NN.NeuralNetwork(7990,2,25,3);

	input_count=len(io_list)
	for i in range(input_count):
		print("Training example: ",i)
		obj.Train(io_list[i].input_Vector.reshape(7990,1),io_list[i].output_Vector.reshape(3,1))
	print("DELTA computation starts...")
	obj.Compute_DELTA(lamda,input_count)
	if GC!=0 :
		print("GC starts....")
		obj.Gradient_Checking(epsilon,lamda,input_count,io_list)
		
	else:
		print("GD starts....")
		obj.Stochastic_Gradient_Descent(alpha,lamda,input_count,io_list,no_iter)
		print("Training finished...!")
			
			# if((obj.output_Layer[:,0])<0.0001): 
				# print("0")
			# else:
				# print("1")
		fo = open("time.txt", "a")
		fo.write("\n")
		fo.write( str(dt.datetime.now()))
		fo.write("\n")
		fo.close()


		print("Come lets verify network....!")

		while 1:
			folder=input("Enter folder name...")
			img=input("Enter image name...")
			h=cv2.imread('E:\Projects\ML\CharacterRecog\TRAINING_DATA\TRAINING_DATA/NUMBERS/'+folder+'/'+img+'.png')
			b,g,r=cv2.split(h)
			input_vector=np.array(r).reshape(7990,1)
			obj.Get_Input_Layer(input_vector)
			output_vector=np.zeros(3).reshape(3,1)
			output_vector[int(folder),0]=1
			obj.Feed_Forward(output_vector)
			output_vector=obj.output_Layer[:,0].reshape(3,1)
			print(output_vector)
			# if(output_vector[int(folder),0]==1):
				# print("number is :",folder)
			ITER=input("TRY AGAIN...?")
			if ITER =='n':
				break;
			