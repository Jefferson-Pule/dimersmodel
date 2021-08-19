import tensorflow as tf
import numpy as np
import math
import logging
import time
import sys
import os

#Hamiltonian parameters
T=0.1
J=1

#RNN parameters
ns=1
epochs=1
nh=80
lr=0.001
seed=1234

class VariationalMonteCarlo(tf.keras.Model):

    # Constructor
    def __init__(self, Lx, Ly, J, 
                 T, num_hidden, learning_rate,
                 epochs, seed=1234):
        
        super(VariationalMonteCarlo, self).__init__()

        """ PARAMETERS """
        self.Lx       = Lx                           # Size along x
        self.Ly       = Ly                           # Size along y
        self.J        = -J                           # Strenght
        self.T        = T                            # Temperature 
        self.N        = 2*Lx*Ly                      # Number of edges
        self.nh       = num_hidden                   # Number of hidden units in the RNN
        self.seed     = seed                         # Seed of random number generator
        self.epochs   = epochs                       # Training epochs 
        self.K        = 2               # Dimension of the local Hilbert space
        self.expected =int(0.5*Lx*Ly)
        
        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate, epsilon=1e-8)

        # Build the model RNN
        # RNN layer: N -> nh
        self.rnn = tf.keras.layers.GRU(self.nh, kernel_initializer='glorot_uniform',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.001),
                                       return_sequences = True,
                                       return_state = True,
                                       stateful = False)

        # Dense layer: nh - > K
        self.dense = tf.keras.layers.Dense(self.K, activation = tf.nn.softmax,
                                           kernel_regularizer = tf.keras.regularizers.l2(0.001))

        # lattice 
        self.buildlattice()

    #Gives coordinates of the x,y horizontal edge
    def coord_h(self,x,y):
        return 2*self.Lx*y+2*x
    #Gives coordinates of the x,y vertical edge
    def coord_v(self,x,y):
        return 2*self.Lx*y+1+2*x

    def buildlattice(self):
        self.horizontal = []
        self.vertical = []
        self.adjacent=[]
        self.important=[]

        # Horizontal Pairs
        for y in range(self.Ly):
            #Periodic Last row
            if y==self.Ly-1:
                for x in range(self.Lx):
                    self.horizontal.append([self.coord_h(x,y), self.coord_h(x,0)])
            
            # Information for check        
                    self.adjacent.append([self.coord_h(x,y), self.coord_v(x,y)])
                    self.adjacent.append([self.coord_h(x,0), self.coord_v(x,y)])
                    
                    #Periodic first column 
                    if x==0:
                        self.adjacent.append([self.coord_h(x,0), self.coord_v(self.Lx-1,y)])
                        self.adjacent.append([self.coord_h(x,y), self.coord_v(self.Lx-1,y)])
                    
                    else:
                        self.adjacent.append([self.coord_h(x,0), self.coord_v(x-1,y)])
                        self.adjacent.append([self.coord_h(x,y), self.coord_v(x-1,y)])
            else:
                for x in range(self.Lx):
                    self.horizontal.append([self.coord_h(x,y), self.coord_h(x+self.Lx,y)])
        
                    #Information for check 
                    self.adjacent.append([self.coord_h(x,y), self.coord_v(x,y)])
                    self.adjacent.append([self.coord_h(x,y+1), self.coord_v(x,y)])
                    
                    #Periodic first column
                    if x==0:
                        self.adjacent.append([self.coord_h(x,y), self.coord_v(self.Lx-1,y)])
                        self.adjacent.append([self.coord_h(x,y+1), self.coord_v(self.Lx-1,y)])
                    
                    else:
                        self.adjacent.append([self.coord_h(x,y), self.coord_v(x-1,y)])
                        self.adjacent.append([self.coord_h(x,y+1), self.coord_v(x-1,y)])

        #Vertical Pairs
        for x in range(self.Lx):
            #Periodic last column 
            if x==self.Lx-1:
                for y in range(self.Ly):
                    self.vertical.append([self.coord_v(x,y), self.coord_v(0,y)])
                    
            else:
                for y in range(self.Ly):
                    self.vertical.append([self.coord_v(x,y), self.coord_v(x+1,y)])
        # Important Pairs
        for y in range(self.Ly):
            for x in range(self.Lx):
                if y==0:
                    if x==0:
                        self.important.append([self.coord_h(x,y),self.coord_v(self.Lx-1,self.Ly-1),self.coord_v(x,self.Ly-1),self.coord_h(self.Lx-1,y),self.coord_v(self.Lx-1,y), self.coord_v(x,y),self.coord_h(x+1,y)])
                    elif x==self.Lx-1:
                        self.important.append([self.coord_h(x,y),self.coord_v(x-1,self.Ly-1),self.coord_v(x,self.Ly-1),self.coord_h(x-1,y),self.coord_v(x-1,y), self.coord_v(x,y),self.coord_h(0,y)])
                    else:
                        self.important.append([self.coord_h(x,y),self.coord_v(x-1,self.Ly-1),self.coord_v(x,self.Ly-1),self.coord_h(x-1,y),self.coord_v(x-1,y), self.coord_v(x,y),self.coord_h(x+1,y)])
                else:
                    if x==0:
                        self.important.append([self.coord_h(x,y),self.coord_v(self.Lx-1,y-1),self.coord_v(x,y-1),self.coord_h(self.Lx-1,y),self.coord_v(self.Lx-1,y), self.coord_v(x,y),self.coord_h(x+1,y)])
                    elif x==self.Lx-1:
                        self.important.append([self.coord_h(x,y),self.coord_v(x-1,y-1),self.coord_v(x,y-1),self.coord_h(x-1,y),self.coord_v(x-1,y), self.coord_v(x,y),self.coord_h(0,y)])   
                    else:
                        self.important.append([self.coord_h(x,y),self.coord_v(x-1,y-1),self.coord_v(x,y-1),self.coord_h(x-1,y),self.coord_v(x-1,y), self.coord_v(x,y),self.coord_h(x+1,y)])

                if y==0:
                    if x==self.Lx-1:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,self.Ly-1),self.coord_h(x,y),self.coord_h(x,y+1), self.coord_h(0,y), self.coord_h(0,y+1),self.coord_v(x,y+1)])    
                    else:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,self.Ly-1),self.coord_h(x,y),self.coord_h(x,y+1),self.coord_h(x+1,y), self.coord_h(x+1,y+1),self.coord_v(x,y+1)])    
                elif y==self.Ly-1:
                    if x==self.Lx-1:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,y-1),self.coord_h(x,y),self.coord_h(x,0), self.coord_h(0,y), self.coord_h(0,0),self.coord_v(x,0)])    
                    else:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,y-1),self.coord_h(x,y),self.coord_h(x,0), self.coord_h(x+1,y), self.coord_h(x+1,0),self.coord_v(x,0)])
                else:
                    if x==self.Lx-1:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,y-1),self.coord_h(x,y),self.coord_h(x,y+1),self.coord_h(0,y), self.coord_h(0,y+1),self.coord_v(x,y+1)])
                    else:
                        self.important.append([self.coord_v(x,y),self.coord_v(x,y-1),self.coord_h(x,y),self.coord_h(x,y+1), self.coord_h(x+1,y), self.coord_h(x+1,y+1),self.coord_v(x,y+1)])                
    
    #Control gives values to control that the conditions are satisfied based on samples 
    
    def control(self, n, sample, eight, eight_found_in, already_checked, eight_already_checked):
        
        #If an element is sample is 1 change all the important elements of Check to 0
        
        important=self.important[n][1:]
        right_down=important[3:]
        left_up=important[:3]
            
        important_left_up= tf.zeros(shape=(0),dtype=tf.float32)
        important_right_down=tf.zeros(shape=(0),dtype=tf.float32)

        for m in important:
            if m>eight_found_in:
                important_1=self.important[m][1:]
                important_left_up_1=important_1[:3]
                important_right_down_1=important_1[3:]
                important_left_up=tf.concat([important_left_up,important_left_up_1], axis=0)
                important_right_down=tf.concat([important_right_down,important_right_down_1], axis=0)

        important_future_left_up,_, multiplicity_future_left_up=tf.unique_with_counts(important_left_up)
        important_future_right_down,_, multiplicity_future_right_down=tf.unique_with_counts(important_right_down)
                
#         print(n, "important", important)
#         print("important_future_left_up",important_future_left_up)
#         print("idx_future_left_up",multiplicity_future_left_up)
#         print("important_future_right_down", important_future_right_down)
#         print("idx_future_left_up",multiplicity_future_right_down)        
    

        #Check for only one condition "multip"
        multip=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1)),dtype=tf.float32)

        change_1=tf.cast(tf.math.add(tf.math.multiply(sample,-1),1),dtype=tf.float32)
        no_change_1=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1)),dtype=tf.float32)

        for x in range(self.N):
            if x in important and x!=0:
                multip=tf.concat([multip, change_1], axis=1)
            elif x!=0 and eight=="True" and x==n:
                change_eight=tf.cast(tf.math.add(sample,1),dtype=tf.float32)
                multip=tf.concat([multip, change_eight], axis=1)
            elif x!=0:
                multip=tf.concat([multip, no_change_1], axis=1)

        #Check for already checked "already_checked_multip"
        already_checked_multip=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1)),dtype=tf.float32)
        change_ac=tf.cast(tf.math.add(sample,1),dtype=tf.float32)
        
        for x in range(self.N):
            if x in important and x!=0:
                already_checked_multip=tf.concat([already_checked_multip, change_ac], axis=1)
            elif x!=0:
                already_checked_multip=tf.concat([already_checked_multip, no_change_1], axis=1)
        
        #Check for at least one "multiply" 
        multiply=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1,2)),dtype=tf.float32)
        
        future_change=tf.cast(tf.math.add(sample,1),dtype=tf.float32)
        
        if eight=="True":
            change=tf.cast(tf.math.add(tf.math.multiply(sample,-1),1),dtype=tf.float32)
            other_change=change
        else:
            change=tf.cast(tf.math.add(tf.math.multiply(sample,-2),2),dtype=tf.float32)
            other_change=tf.cast(tf.math.multiply(change,0.5),dtype=tf.float32)
        no_change=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1)),dtype=tf.float32)

        for x in range(self.N):
            if x in right_down and x!=0:
                change_one=tf.concat([change, other_change], axis=1)
                change_one=tf.reshape(change_one,shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, change_one], axis=1)
                
            elif x in left_up and x!=0:
                change_two=tf.concat([other_change,change],axis=1)
                change_two=tf.reshape(change_two,shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, change_two], axis=1)
            
            elif x==n and x!=0:
                change_eight_n=tf.cast(tf.math.add(tf.math.multiply(sample, 8),1), dtype=tf.float32)
                change_eight_n=tf.concat([change_eight_n,change_eight_n],axis=1)
                change_eight_n=tf.reshape(change_eight_n,shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, change_eight_n], axis=1)
            
            elif x==n and x!=0:
                no_change_one=tf.concat([no_change,no_change],axis=1)
                no_change_one=tf.reshape(no_change_one,shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, no_change_one], axis=1)
            
            elif ((x in important_future_right_down) and (x in important_future_left_up)) and x!=0:
                
                idx_true_right_down=tf.equal(important_future_right_down ,x)
                idx_right_down=tf.reshape(tf.where(idx_true_right_down), shape=(1,))
                multiplicity_r_d=tf.cast(multiplicity_future_right_down[idx_right_down[0]], dtype=tf.float32)
                
                idx_true_left_up=tf.equal(important_future_left_up, x)
                idx_left_up=tf.reshape(tf.where(idx_true_left_up), shape=(1,))
                multiplicity_l_u=tf.cast(multiplicity_future_left_up[idx_left_up[0]], dtype=tf.float32)
                                       
                change_future_one=tf.concat([tf.math.pow(future_change,multiplicity_r_d),tf.math.pow(future_change, multiplicity_l_u)],axis=1)
                change_future_one=tf.reshape(change_future_one, shape=(tf.shape(sample)[0],1,2)) 
                multiply=tf.concat([multiply, change_future_one], axis=1)

            elif x in important_future_right_down and x!=0:
                
                idx_true_right_down=tf.equal(important_future_right_down ,x)
                idx_right_down=tf.reshape(tf.where(idx_true_right_down), shape=(1,))
                multiplicity_r_d=tf.cast(multiplicity_future_right_down[idx_right_down[0]], dtype=tf.float32)
                
                change_future_one=tf.concat([tf.math.pow(future_change,multiplicity_r_d),no_change],axis=1)
                change_future_one=tf.reshape(change_future_one, shape=(tf.shape(sample)[0],1,2)) 
                multiply=tf.concat([multiply, change_future_one], axis=1)
            
            elif x in important_future_left_up and x!=0:
                
                idx_true_left_up=tf.equal(important_future_left_up, x)
                idx_left_up=tf.reshape(tf.where(idx_true_left_up), shape=(1,))
                multiplicity_l_u=tf.cast(multiplicity_future_left_up[idx_left_up[0]], dtype=tf.float32)
                
                change_future_two=tf.concat([no_change, tf.math.pow(future_change, multiplicity_l_u)],axis=1)
                change_future_two=tf.reshape(change_future_two, shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, change_future_two], axis=1)
    
            elif x!=0:
                no_change_one=tf.concat([no_change,no_change],axis=1)
                no_change_one=tf.reshape(no_change_one,shape=(tf.shape(sample)[0],1,2))
                multiply=tf.concat([multiply, no_change_one], axis=1)

        #If the edge is already checked divide by 2
        p=0
        for s in already_checked:
            p+=1 
        edges_already_checked=tf.gather(already_checked,[1], axis=1)
        
        if eight!="True" and (n in edges_already_checked) and p!=0:
            diagonal=tf.ones([1,tf.shape(sample)[0]], dtype=tf.int32)[0]
            sampling=tf.linalg.tensor_diag(diagonal)
            check_for_n=tf.equal(edges_already_checked,n)
            indices_already_checked=tf.where(check_for_n)
            indices_already_checked=tf.gather(indices_already_checked,[0], axis=1)
#             print("already_checked", already_checked)
#             print("edges_already_checked", edges_already_checked)
#             print("indices",indices_already_checked)
            samples_already_checked=tf.gather(already_checked, indices_already_checked)
#             print("samples_already_checked", samples_already_checked)
            samples_already_checked=tf.reshape(samples_already_checked, shape=(tf.shape(samples_already_checked)[0],tf.shape(samples_already_checked)[2])) 
            samp_already_checked=tf.zeros(shape=(tf.shape(sample)[0],), dtype=tf.float32)

            for m in samples_already_checked:
                samp_already_checked+=tf.cast(tf.gather(sampling, m[0]), dtype=tf.float32)
            
#             print("samp_already_checked",samp_already_checked)
            
            change_already_checked=tf.math.add(tf.math.divide(-samp_already_checked,2),1)
            change_already_checked=tf.reshape(change_already_checked, shape=(tf.shape(change_already_checked)[0],1))
#             print("change_already_checked",change_already_checked)
            correction_multiply=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1,2)),dtype=tf.float32)
            
            for x in range(self.N):

                if x in right_down and x!=0:
                    change_future_alr=tf.concat([change_already_checked,no_change],axis=1)
                    change_future_alr=tf.reshape(change_future_alr, shape=(tf.shape(sample)[0],1,2)) 
                    correction_multiply=tf.concat([correction_multiply, change_future_alr], axis=1)

                elif x in left_up and x!=0:
                    change_future_alr=tf.concat([no_change, change_already_checked],axis=1)
                    change_future_alr=tf.reshape(change_future_alr, shape=(tf.shape(sample)[0],1,2))
                    correction_multiply=tf.concat([correction_multiply, change_future_alr], axis=1)

                elif x!=0:
                    no_change_one=tf.concat([no_change,no_change],axis=1)
                    no_change_one=tf.reshape(no_change_one,shape=(tf.shape(sample)[0],1,2))
                    correction_multiply=tf.concat([correction_multiply, no_change_one], axis=1)                
            multiply=tf.math.multiply(multiply, correction_multiply)
            
        f=0
        for s in eight_already_checked:
            f+=1 
        eight_edges_checked=tf.gather(eight_already_checked,[1], axis=1)

        if eight!="True" and (n in eight_edges_checked) and f!=0:

            diagonal=tf.ones([1,tf.shape(sample)[0]], dtype=tf.int32)[0]
            sampling=tf.linalg.tensor_diag(diagonal)
            check_for_n=tf.equal(eight_edges_checked,n)
            indices_eight_already_checked=tf.where(check_for_n)
            indices_eight_already_checked=tf.gather(indices_eight_already_checked,[0], axis=1)
#             print("eight_already_checked", eight_already_checked)
#             print("eight_edges_checked", eight_edges_checked)
#             print("indices_eight_already_checked",indices_eight_already_checked)
            samples_eights_already_checked=tf.gather(eight_already_checked, indices_eight_already_checked)
#             print("samples_eights_already_checked", samples_eights_already_checked)
            samples_eights_already_checked=tf.reshape(samples_eights_already_checked, shape=(tf.shape(samples_eights_already_checked)[0],tf.shape(samples_eights_already_checked)[2])) 
            samp_eight=tf.zeros(shape=(tf.shape(sample)[0],), dtype=tf.float32)

            for m in samples_eights_already_checked:
                samp_eight+=tf.cast(tf.gather(sampling, m[0]), dtype=tf.float32)

#             print("samp_eight",samp_eight)
            
            change_already_checked=tf.math.add(tf.math.divide(-samp_eight,2),1)
            change_already_checked=tf.reshape(change_already_checked, shape=(tf.shape(change_already_checked)[0],1))
#             print("change_already_checked",change_already_checked)
            correction_multiply=tf.cast(tf.ones(shape=(tf.shape(sample)[0],1,2)),dtype=tf.float32)
            
            for x in range(self.N):

                if ((x in important_future_right_down) and (x in important_future_left_up)) and x!=0:
                    
                    idx_true_right_down=tf.equal(important_future_right_down ,x)
                    idx_right_down=tf.reshape(tf.where(idx_true_right_down), shape=(1,))
                    multiplicity_r_d=tf.cast(multiplicity_future_right_down[idx_right_down[0]], dtype=tf.float32)

                    idx_true_left_up=tf.equal(important_future_left_up, x)
                    idx_left_up=tf.reshape(tf.where(idx_true_left_up), shape=(1,))
                    multiplicity_l_u=tf.cast(multiplicity_future_left_up[idx_left_up[0]], dtype=tf.float32)
                    
                    change_future_alr=tf.concat([tf.math.pow(change_already_checked, multiplicity_r_d),tf.math.pow(change_already_checked, multiplicity_l_u)],axis=1)
                    change_future_alr=tf.reshape(change_future_alr, shape=(tf.shape(sample)[0],1,2)) 
                    correction_multiply=tf.concat([correction_multiply, change_future_alr], axis=1)
    
                elif x in important_future_right_down and x!=0:
                    
                    idx_true_right_down=tf.equal(important_future_right_down ,x)
                    idx_right_down=tf.reshape(tf.where(idx_true_right_down), shape=(1,))
                    multiplicity_r_d=tf.cast(multiplicity_future_right_down[idx_right_down[0]], dtype=tf.float32)
                    
                    change_future_alr=tf.concat([tf.math.pow(change_already_checked,multiplicity_r_d),no_change],axis=1)
                    change_future_alr=tf.reshape(change_future_alr, shape=(tf.shape(sample)[0],1,2)) 
                    correction_multiply=tf.concat([correction_multiply, change_future_alr], axis=1)

                elif x in important_future_left_up and x!=0:
                                           
                    idx_true_left_up=tf.equal(important_future_left_up, x)
                    idx_left_up=tf.reshape(tf.where(idx_true_left_up), shape=(1,))
                    multiplicity_l_u=tf.cast(multiplicity_future_left_up[idx_left_up[0]], dtype=tf.float32)
                    
                    change_future_alr=tf.concat([no_change, tf.math.pow(change_already_checked, multiplicity_l_u)],axis=1)
                    change_future_alr=tf.reshape(change_future_alr, shape=(tf.shape(sample)[0],1,2))
                    correction_multiply=tf.concat([correction_multiply, change_future_alr], axis=1)

                elif x!=0:
                    no_change_one=tf.concat([no_change,no_change],axis=1)
                    no_change_one=tf.reshape(no_change_one,shape=(tf.shape(sample)[0],1,2))
                    correction_multiply=tf.concat([correction_multiply, no_change_one], axis=1)                
            multiply=tf.math.multiply(multiply, correction_multiply)
        return multip, multiply, already_checked_multip
    # Prob_changer Returns [1,1] if Check=1, [0,1] if Check=2x, [1,0] if Check=0
    
    def Prob_Changer(self,n,Check_only_one):
        n=int(n)
        
        #Extract the values Check for the nth edge
        indices=tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        for s in range(tf.shape(Check_only_one)[0]):
            indices=indices.write(s,[[[s,n]]])
        
        indices=indices.concat()
        
        Check_for_n=tf.gather_nd(Check_only_one, indices)

        #Apply x^2-x to make (1,0) give (0,0) and (2,2x) not zero 
        First_change=tf.math.add(tf.math.multiply(Check_for_n,Check_for_n),-Check_for_n)
        
        #Divide by itslef to make all non zero values 1 
        First_change=tf.math.divide_no_nan(First_change,First_change)
        
        # Apply -x+1 to make all zero values 1 and all one values 
        First_change=tf.math.add(-First_change,1)

        #Divide by itself to make all non zero values 1
        Second_change=tf.math.divide_no_nan(Check_for_n,Check_for_n)

        Changer=tf.concat([First_change,Second_change],axis=1)
        Changer=tf.reshape(Changer,shape=(tf.shape(Check_only_one)[0],1,2))
        
        return Changer

    def Check_sample(self,samples):
    # Check Only one Sample per site returns 1 if correct 0 otherwise
        only_one_error_sum=tf.zeros(shape=tf.shape(samples)[0])    

        #Check errors in samples and add them 
        for n in range(len(self.adjacent)):
            only_one_error_sum+=tf.cast(samples[:,self.adjacent[n][0]]*samples[:,self.adjacent[n][1]],tf.float32)
        #Make all errors 1 and all success 0
        Check_only_one=tf.math.divide_no_nan(only_one_error_sum,only_one_error_sum)
        #Make all erros 1 and all success 1
        Check_only_one=tf.math.add(-Check_only_one,1)

        # Check That all have at least one:
        Expected_number_of_dimers= int(0.5*Lx*Ly)
        Number_of_dimers=tf.math.reduce_sum(samples, axis=1)

        Check_number=tf.cast(tf.math.add(Number_of_dimers,-Expected_number_of_dimers), dtype=tf.float32)

        Check_number=tf.math.divide_no_nan(Check_number,Check_number)

        Check_number=tf.math.add(-Check_number,1)
    
        #Final check 
        self.Check=tf.math.multiply(Check_only_one,Check_number)
    
    
#     @tf.function
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples,self.nh])
        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)

        #Initial Control for conditions 
        Check_only_one=tf.ones(shape=(nsamples,self.N))
        Check_at_least_one=tf.ones(shape=(nsamples,self.N,2))
        Check_already_checked=tf.ones(shape=(nsamples,self.N))
        zero_sample=tf.zeros(shape=nsamples, dtype=tf.float32)

        for j in range(self.N):
            print("n",j)
            begin_time=datetime.now()
            print(begin_time)
            
############################################################################################################################            
            #Check for correctness
            if j!=0:
                
                Check_only_one_identity=tf.identity(Check_only_one)
                Check_at_least_one_identity=tf.identity(Check_at_least_one)
                Check_already_checked_identity=tf.identity(Check_already_checked)
            
            #Generating testing samples
                
                #Extract the values Check for the jth edge
                indices=tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
                for s in range(tf.shape(Check_only_one)[0]):
                    indices=indices.write(s,[[[s,j]]])

                indices=indices.concat()

                Check_for_n=tf.gather_nd(Check_only_one_identity, indices)
                # transform every 1 and 2 to 0, and keep every zero like zero
                sample_testing_1=tf.math.divide_no_nan(Check_for_n,Check_for_n)
                
                # transform every 1 and zero to zero and 2 to 1
                sample_testing_0=tf.math.multiply(tf.math.add(Check_for_n,-1),Check_for_n)
                sample_testing_0=tf.math.divide_no_nan(sample_testing_0,sample_testing_0)
                
                sample_testing=tf.concat([[sample_testing_1],[sample_testing_0]], axis=0)
#                 print("sample_testing",sample_testing)
                
                Check_correctness=tf.zeros(shape=(1,tf.shape(sample_testing_0)[0]), dtype=tf.int64)

                
            #Control of correctness
                for sample in sample_testing:
                    Check_only_one_identity=tf.identity(Check_only_one)
                    Check_at_least_one_identity=tf.identity(Check_at_least_one)
                    Check_already_checked_identity=tf.identity(Check_already_checked)
#                     print("for sample_being_test", sample)
                    important=self.important[j][1:]
                    already_checked_true=tf.math.greater_equal(Check_already_checked_identity,2)
                    already_checked=tf.where(already_checked_true)

                    eight_already_checked_true=tf.equal(Check_only_one_identity,2)
                    eight_already_checked= tf.where(eight_already_checked_true)

#                     print(j,"already_checked_identity",already_checked)
#                     print(j,"eight_already_checked_true_identity", eight_already_checked)

                    #Condition one satisfied
                    Control_only_one, Control_at_least_one, Control_already_checked=self.control(j,sample, "False", j, already_checked, eight_already_checked)
                    Check_only_one_identity=tf.math.multiply(Check_only_one_identity,Control_only_one)
                    Check_at_least_one_identity=tf.math.multiply(Check_at_least_one_identity,Control_at_least_one)
                    Check_already_checked_identity=tf.math.multiply(Check_already_checked_identity, Control_already_checked)

                    #Condition two satisfied
                    diagonal=tf.ones([1,nsamples], dtype=tf.int32)[0]
                    eight_samples=tf.linalg.tensor_diag(diagonal)


                    #Apply 1/2*x*(x+1)
                    eight_transformation=tf.math.add(-tf.math.divide_no_nan(tf.math.add(Check_at_least_one_identity,-8),tf.math.add(Check_at_least_one_identity,-8)),1)
                    eight_condition=tf.math.reduce_sum(eight_transformation)

                    while eight_condition!=0:
                        already_checked_true=tf.math.greater_equal(Check_already_checked_identity,2)
                        already_checked=tf.where(already_checked_true)

                        eight_already_checked_true=tf.equal(Check_only_one_identity,2)
                        eight_already_checked= tf.where(eight_already_checked_true)

#                         print(j," while loop already_checked identity",already_checked)
#                         print(j," while loop eight_already_checked_true identity", eight_already_checked)

                        # Return the places where there is a zero indices=[[sample, number, side]]

                        where=tf.equal(Check_at_least_one_identity, 8)
                        indices=tf.where(where)

                        k=0
                        for indice in indices:
                            eight_sample=tf.reshape(eight_samples[indice[0]],[nsamples,1])
#                             print(indice)
                            eight_edge=indice[1]
                            if k==0:
                                Control_only_one, Control_at_least_one, Control_already_checked=self.control(eight_edge, eight_sample, "True",j, already_checked, eight_already_checked)
                            else:
                                multip, multiply, ac_multip=self.control(eight_edge, eight_sample, "True",j, already_checked,  eight_already_checked)

                                Control_only_one=tf.math.multiply(Control_only_one, multip)
                                Control_at_least_one=tf.math.multiply(Control_at_least_one, multiply)
                                Control_already_checked=tf.math.add(tf.math.multiply(Control_already_checked, ac_multip),-1)
                                Control_already_checked=tf.math.add(tf.math.divide_no_nan(Control_already_checked,Control_already_checked),1)
                            k+=1

                        # Update Check only one and Check at least one  
                        Check_at_least_one_identity=tf.math.multiply(Check_at_least_one_identity,Control_at_least_one)
                        Check_only_one_identity=tf.math.multiply(Check_only_one_identity, Control_only_one)
                        Check_already_checked_identity=tf.math.multiply(Check_already_checked_identity, Control_already_checked)

#                         print("Check_at_least_one_identity_after_control",Check_at_least_one_identity)

                        # Check for new eights 
                        eight_transformation=tf.math.add(-tf.math.divide_no_nan(tf.math.add(Check_at_least_one_identity,-8),tf.math.add(Check_at_least_one_identity,-8)),1)
                        eight_condition=tf.math.reduce_sum(eight_transformation)
                        
#                     print("Check_only_one_identity",sample, Check_only_one_identity)
                    Number_of_edges=tf.cast(tf.reduce_sum(tf.math.divide_no_nan(Check_only_one_identity,Check_only_one_identity), axis=1), dtype=tf.int64)
                    Number_of_edges=tf.reshape(Number_of_edges, shape=(1, tf.shape(Number_of_edges)[0]))
#                     print("Number_of_edges_sum",sample ,Number_of_edges)
#                     print("zero_add", zero_sample)
                    #Correct because of zero edge
                    Number_of_edges=tf.math.add(zero_sample,Number_of_edges)
#                     print("Number_of_edges",Number_of_edges)
                    Check_correctness=tf.concat([Number_of_edges ,Check_correctness], axis=0)
                
#                 print("Check_correctness",Check_correctness)
                
                grater_than_required_number=tf.math.greater_equal(Check_correctness,tf.constant([self.expected], dtype=tf.int64))
                
#                 print("grater_than_required_number",grater_than_required_number)
                
                Check_correctness_for_zero=tf.where(grater_than_required_number[0], 1, 3)
                
#                 print("Check_correctness_for_zero",Check_correctness_for_zero)
                
                Check_correctness_for_one=tf.where(grater_than_required_number[1],1,0)
                
#                 print("Check_correctness_for_one",Check_correctness_for_one)
                
                Correctness_change=tf.math.multiply(Check_correctness_for_zero,Check_correctness_for_one)
                
#                 print("Correctness_change", Correctness_change)
                
                Correctness_change=tf.cast(tf.reshape(Correctness_change, shape=(tf.shape(Correctness_change)[0],1)),dtype=tf.float32)
                
                multip=tf.cast(tf.ones(shape=(tf.shape(Correctness_change)[0],1)),dtype=tf.float32)
                no_change=tf.cast(tf.ones(shape=(tf.shape(Correctness_change)[0],1)),dtype=tf.float32)

                for x in range(self.N):
                    if x==j and x!=0:
                        multip=tf.concat([multip, Correctness_change], axis=1)
                    elif x!=0:
                        multip=tf.concat([multip, no_change], axis=1)
                Check_only_one=tf.math.multiply(multip,Check_only_one)


############################################################################################################################            
#             print("m",j)
            # Run a single RNN cell
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
            # Compute probabilities
            probs = self.dense(rnn_output)

            #Apply Prob_Changer
            Changer= self.Prob_Changer(j,Check_only_one)   
            probs = tf.math.multiply(probs,Changer)
            #probability log
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            # Sample
            sample = tf.random.categorical(log_probs,num_samples=1)
            
            # Save zero sample
            if j==0:
                zero_sample=tf.reshape(tf.math.add(sample,-1),shape=nsamples)
         
            #Control
            important=self.important[j][1:]
            already_checked_true=tf.math.greater_equal(Check_already_checked,2)
            already_checked=tf.where(already_checked_true)
            
            eight_already_checked_true=tf.equal(Check_only_one,2)
            eight_already_checked= tf.where(eight_already_checked_true)
            
#             print(j,"already_checked",already_checked)
#             print(j,"eight_already_checked_true", eight_already_checked)
            
            #Condition one satisfied
            Control_only_one, Control_at_least_one, Control_already_checked=self.control(j,sample, "False", j, already_checked, eight_already_checked)
            Check_only_one=tf.math.multiply(Check_only_one,Control_only_one)
            Check_at_least_one=tf.math.multiply(Check_at_least_one,Control_at_least_one)
            Check_already_checked=tf.math.multiply(Check_already_checked, Control_already_checked)
            
            #Condition two satisfied
            diagonal=tf.ones([1,nsamples], dtype=tf.int32)[0]
            eight_samples=tf.linalg.tensor_diag(diagonal)


            #Apply 1/2*x*(x+1)
            eight_transformation=tf.math.add(-tf.math.divide_no_nan(tf.math.add(Check_at_least_one,-8),tf.math.add(Check_at_least_one,-8)),1)
            eight_condition=tf.math.reduce_sum(eight_transformation)

            while eight_condition!=0:
                already_checked_true=tf.math.greater_equal(Check_already_checked,2)
                already_checked=tf.where(already_checked_true)

                eight_already_checked_true=tf.equal(Check_only_one,2)
                eight_already_checked= tf.where(eight_already_checked_true)

#                 print(j," while loop already_checked",already_checked)
#                 print(j," while loop eight_already_checked_true", eight_already_checked)

                # Return the places where there is a zero indices=[[sample, number, side]]

                where=tf.equal(Check_at_least_one, 8)
                indices=tf.where(where)
                
                k=0
                for indice in indices:
                    eight_sample=tf.reshape(eight_samples[indice[0]],[nsamples,1])
#                     print(indice)
                    eight_edge=indice[1]
                    if k==0:
                        Control_only_one, Control_at_least_one, Control_already_checked=self.control(eight_edge, eight_sample, "True",j, already_checked, eight_already_checked)
                    else:
                        multip, multiply, ac_multip=self.control(eight_edge, eight_sample, "True",j, already_checked,  eight_already_checked)

                        Control_only_one=tf.math.multiply(Control_only_one, multip)
                        Control_at_least_one=tf.math.multiply(Control_at_least_one, multiply)
                        Control_already_checked=tf.math.add(tf.math.multiply(Control_already_checked, ac_multip),-1)
                        Control_already_checked=tf.math.add(tf.math.divide_no_nan(Control_already_checked,Control_already_checked),1)
                    k+=1

                # Update Check only one and Check at least one  
                Check_at_least_one=tf.math.multiply(Check_at_least_one,Control_at_least_one)
                Check_only_one=tf.math.multiply(Check_only_one, Control_only_one)
                Check_already_checked=tf.math.multiply(Check_already_checked, Control_already_checked)
                
#                 print("eight_at_least_one",Check_at_least_one)
                
                # Check for new eights 
                eight_transformation=tf.math.add(-tf.math.divide_no_nan(tf.math.add(Check_at_least_one,-8),tf.math.add(Check_at_least_one,-8)),1)
                eight_condition=tf.math.reduce_sum(eight_transformation)
            print(f"{Lx}x{Ly} 1 sample Time",datetime.now() - begin_time)
                
            #Samples
            if (j == 0):
                samples = tf.identity(sample)
            else:
                samples = tf.concat([samples,sample],axis=1)
            # Feed result to the next cell
            inputs = tf.one_hot(sample,depth=self.K)
            add = tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)

            logP = logP+tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)
#             print(j,"samples",samples)
#             print(j,"Check_only_one", Check_only_one)
#             print(j, "Check_at_least_one", Check_at_least_one)
#             print(j,"Check_already_checked", Check_already_checked)
#             print(j, "important", important)
    
        return samples, logP

    @tf.function
    def logpsi(self,samples):
        # Shift data
        num_samples = tf.shape(samples)[0]
        data   = tf.one_hot(samples[:,0:self.N-1],depth=self.K)
        x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K)
        inputs = tf.concat([x0,data],axis=1)

        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)

        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),tf.one_hot(samples,depth=self.K)),axis=2)

        return tf.reduce_sum(log_probs,axis=1)
    
    #@tf.function
    def localenergy(self,samples):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
        # Adding Parallel Horizontal
        for n in range(len(self.horizontal)):
            eloc += J * tf.cast(samples[:,self.horizontal[n][0]]*samples[:,self.horizontal[n][1]],tf.float32)
        #Adding Parallel Vertical
        for n in range(len(self.vertical)):
            eloc += J * tf.cast(samples[:,self.vertical[n][0]]*samples[:,self.vertical[n][1]],tf.float32)
        return eloc

from datetime import datetime
#size=[10,14]
a=int(sys.argv[1])
size=[a]
for Lx in size:
    Ly=Lx
    N=2*Lx*Ly
    vmc = VariationalMonteCarlo(Lx,Ly,J,T,nh,lr,epochs,seed)
    with open(f"{Lx}x{Ly}_{ns}dimertime.txt", 'w') as f:
        begin_time=datetime.now()
        print(begin_time)
        samples,_=vmc.sample(ns)
        print("################################################################")
        print(f"{Lx}x{Ly} end Time",datetime.now() - begin_time, file=f)
        print(samples[0], file=f)
        print(f"{Lx}x{Ly} end Time",datetime.now() - begin_time)
        print("Number of samples=", ns, "Lx=", Lx, "Ly=", Ly)
        print("samples",samples, file=f)
        vmc.Check_sample(samples)
        Cs=vmc.Check
        print("Correct Samples give 1",Cs, file=f)
        print("################################################################")

    import torch as torch
    s=samples.numpy()
    sample_dimers=torch.tensor(s)
    torch.save(sample_dimers, f'samples_in_dimers_nh={nh}_size={Lx}x{Ly}x{ns}_T={T}_epoch={epochs-1}.pt' )