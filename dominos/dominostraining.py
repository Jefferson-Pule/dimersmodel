import tensorflow as tf
import numpy as np
import logging
import time 
import sys
import os

# Hamiltonian parameters
Lx = int(sys.argv[1])      # Linear size in x direction
Ly = int(sys.argv[2])      # Linear size in y direction
N = Lx*Ly                  # Total number of atoms 
J = 1.0                    # Strenght  
T = float(sys.argv[3])       # Temperature
# RNN-VMC parameters
lr = 0.001                 # learning rate of Adam optimizer
nh = int(sys.argv[6])      # Number of hidden units in the GRU cell
ns = int(sys.argv[4])      # Number of samples used to approximate the energy at each step
epochs = int(sys.argv[5])  # Training iterations
seed = 1234                # Seed of RNN
#Directories
checkpoint_dir=sys.argv[7]  #Checkpoints directory
current_dir=sys.argv[8]	    #Current directory
os.chdir(current_dir)
# Create a class
class VariationalMonteCarlo(tf.keras.Model):
    # Build our Model
    def __init__(self, Lx, Ly, J, T, nsamples,
                 num_hidden, learning_rate,
                 epochs, seed=1234):
        super(VariationalMonteCarlo, self).__init__()

        #Variables
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.J        = -J               # Strenght
        self.N        = Lx * Ly         # Number of spins
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator
        self.epochs   = epochs          # Training epochs 
        self.K        = 4               # Dimension of the local Hilbert space
        self.T        = T
        self.ns       = nsamples
        
        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer-We use Adam
    
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
        # Generate the list of bonds for NN on a 
        # square lattice with open boundaries
        
        self.buildlattice()
        self.buildlattice_dimers()   
    # Define buildlattice

    def coord_to_site(self,x,y):
        return self.Lx*y+x

    def buildlattice(self):
        self.important=[]
        self.upper_boundary=[]
        self.n_samples_edges=[]
        self.change=tf.constant([[[2,0,0,0],[0,1,0,0],[0,1,1,1],[1,1,1,0],[1,1,0,1]],
                        [[0,2,0,0],[1,0,1,1],[1,0,0,0],[1,1,1,0],[1,1,0,1]],
                        [[0,0,2,0],[1,0,1,1],[0,1,1,1],[0,0,0,1],[1,1,0,1]],
                        [[0,0,0,2],[1,0,1,1],[0,1,1,1],[1,1,1,0],[0,0,1,0]],
                        [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]])
        self.close=tf.constant([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[1,1,1,1]])

       

        for y in range(self.Ly):
            for x in range(self.Lx):

                if y==0 and x==0:  
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x, self.Ly-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(self.Lx-1,y) ,self.coord_to_site(x+1,y)])

                    self.upper_boundary.append(self.coord_to_site(x,y))

                elif y==0 and x==self.Lx-1:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x, self.Ly-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(0,y)])

                    self.upper_boundary.append(self.coord_to_site(x,y))

                elif y==0:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x, self.Ly-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(x+1,y)])

                    self.upper_boundary.append(self.coord_to_site(x,y))
                elif y== self.Ly-1 and x==0:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,0),
                                        self.coord_to_site(self.Lx-1,y) ,self.coord_to_site(x+1,y)])

                elif y== self.Ly-1 and x==self.Lx-1:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,0),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(0,y)])
                elif y== self.Ly-1:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,0),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(x+1,y)])
                elif x==0:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(self.Lx-1,y) ,self.coord_to_site(x+1,y)])
                elif x==self.Lx-1:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(0,y)])
                else:
                    self.important.append([self.coord_to_site(x,y), self.coord_to_site(x,y-1), self.coord_to_site(x,y+1),
                                        self.coord_to_site(x-1,y) ,self.coord_to_site(x+1,y)])
        edges=[]
        for y in range(self.Ly):
            for x in range(self.Lx):
                if y==0:
                    if x==0:
                        edges.append([self.coord_to_site(x,self.Ly-1), self.coord_to_site(x,y+1), self.coord_to_site(self.Lx-1,y), self.coord_to_site(x+1,y)])
                    elif x==self.Lx-1:
                        edges.append([self.coord_to_site(x,self.Ly-1), self.coord_to_site(x,y+1), self.coord_to_site(x-1,y), self.coord_to_site(0,y)])
                    else:
                        edges.append([self.coord_to_site(x,self.Ly-1), self.coord_to_site(x,y+1), self.coord_to_site(x-1,y), self.coord_to_site(x+1,y)])
                elif y==self.Ly-1:
                    if x==0:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,0), self.coord_to_site(self.Lx-1,y), self.coord_to_site(x+1,y)])
                    elif x==self.Lx-1:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,0), self.coord_to_site(x-1,y), self.coord_to_site(0,y)])
                    else:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,0), self.coord_to_site(x-1,y), self.coord_to_site(x+1,y)])
                else:
                    if x==0:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,y+1), self.coord_to_site(Lx-1,y), self.coord_to_site(x+1,y)])
                    elif x==self.Lx-1:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,y+1), self.coord_to_site(x-1,y), self.coord_to_site(0,y)])
                    else:
                        edges.append([self.coord_to_site(x,y-1), self.coord_to_site(x,y+1), self.coord_to_site(x-1,y), self.coord_to_site(x+1,y)])
        
        for x in range(self.ns):
            self.n_samples_edges.append(edges)

        #Not visited values for the deepth first search
            self.Visit=tf.constant([False]*(self.Lx*self.Ly))


    def control(self,j, sample,p):

        important=self.important[j]
        important_without_j=important[1:]
        opposite=[1,0,3,2]

        change_j=tf.map_fn(lambda x: self.change[x[0]][0], sample)
        change_up=tf.map_fn(lambda x: self.change[x[0]][1], sample)
        change_down=tf.map_fn(lambda x: self.change[x[0]][2], sample)
        change_left=tf.map_fn(lambda x: self.change[x[0]][3], sample)
        change_right=tf.map_fn(lambda x: self.change[x[0]][4], sample)
        no_change=tf.ones(shape=(tf.shape(sample)[0], 4), dtype=tf.int32)

        Control=no_change

        for edge in range(self.N):
            if 0 in important:

                if edge==0:
                    k=important.index(0)

                    if k==0:
                        Control=tf.math.multiply(Control, change_j)
                    elif k==1:
                        Control=tf.math.multiply(Control, change_up)
                    elif k==2:
                        Control=tf.math.multiply(Control, change_down)
                    elif k==3:
                        Control=tf.math.multiply(Control, change_left)
                    elif k==4:
                        Control=tf.math.multiply(Control, change_right)

                elif edge==important[0]:
                    Control=tf.concat([Control, change_j], axis=1)

                elif edge==important[1]:
                    Control=tf.concat([Control, change_up], axis=1)

                elif edge==important[2]:
                    Control=tf.concat([Control, change_down], axis=1)

                elif edge==important[3]:
                    Control=tf.concat([Control, change_left], axis=1)

                elif edge==important[4]:
                    Control=tf.concat([Control, change_right], axis=1)

                else:
                    Control=tf.concat([Control, no_change], axis=1)

            else:
                if edge==0:
                    Control=tf.identity(Control)

                elif edge==important[0]:
                    Control=tf.concat([Control, change_j], axis=1)

                elif edge==important[1]:
                    Control=tf.concat([Control, change_up], axis=1)

                elif edge==important[2]:
                    Control=tf.concat([Control, change_down], axis=1)

                elif edge==important[3]:
                    Control=tf.concat([Control, change_left], axis=1)

                elif edge==important[4]:
                    Control=tf.concat([Control, change_right], axis=1)

                else:
                    Control=tf.concat([Control, no_change], axis=1)

        Control=tf.reshape(Control, shape=[tf.shape(sample)[0],N,4])

        if p=="True":
            for n in range(4):

                other_true=tf.math.equal(sample, n)
                other=tf.where(other_true, opposite[n], 4)
                Other_control=self.control(important_without_j[n], other, "False")
                Control=tf.math.multiply(Control,Other_control)
        
        
        return Control
    
    def Closeup(self,j, inpt, parity):
        Boundary=self.upper_boundary
        Boundary_odd=[]
        Boundary_even=[]

        for n in range(j+1,self.Lx):
            if n%2==1:
                Boundary_odd.append(n)

            if n%2==0:
                Boundary_even.append(n)    

        Close_up_change=tf.map_fn(lambda x: self.change[x[0]][2], inpt)
        No_change=tf.ones(shape=(tf.shape(inpt)[0], 4), dtype=tf.int32)
        In_Control_Closeup=No_change


        if parity=="odd":
            for edge in range(self.N):
                if edge==0:
                    Control_Closeup=tf.identity(In_Control_Closeup)
                elif edge in Boundary_even:
                    Control_Closeup=tf.concat([Control_Closeup,Close_up_change],axis=1)
                else:
                    Control_Closeup=tf.concat([Control_Closeup,No_change], axis=1)

        if parity=="even":   
            for edge in range(self.N):
                if edge==0:
                    Control_Closeup=tf.identity(In_Control_Closeup)
                elif edge in Boundary_odd:
                    Control_Closeup=tf.concat([Control_Closeup,Close_up_change],axis=1)
                else:
                    Control_Closeup=tf.concat([Control_Closeup, No_change], axis=1)

        Control_Closeup=tf.reshape(Control_Closeup, shape=[tf.shape(inpt)[0],N,4])

        return Control_Closeup
    
    def Flux_control(self,j,sample, Flux, Number_of_white, Number_of_black):

        Boundary=self.upper_boundary
        Boundary_odd=[]
        Boundary_even=[]

        for n in range(j+1,self.Lx):
            if n%2==1:
                Boundary_odd.append(n)

            if n%2==0:
                Boundary_even.append(n)    

        #Flux check
        Check_for_flux=tf.math.equal(sample,0)
        if j%2==0:
            Flux_change=tf.where(Check_for_flux, 1,0)
        else:
            Flux_change=tf.where(Check_for_flux, -1,0)

        New_flux=tf.math.add(Flux,Flux_change)

        #Number of white and black squares available

        Num_white_available=Number_of_white
        Num_black_available=Number_of_black

        if j==0:

            self.last_one_already_check=tf.where(tf.equal(sample,2),-1,0)
            New_numb_white_available=tf.math.add(Num_white_available,self.last_one_already_check)
            New_numb_black_available=tf.math.add(Num_black_available, -1) 

        elif j%2==0:

            New_numb_white_available=Num_white_available
            New_numb_black_available=tf.math.add(Num_black_available, -1)

        else:

            New_numb_white_available=tf.math.add(Num_white_available, -1)
            New_numb_black_available=Num_black_available


        Check_for_white_and_flux=tf.equal(New_flux, New_numb_white_available)

        Check_for_black_and_flux=tf.equal(New_flux, -New_numb_black_available)



        inpt_white=tf.where(Check_for_white_and_flux,0,4)
        inpt_black=tf.where(Check_for_black_and_flux,0,4)


        There_is_a_true_white=tf.math.reduce_any(Check_for_white_and_flux)
        There_is_a_true_black=tf.math.reduce_any(Check_for_black_and_flux)


        Flux_Control=tf.ones(shape=(tf.shape(sample)[0], self.N,4), dtype=tf.int32)
        Flux_Control_black=tf.ones(shape=(tf.shape(sample)[0], self.N,4),dtype=tf.int32)
        Flux_Control_white=tf.ones(shape=(tf.shape(sample)[0], self.N,4),dtype=tf.int32)

        if There_is_a_true_white:
            Flux_Correction_white=1
            
            if j==Lx-3:
                No_more_flux_white=self.Closeup(j,inpt_white,"odd")
                last_one_not_to_close=tf.where(tf.equal(self.last_one_already_check, -1), 4, 0)
                inpt_white_not_to_close=tf.where(tf.equal(inpt_white, 4),0,1)
                last_one_not_to_close=tf.math.multiply(last_one_not_to_close,inpt_white_not_to_close)
                
                inpt_white=tf.math.add(inpt_white,last_one_not_to_close)
                
                for odd_edge in Boundary_odd:
                    Flux_Correction_white_control=self.control(odd_edge,inpt_white, "True")
                    Flux_Correction_white=tf.math.multiply(Flux_Correction_white,Flux_Correction_white_control)
                Flux_Control_white=tf.math.multiply(Flux_Correction_white,No_more_flux_white)
            
            else:
                for odd_edge in Boundary_odd:
                    Flux_Correction_white_control=self.control(odd_edge,inpt_white, "True")
                    Flux_Correction_white=tf.math.multiply(Flux_Correction_white,Flux_Correction_white_control)
                No_more_flux_white=self.Closeup(j,inpt_white,"odd")
                Flux_Control_white=tf.math.multiply(Flux_Correction_white,No_more_flux_white)


        if There_is_a_true_black:
            
            Flux_Correction_black=1
            
            for even_edge in Boundary_even:
                Flux_Correction_black_control=self.control(even_edge,inpt_black, "True")
                Flux_Correction_black=tf.math.multiply(Flux_Correction_black,Flux_Correction_black_control)
            No_more_flux_black=self.Closeup(j,inpt_black,"even")
            Flux_Control_black=tf.math.multiply(Flux_Correction_black,No_more_flux_black)


        Flux_Control=tf.math.multiply(Flux_Control_black, Flux_Control_white)

        return Flux_Control, New_flux ,New_numb_white_available, New_numb_black_available

    def Prob_Changer(self,j,Check):

        indices=[[0,j]]
#        print("indices before loop",indices)


        for s in range(self.ns-1):
            #print("s",s,"j",j)
            indices.append([s+1,j])
        #print("indices after loop",indices)
        indices=tf.constant(indices, dtype=tf.int32)
        indices=tf.reshape(indices, shape=(tf.shape(indices)[0],1,2))
        Changer=tf.gather_nd(Check, indices)
        Changer=tf.math.divide_no_nan(Changer, Changer)

        return Changer

    #This function checks for the size of the connected components of the graph given Check (only check for the last two rows).
    
    def Check_for_boundaries(self,direction_being_check,j,Check):

    #   Find the edges that are open based on Check
#         print("direction_being_check")
        sample_checker_1=tf.constant([[direction_being_check]])
        sample_checker=sample_checker_1
#         print("sample_checker before",sample_checker)
        for i in range(self.ns-1):
            sample_checker=tf.concat([sample_checker,sample_checker_1], axis=0)

        sample_checker=tf.reshape(sample_checker, shape=(tf.shape(Check)[0],1))

#         print("sample_checker",sample_checker)

        Control_checker=tf.cast(self.control(j, sample_checker, "True"), dtype=tf.float32)
        Control_checker=tf.multiply(Check, Control_checker)
     
        #Control in case of one 
        Sum_Check=tf.reduce_sum(Control_checker, axis=2)

        Look_for_sum_one=tf.equal(Sum_Check, 1)

        index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
        
        while index.get_shape()[0]!=0 and index.get_shape()[0]!=None:

            inpt=tf.gather_nd(Control_checker,index)
#                 print("inpt",inpt)
            edges_to_check,_=tf.unique(tf.map_fn(lambda x: x[1], index))
#                 print("edges_to_check", edges_to_check)

            for edge in edges_to_check:
#                     print("edge being tested",edge)
                multip_for_index=tf.zeros(shape=(tf.shape(index)[0]), dtype=tf.int32)

                multip_for_index=tf.one_hot(multip_for_index, depth=2, on_value=self.Lx*self.Ly, off_value=0, dtype=tf.int32)

                multip_index=tf.math.add(index, multip_for_index)

                Look_for_edge_in_index=tf.equal(multip_index,edge)

                Look_for_edge_in_index=tf.math.reduce_any(Look_for_edge_in_index, axis=1)

                inpt_index=tf.cast(tf.where(Look_for_edge_in_index), dtype=tf.int32)
#                     print("inpt_indx",inpt_index)
                inpts_to_concat=tf.gather_nd(inpt, inpt_index)
#                     print(j,"inpts_to_concat",inpts_to_concat)
                number_of_sample=tf.map_fn(lambda x: index[x[0]][0], inpt_index)
#                     print("number_of_sample",number_of_sample)
                inpts=tf.ones(shape=(4))
                no_change=tf.zeros(shape=(4))

                for n in range(self.ns):
                    if n in number_of_sample:
                        find_this_sample_index=tf.reshape(tf.where(tf.equal(number_of_sample,n)),shape=(1,))
#                             print("find sample index",find_this_sample_index)
                        if n==0:
                            inpts=tf.math.multiply(inpts, inpts_to_concat[find_this_sample_index[0]])
                        else:
                            inpts=tf.concat([inpts,inpts_to_concat[find_this_sample_index[0]]], axis=0)
                    else:
                        if n==0:
                            inpts=tf.math.multiply(inpts, no_change)
                        else:
                            inpts=tf.concat([inpts, no_change], axis=0)
                inpts=tf.reshape(inpts, shape=(self.ns,4))

                inpts=tf.cast(tf.reshape(tf.map_fn(lambda x: self.from_Check_to_direction(x), inpts),shape=(self.ns,1)),dtype=tf.int32)

                Control_for_new_ones=tf.cast(self.control(edge, inpts,"True"),dtype=tf.float32)
                Control_checker=tf.multiply(Control_checker,Control_for_new_ones)

            Sum_Check=tf.reduce_sum(Control_checker, axis=2)

            Look_for_sum_one=tf.equal(Sum_Check, 1)

            index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)

#         print("sample",j,"direction",direction_being_check,"Control checker after sum chek", Control_checker)

        Control_checker=tf.math.divide_no_nan(Control_checker,Control_checker)
    #     Control_checker=Check

        Edges_available=tf.math.multiply(tf.constant(self.n_samples_edges, dtype=tf.float32), Control_checker)
#         print("Edges_available", Edges_available)

        Edges_available=tf.cast(tf.sort(Edges_available, axis=2, direction='DESCENDING'),dtype=tf.int32)
#         print("Edges_available", Edges_available)

    #     #Look for non zero edges positions
    #     Truth_zero=tf.math.equal(Edges_available,0)
    #     Truth_non_zero=tf.math.logical_not(Truth_zero)
    #     Edge_with_non_zero_indx=tf.where(Truth_non_zero)

        corrector=tf.constant([4])

        for sample_number in range(self.ns):

            for v in range(2*self.Lx):
                self.Visit=tf.constant([False]*(self.Lx*self.Ly))
                ans=0

                edge=self.Lx*self.Ly-1-v

                ans=self.depthFirst(sample_number,edge, Edges_available,ans)

#                 print("sample", sample_number, "point", edge, "ans", ans)            
                if sample_number==0:
                    if ans%2==1:
                        corrector=tf.multiply(tf.constant([1]), direction_being_check)
                        break
                else:
                    if ans%2==1:
                        corrector=tf.concat([corrector,tf.constant([direction_being_check])], axis=0)
                        break
                    elif v==2*self.Lx-1:
                        corrector=tf.concat([corrector,tf.constant([4])], axis=0)

        Boundaries_changer=tf.map_fn(lambda x: self.close[x], corrector)
        Boundaries_no_change=tf.ones(shape=tf.shape(Boundaries_changer), dtype=tf.int32)
#         print("sample",j,"direction being checked",direction_being_check,"boundaries_changer",Boundaries_changer)

        Control_Boundaries=tf.ones(shape=tf.shape(Boundaries_changer), dtype=tf.int32)

        for edge in range(self.N):
            if edge==0:
                Control_Boundaries=tf.math.multiply(Control_Boundaries, Boundaries_no_change)
            elif edge==j:
                Control_Boundaries=tf.concat([Control_Boundaries, Boundaries_changer], axis=1)
            else:
                Control_Boundaries=tf.concat([Control_Boundaries, Boundaries_no_change], axis=1)
        Control_Boundaries=tf.cast(tf.reshape(Control_Boundaries, shape=tf.shape(Check)), dtype=tf.float32)        

        return Control_Boundaries


    
    # Creates a nsamples with N entries.
        
        
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)

        hidden_state = tf.zeros(shape=[nsamples,self.nh])

        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)
        
        #Check
        
        Check=tf.ones(shape=(nsamples,self.N,4))
        self.saveconstrains=tf.ones(shape=(nsamples,1,4))
        
        #Flux in case Ly is odd
        
        Flux=tf.zeros(shape=(nsamples,1), dtype=tf.int32)
        Number_of_white=tf.math.multiply(int(self.Lx/2),tf.ones(tf.shape(Flux),dtype=tf.int32))
        Number_of_black=tf.math.multiply(int(self.Lx/2),tf.ones(tf.shape(Flux),dtype=tf.int32))
        Only_check_for_connection_if_1=0
        
    
        for j in range(self.N):
            
        #Start Checking since this one
            if j==max(self.Lx*(self.Ly-5),self.Lx*int((self.Ly-1)/2)):
                Only_check_for_connection_if_1=1
                
            if Only_check_for_connection_if_1==1:
    
                Control_connection=tf.ones(shape=tf.shape(Check), dtype=tf.float32)
                for direction in range(4):
                    Check_for_connections=self.Check_for_boundaries(direction,j,Check)
                    Control_connection=tf.math.multiply(Control_connection,Check_for_connections)
                Check=tf.math.multiply(Check, Control_connection)
	##########################################################################################################################

            # Run a single RNN cell
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
#             print("rnn out",rnn_output)
            # Compute log probabilities
            probs = self.dense(rnn_output)
#             print("probs before change", probs)
            #Probability Changer
            Prob_changer=self.Prob_Changer(j, Check)
#             print(Prob_changer)
            #New Probability
            new_prob=tf.math.multiply(Prob_changer, probs)
            
            #Renormalization
            sum_probs=tf.reshape(tf.reduce_sum(new_prob,axis=2), shape=(nsamples,1,1))  
            probs=tf.math.divide_no_nan(new_prob,sum_probs)
#            print(j,"probs after change",probs)
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            
            #Save constrains 
            if j==0:
                self.saveconstrains=tf.math.multiply(self.saveconstrains, probs)
                self.saveconstrains=tf.math.divide_no_nan(self.saveconstrains,self.saveconstrains)
                
            else:
                self.saveconstrains=tf.concat([self.saveconstrains, probs], axis=1)
                self.saveconstrains=tf.math.divide_no_nan(self.saveconstrains,self.saveconstrains)
#             print(j," saveconstrains",self.saveconstrains)
            # Sample
            sample = tf.random.categorical(log_probs,num_samples=1)
#            print(j,"sample",sample)
            sample = tf.cast(sample,dtype=tf.int32 )
	#####################################################################################
            Control_sample_selection=self.control(j, sample,"True")
#             print(j,"Control from sample selection", Control_sample_selection)
            
            #Control the flux through the boundary
            if self.Ly%2 !=0:
                if j in self.upper_boundary:
#                     print("Number of white",Number_of_white," Number_of_black", Number_of_black)
                    Flux_Control,New_flux ,New_numb_white_available, New_numb_black_available=self.Flux_control(j,sample, Flux, Number_of_white, Number_of_black)
                    Control_sample_selection=tf.math.multiply(Control_sample_selection,Flux_Control)
                
                #Update Flux, and number of white and black available
                Flux=New_flux
                if j in self.upper_boundary and j==Lx-3:
                    last_one_checked_already=tf.multiply(-1,self.last_one_already_check)

                    Number_of_white=tf.math.add(New_numb_white_available,last_one_checked_already)

                    Number_of_black=New_numb_black_available

                    
                elif j in self.upper_boundary:
                    Number_of_white=New_numb_white_available
                    Number_of_black=New_numb_black_available

            Check=tf.math.multiply(Check,tf.cast(Control_sample_selection, dtype=tf.float32))


            #Control in case of one 
            Sum_Check=tf.reduce_sum(Check, axis=2)

            Look_for_sum_one=tf.equal(Sum_Check, 1)

            index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
            
#            print("index before while", index)

#            print("Value ",index.get_shape()[0])
            
            while index.get_shape()[0]!=0 and index.get_shape()[0]!=None:

                inpt=tf.gather_nd(Check,index)
#                 print("inpt",inpt)
                edges_to_check,_=tf.unique(tf.map_fn(lambda x: x[1], index))
#                 print("edges_to_check", edges_to_check)
                
                for edge in edges_to_check:
#                     print("edge being tested",edge)
                    multip_for_index=tf.zeros(shape=(tf.shape(index)[0]), dtype=tf.int32)

                    multip_for_index=tf.one_hot(multip_for_index, depth=2, on_value=self.Lx*self.Ly, off_value=0, dtype=tf.int32)

                    multip_index=tf.math.add(index, multip_for_index)

                    Look_for_edge_in_index=tf.equal(multip_index,edge)

                    Look_for_edge_in_index=tf.math.reduce_any(Look_for_edge_in_index, axis=1)

                    inpt_index=tf.cast(tf.where(Look_for_edge_in_index), dtype=tf.int32)
#                     print("inpt_indx",inpt_index)
                    inpts_to_concat=tf.gather_nd(inpt, inpt_index)
#                     print(j,"inpts_to_concat",inpts_to_concat)
                    number_of_sample=tf.map_fn(lambda x: index[x[0]][0], inpt_index)
#                     print("number_of_sample",number_of_sample)
                    inpts=tf.ones(shape=(4))
                    no_change=tf.zeros(shape=(4))

#                    print("index:",index)
#                    print("number_of_sample", number_of_sample)

                    for n in range(nsamples):
                        if n in number_of_sample:
                            find_this_sample_index=tf.reshape(tf.where(tf.equal(number_of_sample,n)),shape=(1,))
#                             print("find sample index",find_this_sample_index)
                            if n==0:
                                inpts=tf.math.multiply(inpts, inpts_to_concat[find_this_sample_index[0]])
                            else:
                                inpts=tf.concat([inpts,inpts_to_concat[find_this_sample_index[0]]], axis=0)
                        else:
                            if n==0:
                                inpts=tf.math.multiply(inpts, no_change)
                            else:
                                inpts=tf.concat([inpts, no_change], axis=0)
                    inpts=tf.reshape(inpts, shape=(tf.shape(Check)[0],4))

                    inpts=tf.cast(tf.reshape(tf.map_fn(lambda x: self.from_Check_to_direction(x), inpts),shape=(nsamples,1)),dtype=tf.int32)

                    Control_for_new_ones=tf.cast(self.control(edge, inpts,"True"),dtype=tf.float32)
                    Check=tf.multiply(Check,Control_for_new_ones)
            
                Sum_Check=tf.reduce_sum(Check, axis=2)

                Look_for_sum_one=tf.equal(Sum_Check, 1)

                index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
            
#             print(j,"Check after sum chek", Check)

            #Save samples
            if (j == 0):
                samples = tf.identity(sample)
                
            else:
                samples = tf.concat([samples,sample],axis=1)
 
            # Feed result to the next cell
            inputs = tf.one_hot(sample,depth=self.K)
            add = tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)
            logP = logP+add 
            
        return samples, logP
    
#    @tf.function
    
    def logpsi(self,samples):
        # Shift data
        num_samples = tf.shape(samples)[0]
        data   = tf.one_hot(samples[:,0:self.N-1],depth=self.K)
        x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K)
        inputs = tf.concat([x0,data],axis=1)

        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)
        new_prob     = tf.math.multiply(probs, tf.stop_gradient(self.saveconstrains))
        sum_probs=tf.reshape(tf.reduce_sum(new_prob,axis=2), shape=(self.ns,self.N,1))  
        probs=tf.math.divide_no_nan(new_prob,sum_probs)

#        print("probs in lopsi", probs)
        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),tf.one_hot(samples,depth=self.K)),axis=2)

        return tf.reduce_sum(log_probs,axis=1) 

#    @tf.function

    def localenergy(self,samples):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
        # Adding Parallel Horizontal
        for n in range(len(self.horizontal)):
            eloc += self.J * tf.cast(samples[:,self.horizontal[n][0]]*samples[:,self.horizontal[n][1]],tf.float32)
        #Adding Parallel Vertical
        for n in range(len(self.vertical)):
            eloc += self.J * tf.cast(samples[:,self.vertical[n][0]]*samples[:,self.vertical[n][1]],tf.float32)
        return eloc

    def from_Check_to_direction(self,Check_input):

        if tf.math.reduce_all(tf.equal(Check_input,tf.constant([1,0,0,0], dtype=tf.float32))):
            return 0
        elif tf.math.reduce_all(tf.equal(Check_input,tf.constant([0,1,0,0], dtype=tf.float32))):
            return 1
        elif tf.math.reduce_all(tf.equal(Check_input,tf.constant([0,0,1,0], dtype=tf.float32))):
            return 2
        elif tf.math.reduce_all(tf.equal(Check_input,tf.constant([0,0,0,1], dtype=tf.float32))):
            return 3
        else:
            return 4

    def from_direction_to_Check(self,x):
        if x==0:
            return tf.constant([1,0,0,0])
        if x==1:
            return tf.constant([0,1,0,0])
        if x==2:
            return tf.constant([0,0,1,0])
        if x==3:
            return tf.constant([0,0,0,1])

###########################################################################################################################

#Transform to the dimers model

    def Transform_to_dimers(self,samples):
    
        for sample in range(self.ns):
            if sample==0:
                dimers_position=self.position_in_dimers
            else:
                dimers_position=tf.concat([dimers_position,self.position_in_dimers], axis=0)
        dimers_position=tf.reshape(dimers_position, shape=(tf.shape(samples)[0], self.Lx*self.Ly, 4))

        sam=tf.reshape(samples, shape=(tf.shape(samples)[0]*tf.shape(samples)[1],))

        dimers=tf.reshape(tf.map_fn(lambda x: self.from_direction_to_Check(x), sam), shape=(tf.shape(samples)[0],tf.shape(samples)[1],4))

        edges_with_dimers=tf.math.multiply(dimers, dimers_position)    

        for edge in range(2*self.Lx*self.Ly):
            if edge==0:
                True_edge_has_dimer=tf.equal(edges_with_dimers, -1)
        #         print(edge,check_if_edge_has_dimer)
                reduce_True=tf.math.reduce_any(True_edge_has_dimer, axis=[1,2], keepdims=True)
        #         print(edge,reduce_True)
                check_if_edge_has_dimer=reduce_True
        #         print(edge,"check_if_edge_has_dimer",check_if_edge_has_dimer)
            else:
                True_edge_has_dimer=tf.equal(edges_with_dimers, edge)
        #         print(edge,check_if_edge_has_dimer)
                reduce_True=tf.math.reduce_any(True_edge_has_dimer, axis=[1,2], keepdims=True)
        #         print(edge,reduce_True)
                check_if_edge_has_dimer=tf.concat([check_if_edge_has_dimer,reduce_True],axis=1)
        #         print(edge,"check_if_edge_has_dimer",check_if_edge_has_dimer)

        dimers_samples=tf.reshape(tf.where(check_if_edge_has_dimer, 1,0), shape=(tf.shape(check_if_edge_has_dimer)[0],tf.shape(check_if_edge_has_dimer)[1]))

        return dimers_samples
    
    def number_of_errors(self,samples):
        errors = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
        for n in range(len(self.important_for_nodes)):
            if n==0:                
                errors+=tf.cast(samples[:, self.important_for_nodes[n][0]]+samples[:, self.important_for_nodes[n][1]]
                              +samples[:, self.important_for_nodes[n][2]]+samples[:, self.important_for_nodes[n][3]], dtype=tf.float32)
            else:
                nerror=tf.cast(samples[:, self.important_for_nodes[n][0]]+samples[:, self.important_for_nodes[n][1]]
                              +samples[:, self.important_for_nodes[n][2]]+samples[:, self.important_for_nodes[n][3]], dtype=tf.float32)
                errors=tf.concat([errors,nerror], axis=0)
#         print("self.important_for_nodes",self.important_for_nodes)
#         print("erros inside of errors",errors)        
        errors=tf.map_fn(lambda x: abs(x-1), errors)
        errors=tf.reshape(errors, shape=(self.Lx*self.Ly, tf.shape(samples)[0]))
        errors=tf.reduce_sum(errors, axis=0)
    
        return errors

    def coord_h(self,x,y):
        return 2*self.Lx*y+2*x
    #Gives coordinates of the x,y vertical edge
    def coord_v(self,x,y):
        return 2*self.Lx*y+1+2*x        
        
    def coord_dimers_h(self,x,y):
        h=2*self.Lx*y+2*x
        if h==0:
            h=-1
        return h
    #Gives coordinates of the x,y vertical edge
    def coord_dimers_v(self,x,y):
        v=2*self.Lx*y+1+2*x
        if v==0:
            v=-1
        return v
    
    def buildlattice_dimers(self):
        self.horizontal = []
        self.vertical = []
        self.adjacent=[]
        self.position_in_dimers=tf.ones(shape=(4), dtype=tf.int32)
        self.important_for_nodes=[]
        
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
    
       #Position in dimers 
        for y in range(self.Ly):
            for x in range(self.Lx):
                if y==0:
                    if x==0:
                        adjacent=tf.constant([self.coord_dimers_v(self.Lx-1,self.Ly-1), self.coord_dimers_v(self.Lx-1,y),
                                           self.coord_dimers_h(self.Lx-1,y), self.coord_dimers_h(x,y)], dtype=tf.int32)
                        self.position_in_dimers=tf.multiply(adjacent,self.position_in_dimers)

                    else:
                        adjacent=tf.constant([self.coord_dimers_v(x-1,self.Ly-1), self.coord_dimers_v(x-1,y),
                                           self.coord_dimers_h(x-1,y), self.coord_dimers_h(x,y)], dtype=tf.int32)
                        self.position_in_dimers=tf.concat([self.position_in_dimers,adjacent], axis=0)

                else:
                    if x==0:
                        adjacent=tf.constant([self.coord_dimers_v(x-1,y), self.coord_dimers_v(self.Lx-1,y),
                                           self.coord_dimers_h(self.Lx-1,y), self.coord_dimers_h(x,y)], dtype=tf.int32)
                        self.position_in_dimers=tf.concat([self.position_in_dimers,adjacent], axis=0)

                    else:
                        adjacent=tf.constant([self.coord_dimers_v(x-1,y-1), self.coord_dimers_v(x-1,y),
                                           self.coord_dimers_h(x-1,y), self.coord_dimers_h(x,y)], dtype=tf.int32)
                        self.position_in_dimers=tf.concat([self.position_in_dimers,adjacent], axis=0)
        self.position_in_dimers=tf.reshape(self.position_in_dimers, shape=(self.Lx*self.Ly,4))

        #important nodes set
        for y in range(self.Ly):
            for x in range(self.Lx):
                if y==0:
                    if x==0:
                         self.important_for_nodes.append( [self.coord_v(self.Lx-1,self.Ly-1),self.coord_h(self.Lx-1,y),self.coord_v(self.Lx-1,y),self.coord_h(x,y) ])
                    else:
                         self.important_for_nodes.append([self.coord_v(x-1,self.Ly-1),self.coord_h(x-1,y),self.coord_v(x-1,y),self.coord_h(x,y) ])
                elif x==0:
                     self.important_for_nodes.append([self.coord_v(self.Lx-1,y-1),self.coord_h(self.Lx-1,y),self.coord_v(self.Lx-1,y),self.coord_h(x,y) ]) 
                else:
                     self.important_for_nodes.append([self.coord_v(x-1,y-1),self.coord_h(x-1,y),self.coord_v(x-1,y),self.coord_h(x,y) ])

        
        
    def Check_sample_dimers(self,samples):
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
        self.Check_dimers=tf.math.multiply(Check_only_one,Check_number)

##########################################################################################################################
        
    # Function to get the edges that are available based on check, [sample, edge_comming from, edge_connected to]
    def Get_edge(self,x,Edges_available):
        a=tf.cast(x[0], dtype=tf.float32)
        b=tf.cast(x[1], dtype=tf.float32)
        c=tf.gather_nd(Edges_available, x)
        return tf.cast(tf.concat([a,b,c], axis=0),dtype=tf.int64)
    
    def depthFirst(self,sample_number,edge, Edges_available,ans):
    #     print("Visit before", self.Visit)
        indx=tf.one_hot(edge, depth=(self.Lx*self.Ly))
        new_true=tf.reshape(tf.equal(indx, 1), shape=tf.shape(self.Visit))
        self.Visit=tf.math.logical_or(self.Visit, new_true)
    #     print("initial ans", ans)
    #     print("Visit after", self.Visit)
        ans+=1
        k=0
        for connection in Edges_available[sample_number][edge]:
            if connection==0:
                k+=1
                if k==4:
    #                 print("k gave 4",k,"edge", edge, "connection", connection)
                    if self.Visit[connection]==False:
                        ans=self.depthFirst(sample_number, connection, Edges_available, ans)
    #                     print("edge",edge,"Visit after loop", self.Visit)
            else:
    #             print("edge", edge, "connection", connection)
                if self.Visit[connection]==False:
                    ans=self.depthFirst(sample_number, connection, Edges_available, ans)
    #                 print("edge",edge,"Visit after loop", self.Visit)

        return ans

# Binding Symmetries

def DSB(samples):
    dsb=tf.zeros(shape=(vmc.ns), dtype=tf.int32)
    N=vmc.Lx*vmc.Ly
    for n in range(2*N):
        if n%2==0:
            dsb+=samples[:,n]
        else:
            dsb-=samples[:,n]
    return 1/N*tf.cast(dsb,dtype=tf.float64)
def PSB(samples):
    psb=tf.zeros(shape=(vmc.ns), dtype=tf.int32)
    for vertical in vmc.vertical:
        
        psb-=samples[:,vertical[0]]*samples[:,vertical[1]]
        
    for horizontal in vmc.horizontal:
        
        psb+=samples[:,horizontal[0]]*samples[:,horizontal[1]]
    
    return psb

def BDSB(samples):
    dsb=DSB(samples)
    dsb4=tf.math.pow(dsb,4)
    dsb2=tf.math.pow(dsb,2)
    dsb4_mean=tf.math.reduce_mean(dsb4)
    dsb2_mean=tf.math.reduce_mean(dsb2)
    return 1-tf.math.divide(dsb4_mean, 3*tf.math.pow(dsb2_mean,2))

def BPSB(samples):
    psb=PSB(samples)
    psb4=tf.math.pow(psb,4)
    psb2=tf.math.pow(psb,2)
    psb4_mean=tf.math.reduce_mean(psb4)
    psb2_mean=tf.math.reduce_mean(psb2)
    return 1-tf.math.divide(psb4_mean, 3*tf.math.pow(psb2_mean,2))

def create_or_restore_training_state(checkpoint_dir):
    vmc=VariationalMonteCarlo(Lx,Ly,J,T,ns,nh,lr,epochs,seed)
    optimizer=vmc.optimizer
    epoch=tf.Variable(0)
    global_rng_state = tf.random.experimental.get_global_generator().state
    
    # create the checkpoint variable

    checkpoint = tf.train.Checkpoint(epoch=epoch,
                                     optimizer=optimizer,
                                     vmc=vmc, global_rng_state=global_rng_state)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    checkpoint_dir,
                                                    max_to_keep=3)

    # now, try to recover from the saved checkpoint, if successful, it should
    # re-populate the fields of the checkpointed variables.
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    checkpoint.restore(latest_checkpoint).expect_partial()
    if latest_checkpoint:
        global_rng_alg = tf.random.experimental.get_global_generator().algorithm
        tf.random.experimental.set_global_generator(
            tf.random.experimental.Generator(
                state=global_rng_state, alg=global_rng_alg))
        logging.info("training state restored at epoch {}".
              format(int(epoch.numpy()) ))
    else:
        logging.info("No checkpoint detected, starting from initial state")

    return vmc, optimizer, epoch, checkpoint_manager






#Training Step

#Logging information
logging.basicConfig(filename='information.log', level=logging.INFO, format='%(asctime)s:%(message)s')

#Look for checkpoints
vmc,optimizer,epoch, checkpoint_manager=create_or_restore_training_state(checkpoint_dir)
vmc.optimizer=optimizer
print("Running in checkpoint directory",checkpoint_dir)

with open("loss_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as Loss,\
open("ener_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as energy,\
open("free_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as free,\
open("vari_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as variance,\
open("bdsb_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as bdsb,\
open("bpsb_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as bpsb,\
open("errors_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as erro,\
open("allt_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as f:

    print("Num GPUs Acailable", len(tf.config.list_physical_devices('GPU')), file=f) 
    print(f"Running for size={Lx}x{Ly} ns={ns} nh={nh} and epochs={epochs}", file=f)
    while epoch < epochs+1:
        logging.info("Epoch {}, initiated".format(int(epoch.numpy())))
        print("epoch ",epoch.numpy(),file=f) 
        start_time=time.time()
        count=0
        samples, _ = vmc.sample(ns)
        samples_in_dimers=vmc.Transform_to_dimers(samples)
        print("time to sample",time.time()-start_time, file=f)

        Bind_d_symmetry_breaking=BDSB(samples_in_dimers)
        Bind_p_symmetry_breaking=BPSB(samples_in_dimers)
        print("bdsd=",Bind_d_symmetry_breaking, file=f)
        print("bpsd=",Bind_p_symmetry_breaking, file=f)

        print("samples", samples)
        print("in dimers",samples_in_dimers)
        print("##############Sample Generator finished################")
        vmc.Check_sample_dimers(samples_in_dimers)
        Check_correctness=vmc.Check_dimers
        print("Check_correctness",Check_correctness, file=f)
    
        # Evaluate the loss function in AD mode
        with tf.GradientTape() as tape:
            logpsi = vmc.logpsi(samples)

            eloc = vmc.localenergy(samples_in_dimers)
            print("eloc",eloc)
            errors=vmc.number_of_errors(samples_in_dimers)

            print("erros in each sample",errors, file=f)
            print("erros av", tf.stop_gradient(tf.math.reduce_mean(errors)), file=f)
#            errors=tf.math.multiply(50,errors)
            errors_factor=tf.random.uniform(shape=(tf.shape(errors)),minval=50, maxval=51)
            print("erros factor",errors_factor, file=f)
            print("eloc", eloc, file=f)
            eloc=tf.math.add(tf.math.multiply(errors_factor, errors), eloc)
            print("c*errors+eloc",eloc, file=f)
            Free_energy=tf.math.add(eloc, tf.math.scalar_mul(T, logpsi))
            print("free energy", Free_energy, file=f)
            Free_mean=tf.reduce_mean(Free_energy)
            print("free energy mean", Free_mean, file=f)
            loss = tf.reduce_mean(tf.multiply(logpsi,tf.stop_gradient(Free_energy-Free_mean)))
            print("loss",loss, file=f)
            print("#################################################################################################", file=f)
    
        # Compute the gradients
        gradients = tape.gradient(loss, vmc.trainable_variables)
    
        # Update the parameters
        vmc.optimizer.apply_gradients(zip(gradients, vmc.trainable_variables))
    
        energies = eloc.numpy()
        free_energies= Free_energy.numpy()
        bind_d_symmetry_breaking=Bind_d_symmetry_breaking.numpy()
        bind_p_symmetry_breaking=Bind_p_symmetry_breaking.numpy()

        avg_E = np.mean(energies)/float(N)
        avg_F = np.mean(free_energies)/float(N)
        var_E = np.var(energies)/float(N)
        avg_error=tf.math.reduce_mean(errors).numpy()
        #Save data in files 
        np.savetxt(bdsb,np.atleast_1d(bind_d_symmetry_breaking))
        logging.info("Epoch {}, bdsb saved".format(int(epoch.numpy())))
        
        np.savetxt(bpsb,np.atleast_1d(bind_p_symmetry_breaking))
        logging.info("Epoch {}, bdsb saved".format(int(epoch.numpy())))
        
        np.savetxt(energy,np.atleast_1d(avg_E))
        logging.info("Epoch {}, Energy saved".format(int(epoch.numpy())))
        
        np.savetxt(free,np.atleast_1d(avg_F))
        logging.info("Epoch {}, Free saved".format(int(epoch.numpy())))
        
        np.savetxt(variance,np.atleast_1d(var_E))
        logging.info("Epoch {}, Variance saved".format(int(epoch.numpy())))

        np.savetxt(Loss,np.atleast_1d(loss))
        logging.info("Epoch {}, Variance saved".format(int(epoch.numpy())))

        np.savetxt(erro, np.atleast_1d(avg_error))
        logging.info("Epoch {}, error saved".format(int(epoch.numpy())))
        
        #Save Check point 
        path=checkpoint_manager.save()
        logging.info("Epoch {}, Training state saved at {}".format(int(epoch.numpy()),path))
         
        epoch.assign_add(1)
	#Print several samples

        print("samples", file=f)
        count=0
        for i in samples_in_dimers:
            if count%2==0:
                print(i, file=f)
            count+=1
        print("end of samples", file=f)

global_rng_state = tf.random.experimental.get_global_generator().state
checkpoint = tf.train.Checkpoint(epoch=epoch,optimizer=optimizer,vmc=vmc, global_rng_state=global_rng_state)
checkpoint_manager = tf.train.CheckpointManager(checkpoint,current_dir,max_to_keep=3)
path=checkpoint_manager.save()
logging.info("Epoch {}, Training state saved at {}".format(int(epoch.numpy()),path))

import torch as torch
s=samples.numpy()
sample_dominos=torch.tensor(s)
torch.save(sample_dominos, f'samples_in_dominos_nh={nh}_size={Lx}x{Ly}x{ns}_T={T}_epoch={epoch.numpy()-1}.pt' )

s=samples_in_dimers.numpy()
sample_dimers=torch.tensor(s)
torch.save(sample_dimers, f'samples_in_dimers_nh={nh}_size={Lx}x{Ly}x{ns}_T={T}_epoch={epoch.numpy()-1}.pt' )

    


