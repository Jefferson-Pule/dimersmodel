import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch as torch
import sys

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

        indices=tf.constant([[0,j]])

        for s in range(tf.shape(Check)[0]-1):
            indices=tf.concat([indices,tf.constant([[s+1,j]])], axis=0)
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
        for i in range(tf.shape(Check)[0]-1):
            sample_checker=tf.concat([sample_checker,sample_checker_1], axis=0)

        sample_checker=tf.reshape(sample_checker, shape=(tf.shape(Check)[0],1))

#         print("sample_checker",sample_checker)

        Control_checker=tf.cast(self.control(j, sample_checker, "True"), dtype=tf.float32)
        Control_checker=tf.multiply(Check, Control_checker)
################################################        
        #Control in case of one 
        Sum_Check=tf.reduce_sum(Control_checker, axis=2)

        Look_for_sum_one=tf.equal(Sum_Check, 1)

        index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
#             print(j,"index", index)

        while tf.shape(index)[0]!=0:

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
                inpts=tf.reshape(inpts, shape=(tf.shape(Check)[0],4))

                inpts=tf.cast(tf.reshape(tf.map_fn(lambda x: self.from_Check_to_direction(x), inpts),shape=(self.ns,1)),dtype=tf.int32)

                Control_for_new_ones=tf.cast(self.control(edge, inpts,"True"),dtype=tf.float32)
                Control_checker=tf.multiply(Control_checker,Control_for_new_ones)

            Sum_Check=tf.reduce_sum(Control_checker, axis=2)

            Look_for_sum_one=tf.equal(Sum_Check, 1)

            index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)

        print("sample",j,"direction",direction_being_check,"Control checker after sum chek", Control_checker)
                        
################################################


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
        print("sample",j,"direction being checked",direction_being_check,"boundaries_changer",Boundaries_changer)

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


        
#     @tf.function
    
    # Creates a nsamples with N entries.
        
        
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)

        hidden_state = tf.zeros(shape=[nsamples,self.nh])

        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)
        
        #Check
        
        Check=tf.ones(shape=(nsamples,self.N,4))

        
        #Flux in case Ly is odd
        
        Flux=tf.zeros(shape=(nsamples,1), dtype=tf.int32)
        Number_of_white=tf.math.multiply(int(self.Lx/2),tf.ones(tf.shape(Flux),dtype=tf.int32))
        Number_of_black=tf.math.multiply(int(self.Lx/2),tf.ones(tf.shape(Flux),dtype=tf.int32))
        Only_check_for_connection_if_1=0
        for j in range(self.N):
# ##########################################################################################################################

##########################################################################################################################
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

#         #Start Checking since this one
#             if j==max(self.Lx*(self.Ly-5),self.Lx*int((self.Ly-1)/2)):
#                 Only_check_for_connection_if_1=1

#             if Only_check_for_connection_if_1==1:
#                 Control_connection=tf.ones(shape=tf.shape(Check), dtype=tf.float32)
#                 for direction_being_check in range(4):
#                     sample_checker_1=tf.constant([[direction_being_check]])
#                     sample_checker=sample_checker_1
#         #             print("sample_checker before",sample_checker)
#                     for i in range(tf.shape(Check)[0]-1):
#                         sample_checker=tf.concat([sample_checker,sample_checker_1], axis=0)

#                     sample_checker=tf.reshape(sample_checker, shape=(tf.shape(Check)[0],1))

#         #             print("sample_checker",sample_checker)

#                     Control_sample_selection=self.control(j, sample_checker,"True")
#         #             print(j,"Control from sample selection", Control_sample_selection)
#                     #Control in case of one 
#                     Sum_Check=tf.reduce_sum(Check, axis=2)

#                     Look_for_sum_one=tf.equal(Sum_Check, 1)

#                     index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
#         #             print(j,"index", index)

#                     while tf.shape(index)[0]!=0:

#                         inpt=tf.gather_nd(Check,index)
#         #                 print("inpt",inpt)
#                         edges_to_check,_=tf.unique(tf.map_fn(lambda x: x[1], index))
#         #                 print("edges_to_check", edges_to_check)

#                         for edge in edges_to_check:
#         #                     print("edge being tested",edge)
#                             multip_for_index=tf.zeros(shape=(tf.shape(index)[0]), dtype=tf.int32)

#                             multip_for_index=tf.one_hot(multip_for_index, depth=2, on_value=self.Lx*self.Ly, off_value=0, dtype=tf.int32)

#                             multip_index=tf.math.add(index, multip_for_index)

#                             Look_for_edge_in_index=tf.equal(multip_index,edge)

#                             Look_for_edge_in_index=tf.math.reduce_any(Look_for_edge_in_index, axis=1)

#                             inpt_index=tf.cast(tf.where(Look_for_edge_in_index), dtype=tf.int32)
#         #                     print("inpt_indx",inpt_index)
#                             inpts_to_concat=tf.gather_nd(inpt, inpt_index)
#         #                     print(j,"inpts_to_concat",inpts_to_concat)
#                             number_of_sample=tf.map_fn(lambda x: index[x[0]][0], inpt_index)
#         #                     print("number_of_sample",number_of_sample)
#                             inpts=tf.ones(shape=(4))
#                             no_change=tf.zeros(shape=(4))

#                             for n in range(nsamples):
#                                 if n in number_of_sample:
#                                     find_this_sample_index=tf.reshape(tf.where(tf.equal(number_of_sample,n)),shape=(1,))
#         #                             print("find sample index",find_this_sample_index)
#                                     if n==0:
#                                         inpts=tf.math.multiply(inpts, inpts_to_concat[find_this_sample_index[0]])
#                                     else:
#                                         inpts=tf.concat([inpts,inpts_to_concat[find_this_sample_index[0]]], axis=0)
#                                 else:
#                                     if n==0:
#                                         inpts=tf.math.multiply(inpts, no_change)
#                                     else:
#                                         inpts=tf.concat([inpts, no_change], axis=0)
#                             inpts=tf.reshape(inpts, shape=(tf.shape(Check)[0],4))

#                             inpts=tf.cast(tf.reshape(tf.map_fn(lambda x: self.from_Check_to_direction(x), inpts),shape=(nsamples,1)),dtype=tf.int32)

#                             Control_for_new_ones=tf.cast(self.control(edge, inpts,"True"),dtype=tf.float32)
#                             Check=tf.multiply(Check,Control_for_new_ones)

#                         Sum_Check=tf.reduce_sum(Check, axis=2)

#                         Look_for_sum_one=tf.equal(Sum_Check, 1)

#                         index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)

#                     print("For",j,"in direction",direction_being_check,"Check after sum chek", Check)

#                     Sum_Check=tf.reduce_sum(Check, axis=2)
#                     Look_for_sum_zero=tf.equal(Sum_Check, 0)
#                     indx_where_zero=tf.where(Look_for_sum_zero)

#                     if tf.shape(indx_where_zero)[0]!=0:
#                         print("For",j,"in direction",direction_being_check,"indx_where_zero",indx_where_zero)
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
            print(j,"probs after change",probs)
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            
            # Sample
            sample = tf.random.categorical(log_probs,num_samples=1)
            print(j,"sample",sample)
            sample = tf.cast(sample,dtype=tf.int32 )
#####################################################################################
            Control_sample_selection=self.control(j, sample,"True")
#             print(j,"Control from sample selection", Control_sample_selection)
            
            #Control the flux through the boundary
            if self.Ly%2 !=0:
                if j in self.upper_boundary:
#                     print("Number of white",Number_of_white," Number_of_black", Number_of_black)
                    Flux_Control,New_flux ,New_numb_white_available, New_numb_black_available=self.Flux_control(j,sample, Flux, Number_of_white, Number_of_black)
#                     print("Control Flux 11",j, Flux_Control[11])
#                     print("Control Flux 94",j, Flux_Control[94])
#                     print("Control Flux 53",j, Flux_Control[53])
                    Control_sample_selection=tf.math.multiply(Control_sample_selection,Flux_Control)
                
                #Update Flux, and number of white and black available
                Flux=New_flux
                if j in self.upper_boundary and j==Lx-3:
                    last_one_checked_already=tf.multiply(-1,self.last_one_already_check)
#                     print("las_one_checked_already 11", last_one_checked_already[11])
#                     print("las_one_checked_already 94", last_one_checked_already[94])
#                     print("las_one_checked_already 53", last_one_checked_already[53])
                    Number_of_white=tf.math.add(New_numb_white_available,last_one_checked_already)
#                     print("Number of white 11",Number_of_white[11])
#                     print("Number of white 94",Number_of_white[94])
#                     print("Number of white 53",Number_of_white[53])
                    Number_of_black=New_numb_black_available
#                     print("Number of black 11",Number_of_black[11])
#                     print("Number of black 94",Number_of_black[94])
#                     print("Number of black 53",Number_of_black[53])
                    
                elif j in self.upper_boundary:
                    Number_of_white=New_numb_white_available
                    Number_of_black=New_numb_black_available
#                     print("Number of white 11",Number_of_white[11])
#                     print("Number of white 94",Number_of_white[94])
#                     print("Number of black 53",Number_of_black[53])
#                     print("Number of black 11",Number_of_black[11])
#                     print("Number of black 94",Number_of_black[94])
#                     print("Number of black 53",Number_of_black[53])

#                 print(j,"Flux 11", Flux[11])
#                 print(j,"Flux 94", Flux[53])
#                 print(j,"Flux 53", Flux[94])

            Check=tf.math.multiply(Check,tf.cast(Control_sample_selection, dtype=tf.float32))
#             print(j,"Check after flux 11", Check[11])
#             print(j,"Check after flux 94", Check[94])
#             print(j,"Check after flux 53", Check[53])

            #Control in case of one 
            Sum_Check=tf.reduce_sum(Check, axis=2)

            Look_for_sum_one=tf.equal(Sum_Check, 1)

            index=tf.cast(tf.where(Look_for_sum_one), dtype=tf.int32)
#             print(j,"index", index)
            
            while tf.shape(index)[0]!=0:

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
            
            print(j,"Check after sum chek", Check)

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





# Hamiltonian parameters

#Lx = 14      # Linear size in x direction
Lx = int(sys.argv[1])      # Linear size in x direction
Ly = int(sys.argv[2])      # Linear size in y direction
N = Lx*Ly   # Total number of spins 
J = 1.0     # Strenght  
T = 2.5     # Temperature
# RNN-VMC parameters
lr = 0.001     # learning rate of Adam optimizer
nh = 10        # Number of hidden units in the GRU cell
ns = int(sys.argv[3])       # Number of samples used to approximate the energy at each step
epochs = 1000  # Training iterations
seed = 1234    # Seed of RNG
vmc = VariationalMonteCarlo(Lx,Ly,J,T,ns,nh,lr,epochs,seed)
#########I do not know the exact energy#########
# Exact energy
#exact_energy =






samples,_=vmc.sample(ns)



samples_to_check=samples
print(samples_to_check)
#Check for sample
dimers_samples=vmc.Transform_to_dimers(samples_to_check)
vmc.Check_sample_dimers(dimers_samples)
Check_correctness=vmc.Check_dimers
print(Check_correctness)




s=samples.numpy()
print(s)
sample_100_dominos=torch.tensor(s)
print(sample_100_dominos)
torch.save(sample_100_dominos, 'sample_100_dominos14x14checkfinal.pt')





