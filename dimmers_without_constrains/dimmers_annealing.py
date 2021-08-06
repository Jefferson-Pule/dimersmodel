import tensorflow as tf
import numpy as np
import math
import logging
import time
import sys
import os

#Hamiltonian parameters
Lx=int(sys.argv[1])
Ly=int(sys.argv[2])
T=float(sys.argv[3])
N=2*Lx*Ly
J=1

#RNN parameters
ns=int(sys.argv[4])
epochs=int(sys.argv[5])
nh=int(sys.argv[6])
lr=0.001
seed=1234

#Directories
checkpoint_dir=sys.argv[7]  #Checkpoints directory
current_dir=sys.argv[8]	    #Current directory
os.chdir(current_dir)


# Create Model
class VariationalMonteCarlo(tf.keras.Model):

    # Constructor
    def __init__(self, Lx, Ly, J, 
                 T,number_of_samples ,num_hidden, learning_rate,
                 epochs, seed=1234):
        
        super(VariationalMonteCarlo, self).__init__()

        """ PARAMETERS """
        self.Lx       = Lx                           # Size along x
        self.Ly       = Ly                           # Size along y
        self.J        = -J                           # Strenght
        self.T        = T                            # Temperature 
        self.N        = 2*Lx*Ly                      # Number of edges
        self.ns       = ns                           # Number of samples
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
    
    #Controlers 
    
    def control(self,j, sample):

        sam=sample.numpy()
        important_edges=self.important[j][1:]    
        close_important_if_one=np.ones(shape=(sam.shape[0],self.N))
#         control_for_must_be_one=np.ones(shape=(sam.shape[0], self.N,2))

        change=-sam+1
        change=np.reshape(change,sam.shape[0])
        sam=np.reshape(sam,sam.shape[0])
        
        close_important_if_one[:,j]*=sam
        
        for m in important_edges:
            close_important_if_one[:,m]*=change

        return tf.convert_to_tensor(close_important_if_one, dtype=tf.float32)
    
    def prob_changer(self,j, check):
        indices=np.copy(self.colum_indices)
        indices[:,0,1]+=j
        indices=tf.constant(indices)
        value_in_check=tf.gather_nd(check, indices)
#         print(j,"value_in_check",value_in_check)
        #Apply x^2-x to make (1,0) give (0,0) and (2,2x) not zero 
        First_change=tf.math.add(tf.math.multiply(value_in_check,value_in_check),-value_in_check)
        
        #Divide by itslef to make all non zero values 1 
        First_change=tf.math.divide_no_nan(First_change,First_change)
        
        # Apply -x+1 to make all zero values 1 and all one values 
        First_change=tf.math.add(-First_change,1)

        #Divide by itself to make all non zero values 1
        Second_change=tf.math.divide_no_nan(value_in_check,value_in_check)

        prob_changer=tf.concat([First_change,Second_change],axis=1)
        prob_changer=tf.reshape(prob_changer,shape=(self.ns,1,2))
#         print(j,"prob_changer",prob_changer)
        return prob_changer

    # Generate Samples 
    
    def sample(self,nsamples):
        # Zero initialization for visible and hidden state 
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples,self.nh])
        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)
        
        Check_for_only_one=tf.ones(shape=(nsamples,self.N))

        self.saveconstrains=tf.ones(shape=(nsamples,1,self.K))
    
        for j in range(self.N):

            # Run a single RNN cell
            
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
            
            # Compute probabilities
            
            probs = self.dense(rnn_output)
#             print(j,"probs before change",probs)
#            print(Check_for_only_one)
            prob_changer=self.prob_changer(j,Check_for_only_one)
            new_probs=tf.math.multiply(prob_changer,probs)
            # Normalization 
            sum_probs=tf.reshape(tf.reduce_sum(new_probs,axis=2), shape=(nsamples,1,1))
            probs=tf.math.divide_no_nan(new_probs,sum_probs)
#             print(j,"probs after change",probs)
            
            # Compute log probabilities
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])

            #Single Sample generator
            
            sample = tf.random.categorical(log_probs,num_samples=1)
#             print("sample",sample)
                                           
            #Change Checks
            control_only_one=vmc.control(j,sample)
            Check_for_only_one=tf.math.multiply(Check_for_only_one,control_only_one)
#             print(j,"Check_for_only_one",Check_for_only_one)
            #Save Samples
            
            if (j == 0):
                samples = tf.identity(sample)
                
            else:
                samples = tf.concat([samples,sample],axis=1)
        
            # Feed result to the next cell
            
            inputs = tf.one_hot(sample,depth=self.K)
            
            add = tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)

            logP = logP+add
            
            #Save the conditions of the probabilitites
            
            if j==0:
                self.saveconstrains=tf.math.multiply(self.saveconstrains, probs)
                self.saveconstrains=tf.math.divide_no_nan(self.saveconstrains,self.saveconstrains)
                
            else:
                self.saveconstrains=tf.concat([self.saveconstrains, probs], axis=1)
                self.saveconstrains=tf.math.divide_no_nan(self.saveconstrains,self.saveconstrains)
#             print(j,"saveconstrains",self.saveconstrains)
    
        return samples, logP
    
    #Calculate probailities in paralel

#     @tf.function
    def logpsi(self,samples):
        # Shift data
        num_samples = tf.shape(samples)[0]
        data   = tf.one_hot(samples[:,0:self.N-1],depth=self.K)
        x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K)
        inputs = tf.concat([x0,data],axis=1)

        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)
        new_probs    =tf.math.multiply(probs, self.saveconstrains)
        sum_probs=tf.reshape(tf.reduce_sum(new_probs,axis=2), shape=(self.ns,self.N,1))  
        probs=tf.math.divide_no_nan(new_probs,sum_probs)
#        print("probs in logpsi",probs)
        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),tf.one_hot(samples,depth=self.K)),axis=2)

        return tf.reduce_sum(log_probs,axis=1)
    
    #Calculate Energy 

#     @tf.function
    def localenergy(self,samples):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
        # Adding Parallel Horizontal
        for n in range(len(self.horizontal)):
            eloc += self.J * tf.cast(samples[:,self.horizontal[n][0]]*samples[:,self.horizontal[n][1]],tf.float32)
        #Adding Parallel Vertical
        for n in range(len(self.vertical)):
            eloc += self.J * tf.cast(samples[:,self.vertical[n][0]]*samples[:,self.vertical[n][1]],tf.float32)
        return eloc
    
#    @tf.function
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
                
        errors=tf.map_fn(lambda x: abs(x-1), errors)
        errors=tf.reshape(errors, shape=(self.Lx*self.Ly, tf.shape(samples)[0]))
        errors=tf.reduce_sum(errors, axis=0)
    
        return errors

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
        self.important_for_nodes=[]
        self.colum_indices=[]
        
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

        for n in range(self.ns):
            self.colum_indices.append([[n,0]])
        self.colum_indices=np.array(self.colum_indices)
        
    def check_sample(self,samples):
    # Check Only one Sample per site returns 1 if correct 0 otherwise
        only_one_error_sum=tf.zeros(shape=tf.shape(samples)[0])    

        #Check errors in samples and add them 
        for n in range(len(self.adjacent)):
            only_one_error_sum+=tf.cast(samples[:,self.adjacent[n][0]]*samples[:,self.adjacent[n][1]],tf.float32)
        #Make all errors 1 and all success 0
        check_only_one=tf.math.divide_no_nan(only_one_error_sum,only_one_error_sum)
        #Make all erros 1 and all success 1
        check_only_one=tf.math.add(-check_only_one,1)

        # Check That all have at least one:
        expected_number_of_dimers= int(0.5*Lx*Ly)
        number_of_dimers=tf.math.reduce_sum(samples, axis=1)

        check_number=tf.cast(tf.math.add(number_of_dimers,-expected_number_of_dimers), dtype=tf.float32)

        check_number=tf.math.divide_no_nan(check_number,check_number)

        check_number=tf.math.add(-check_number,1)
    
        #Final check 
        self.check=tf.math.multiply(check_only_one, check_number)


# Initializing model

vmc = VariationalMonteCarlo(Lx,Ly,J,T,ns,nh,lr,epochs,seed)

# Binding Symmetries

def DSB(samples):
    dsb=tf.zeros(shape=(vmc.ns), dtype=tf.int64)
    N=vmc.Lx*vmc.Ly
    for n in range(2*N):
        if n%2==0:
            dsb+=samples[:,n]
        else:
            dsb-=samples[:,n]
    return 1/N*tf.cast(dsb,dtype=tf.float64)
def PSB(samples):
    psb=tf.zeros(shape=(vmc.ns), dtype=tf.int64)
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

# Checkpoints 
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

#Logging information

logging.basicConfig(filename='information_for_T={}.log'.format(T), level=logging.INFO, format='%(asctime)s:%(message)s', force=True)

#Look or load for checkpoints 

vmc,optimizer,epoch, checkpoint_manager=create_or_restore_training_state(checkpoint_dir)
vmc.optimizer=optimizer
print("Running in checkpoint directory",checkpoint_dir)

# Training Step
while T>=0.5:
    with open("loss_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as Loss,\
    open("ener_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as energy,\
    open("free_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as free,\
    open("vari_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T),"a") as variance,\
    open("bdsb_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as bdsb,\
    open("bpsb_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as bpsb,\
    open("errors_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as erro,\
    open("allt_nh_{}_size_{}x{}x{}_T_{}.txt".format(nh,Lx,Ly,ns,T), "a") as f:

        print("Num GPUs Acailable", len(tf.config.list_physical_devices('GPU')), file=f) 
        print(f"Running for T={T} size={Lx}x{Ly} ns={ns} nh={nh} and epochs={epochs}", file=f)        
    
        while epoch < epochs+1:
            print("epoch ",epoch.numpy(),file=f)   
            start_time=time.time()
            samples, _ = vmc.sample(ns)
            print("time to sample",time.time()-start_time, file=f)
        
            Bind_d_symmetry_breaking=BDSB(samples)
            Bind_p_symmetry_breaking=BPSB(samples)
            print("bdsd=",Bind_d_symmetry_breaking, file=f)
            print("bpsd=",Bind_p_symmetry_breaking, file=f)

            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                logpsi = vmc.logpsi(samples)
                eloc = vmc.localenergy(samples)
                errors=vmc.number_of_errors(samples)
                print("erros in each sample",errors, file=f)
                print("erros av", tf.stop_gradient(tf.math.reduce_mean(errors)), file=f)
                errors=tf.math.multiply(-3,errors)
                Free_energy=tf.math.add(eloc, tf.math.scalar_mul(T, logpsi))
                Free_mean=tf.reduce_mean(Free_energy)
                loss = tf.reduce_mean(tf.multiply(logpsi,tf.add(tf.stop_gradient(errors),tf.stop_gradient(Free_energy-Free_mean))))
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
	    avg_error=np.tf.math.reduce_mean(errors)
        
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
        
            #Add one to epoch
            epoch.assign_add(1)
	    #Print several samples

            print("samples", file=f)
            count=0
            for i in samples:
                if count%2==0:
                    print(i, file=f)
                count+=1
            print("end of samples", file=f)
        
    #Save last checkpoint in current directory

    #global_rng_state = tf.random.experimental.get_global_generator().state
    #checkpoint = tf.train.Checkpoint(epoch=epoch,optimizer=optimizer,vmc=vmc, global_rng_state=global_rng_state)
    #checkpoint_manager = tf.train.CheckpointManager(checkpoint,current_dir,max_to_keep=3)
    #path=checkpoint_manager.save()
    #logging.info("Epoch {}, Training state saved at {}".format(int(epoch.numpy()),path))

    import torch as torch
    s=samples.numpy()
    sample_dimers=torch.tensor(s)
    torch.save(sample_dimers, f'samples_nh={nh}_size={Lx}x{Ly}x{ns}_T={T}_epoch={epoch.numpy()-1}.pt' )
    T-=0.025 
    T=round(T, 4 - int(math.floor(math.log10(abs(T)))) - 1)
    epoch.assign(0)
