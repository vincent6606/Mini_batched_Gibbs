import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy import sparse
import io
import time
from matplotlib import colors
cmap = colors.ListedColormap(['black', 'white'])
# np.random.seed(2017)
import scipy.stats as st
import tqdm
import multiprocessing
from multiprocessing import Queue
import argparse



parser = argparse.ArgumentParser(description='Potts model Simulation')
parser.add_argument('--size', default=50, type=int,
                    help='size of graph(NxN)')

parser.add_argument('--epoch', default=1000000, type=int,
                    help='number of epochs to run')

parser.add_argument('--no_output', default = False, action='store_true')





def gkern(kernlen=21, nsig=5):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def make_A(N):
    a = gkern(2*N-1)
    A = np.zeros([N**2,N**2])
    k = 0
    for j in range(N):
        for i in range(N):
            A[:,(N**2-1)-k] = a[j:j+N,i:i+N].flatten()
            k+=1
    return A
    


class IsingLattice(multiprocessing.Process):

    

    def __init__(self, q,epoch,temp=5,lam=10,initial_state='r', size=(100,100), sp=0.5,show=False):
        # threading.Thread.__init__(self)
        super(IsingLattice, self).__init__()
        self.sqr_size = size
        self.size = size[0]
        self.T = temp
        self._build_system(initial_state)
        self.q = q
    

        self.create_A(sp)
        self.A_list = []
        self.sp = sp

        self.errors = []
        self._EPOCHS = epoch
        self.points = epoch
        
    
        self.D = 2
        self.marginals = np.zeros((self.D,self.sqr_size[0],self.sqr_size[0]))   

        # true distribution 
        self.true = np.ones((self.D, self.sqr_size[0],self.sqr_size[0]))/self.D
        self.states = [-1,1]


        self.psi = np.sum(2*self.A)
        # self.lam = int(lam*self.psi**2)
        self.lam=int(lam*self.psi**2)
        print(self.lam,'lam')
        print(self.psi,'psi')

    def create_A(self,sp):
        #create mask
        self.A = make_A(self.size)
        np.fill_diagonal(self.A, 0)
        self.A = self.A
        
        
        
    def _build_system(self, initial_state):
        """Build the system
        Build either a randomly distributed system or a homogeneous system (for
        watching the deterioration of magnetization
        Args
        ----
        initial_state (str: "r" or other) : Initial state of the lattice.
            currently only random ("r") initial state, or uniformly magnetized, 
            is supported 
        """

        np.random.seed(2017)
        system = np.ones(self.sqr_size,dtype=int)
        self.system = system

    
    def _energy(self):
        # M_phi_ij = 2Aij because Ising is [-1,1], hence (self.A*2)
        S = (self.lam*(self.A*2)/(self.T*2))/self.psi
        S = np.random.poisson(S,S.shape) # prevent double counting
        eu = np.log(1+self.psi/(self.lam))
        X = self.system.flatten()
        # divide by 2 in the end to match the indicator function
        energy = eu*(np.dot(X.T,np.dot(S,X))+np.sum(S))/2



        # e = 0
        # for i in [-1,1]:
        #     X = self.system.flatten() == i
        #     e +=np.dot(X.T,np.dot(self.A,X))
        # true = e/self.T/2
        # print(true - energy)

        return energy

    def calc_marginals(self):
        # marginals is a DxNxN tensor
        # i goes from 0 to D-1
        # counts the number of times state d occurs 
        for i in range(self.D):

            self.marginals[i]+=self.system==self.states[i]
            # print(marginals)
    def compute_error(self,epoch):
 
        # frequency of each state 
        # print('de',((epoch-(self._EPOCHS-self.points)+1)))
        # print('raw counts',self.marginals)
        y_bar = self.marginals/(epoch+1)
        # print('frequency',y_bar)

        y_bar -= self.true
        y_bar = y_bar.reshape(self.D,-1)
        # print('yb',y_bar)
        # norm of the distance away from true distribution
        # print('mean',np.mean((np.linalg.norm(y_bar,axis=0))))
        return np.mean((np.linalg.norm(y_bar,axis=0)))

    def run(self):
        """Run the simulation
        """
        print(' started! '+str(self.sp))

        # estimate energy for the first iteration
        Ex = self._energy()
        print('initial energy',Ex)

        for epoch in tqdm.trange(self._EPOCHS,desc='Gibbs step'):
        # for epoch in range(self._EPOCHS):
            # Randomly select a site on the lattice


            N, M = np.random.randint(0, self.size, 2)

            # flip the variable to y
            self.system[N,M] *= -1
            Ey = self._energy()
            # print('Ey',Ey,'Ex',Ex)

            # flip the state
            p = np.exp(Ey)/(np.exp(Ey)+np.exp(Ex))

            # print('p',p)


            if np.random.random() <= p:
                # if likely, accept the change
                Ex = Ey
            else:
                # revert back to before
                # Ex does not change
                self.system[N,M] *= -1

            # print('fy',Ey)
            # print('fx',Ex)
        
 
            
            self.calc_marginals()
            error = self.compute_error(epoch)
            self.errors.append(error)
            # print('e',error)
        
        # print(str(self.sp)+" Net Magnetization: {:.2f}".format(self.magnetization))


        print('final error: ',self.errors[-1])
        self.q.put([[self.lam]+self.errors])
        return True

if __name__ == "__main__":

    # variable to store the errors from marginals
    global args
    args = parser.parse_args()
    marginals = []

    tasks = []
    q = multiprocessing.Manager().Queue()
    sparsity = [1,2,4]
    # temperature = [10,12,14,18,20]
    for i in sparsity:
        tasks.append(IsingLattice(q,epoch=args.epoch,temp=0.7,lam=i, initial_state="r", size=(args.size,args.size),sp=float(i)))

    for task in tasks:
        task.start()

    try:
        for task in tasks:
            task.join()
    # handle keyboard interrupts to avoid un-closed processes 
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
            task.join()
        print("Keyboard interrupt in main")
   
    finally:
        print("Cleaning up Main")

    while q:
        try:
            item = q.get(block=False)
            # print(item)
            marginals.append(item)
        except :
            break

    if args.no_output == False:
        with open('MIN_Ising_Dep_{}_{}.csv'.format(args.size,args.epoch),'wb') as f:
            for i in marginals:
                np.savetxt(f, i,delimiter=',')






