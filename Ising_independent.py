import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy.misc import factorial
from scipy import sparse
import io
import time
from matplotlib import colors
cmap = colors.ListedColormap(['black', 'white'])
np.random.seed(2017)
import scipy.stats as st
import tqdm
# import threading
import multiprocessing
from multiprocessing import Queue


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

    _EPOCHS = int(1e5)

    def __init__(self, q,inv_tempature=5, initial_state='r', size=(100,100), sp=0.5,show=False):
        # threading.Thread.__init__(self)
        super(IsingLattice, self).__init__()
        self.sqr_size = size
        self.size = size[0]
        self.T = inv_tempature
        self._build_system(initial_state)
        self.q = q
    

        self.create_A(sp)
        self.A_list = []
        self.sp = sp

        self.errors = []
     
        self.points = self._EPOCHS
        
    
        self.D = 2
        self.marginals = np.zeros((self.D,self.sqr_size[0],self.sqr_size[0]))   

        # true distribution 
        self.true = np.ones((self.D, self.sqr_size[0],self.sqr_size[0]))/self.D

        self.states = [-1,1]
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
        if initial_state == 'r':
            system = np.random.randint(0, 2, self.sqr_size)
            system[system==0] = -1
        else:
            system = np.ones(self.sqr_size)
        

        self.system = system
        print("system. ",self.magnetization)
        # print(system)
        
    def _energy(self, N, M):
        """Calculate the energy of spin interaction at a given lattice site
        i.e. the interaction of a Spin at lattice site n,m with its 4 neighbors
        - S_n,m*(S_n+1,m + Sn-1,m + S_n,m-1, + S_n,m+1)
        Args
        ----
        N (int) : lattice site coordinate
        M (int) : lattice site coordinate
        Return
        """
        
        m = np.random.rand(1,self.size**2)
        m[m<self.sp]=0
        m[m>=self.sp]=1

        W = (m*self.A[:,[self.size*N+M]].reshape(1,-1))
        return np.dot((self.system.flatten()),(W.T)*self.system[(N),(M)])


    @property
    def magnetization(self):
        """Find the overall magnetization of the system
        """
        print(np.sum(self.system),self.size**2 )
        
        return np.abs(float(np.sum(self.system))/self.size**2)





    def calc_marginals(self):
        # marginals is a DxNxN tensor
        # i goes from 0 to D-1
        # counts the number of times state d occurs 
        for i in range(self.D):
            self.marginals[i]+=self.system==self.states[i]

    def compute_error(self,epoch):
 
        # frequency of each state 
        # print('de',((epoch-(self._EPOCHS-self.points)+1)))
        # print('raw counts',self.marginals)
        y_bar = self.marginals/((epoch-(self._EPOCHS-self.points)+1))
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
        print(str(self.sp)+" Net Magnetization: {:.2f}".format(self.magnetization))
        print(' started! '+str(self.sp))
        for epoch in tqdm.trange(self._EPOCHS,desc='Gibbs step'):
        # for epoch in range(self._EPOCHS):
            # Randomly select a site on the lattice


            N, M = np.random.randint(0, self.size, 2)

            # Calculate energy of a flipped spin


            # current energy
            Ex = self._energy(N,M)
            #flip the state
            self.system[N,M]*=-1
            Ey = self._energy(N,M)



           # Using Gibbs sampling
            p = np.exp(Ey*self.T)/(np.exp(Ex*self.T)+np.exp(Ey*self.T))

            if np.random.random() <= p:
                # if likely, accept the change
                pass
            else:
                # revert back to before
                self.system[N,M] *=-1 

            #marginals
            # hack to record errors for 200 iterations
            if epoch>=(self._EPOCHS-self.points):
                self.calc_marginals()
                error = self.compute_error(epoch)
                self.errors.append(error)
        
        print(str(self.sp)+" Net Magnetization: {:.2f}".format(self.magnetization))


        print('final error: ',self.errors[-1])
        self.q.put([[self.sp]+self.errors])
        return True

if __name__ == "__main__":


    marginals = []

    tasks = []
    q = multiprocessing.Manager().Queue()
    for i in range(3):
        tasks.append(IsingLattice(q,inv_tempature=11, initial_state="r", size=(50,50),sp=float(i)/20))


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
        print "Keyboard interrupt in main"
   
    finally:
        print "Cleaning up Main"

    while q:
        try:
            item = q.get(block=False)
            # print(item)
            marginals.append(item)
        except :
            break

    print(marginals)
    with open('Ising_Indep_{}_{}.csv'.format(100,1e3),'wb') as f:
        for i in marginals:
            np.savetxt(f, i,delimiter=',')





