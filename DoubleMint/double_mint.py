

import numpy as np

from scipy import sparse
import io

import tqdm
import multiprocessing
import csv
from multiprocessing import Queue
import argparse
# np.random.seed(2012)
import scipy.stats as st


parser = argparse.ArgumentParser(description='MGPMH Potts model Simulation')
parser.add_argument('--size', default=50, type=int,
                    help='size of graph(NxN)')

parser.add_argument('--epoch', default=1e5, type=int,
                    help='number of epochs to run')

parser.add_argument('--lam', default=2, type=int,
                    help='factor or lambda')

parser.add_argument('--states', default=10, type=int,
                    help='factor or lambda')

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
    A = np.zeros([N**2, N**2])
    k = 0
    for j in range(N):
        for i in range(N):
            A[:, (N**2-1)-k] = a[j:j+N, i:i+N].flatten()
            k += 1
    return A


class PottsLattice(multiprocessing.Process):

    

    def __init__(self,q, temp, lam1,lam2,states, epoch=1e6,initial_state='r', size=(100, 100), show=False):
        super(PottsLattice, self).__init__()

        self.sqr_size = size
        self.size = size[0]
        self.T = temp
        self._build_system(initial_state)

        # marginals
        self.create_A()
        # 10 states
        self.D = states
        self.states = range(1, self.D+1)

        # L is the maximum sum of factors
        self.L = self.A.sum(axis=0).max()
        # factor of lambda
        
        self.lam1 = int(lam1*(self.L**2))
        self.psi = np.sum(self.A)
        self.lam2 = int(lam2*(self.psi**2))
        # self.lam2 = int(lam2)
        print(self.lam1)
        print(self.lam2)
        print(self.psi,'psi')
        print(self.L,'L')

        # true distribution 
        self.true = np.ones((self.D, self.sqr_size[0],self.sqr_size[0]))/self.D
        self.q = q
        self.errors = []
        self._EPOCHS = epoch
        self.points = epoch
        self.marginals = np.zeros((self.D,self.sqr_size[0],self.sqr_size[0]))



    def create_A(self):
        # create mask
        self.A = make_A(self.size)
        np.fill_diagonal(self.A, 0)

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
        # np.random.seed(2017)
        if initial_state == 'r':
            system = np.ones(self.sqr_size, dtype=int)
            # system = np.random.randint(1,11,self.sqr_size,dtype=int)
        else:
            system = np.ones(self.sqr_size)

        self.system = system

        # print(system)
    # @profile
    def _energy(self, N, M, state, s):
        """Calculate the energy of spin interaction at a given lattice site for a given state

        Args
        ----
        N (int) : lattice site coordinate
        M (int) : lattice site coordinate
        Return
        """
        # get phi(x), which depends on the state at X[N,M] and all its neighbors
        mask = self.system.flatten() == state
        eu = (self.L/self.lam1)*np.dot(s,mask)
        return eu/self.T


    # @profile
    def full_energy(self):
        eu = np.log(1+self.psi/(self.lam2))
        # divide by a factor of two to negate the effects of double counting,
        # i.e. X_TAX will be twice the energy because Aij == Aji
        small_lambda = (self.lam2*self.A/(self.T*2))/self.psi
        big_lambda = small_lambda.sum()
        p = small_lambda/big_lambda
        B = np.random.poisson(big_lambda)
        out = np.random.multinomial(B, p.flatten(),1)

        e = 0
        for i in np.nonzero(out[0,:])[0]:
            x = i // self.size**2
            y = i % self.size**2
            if self.system[x // self.size, x % self.size] == self.system[y//self.size,y%self.size]:
                e += out[0,i]
        return e*eu
        # e = 0
        # for i in range(1,self.D+1):
        #     X = self.system.flatten() == i
        #     e +=np.dot(X.T,np.dot(self.A,X))
        # res = e/self.T
        # return res

    # @profile
    def calc_marginals(self):
        # marginals is a DxNxN tensor
        # i goes from 0 to D-1
        # counts the number of times state d occurs 
        for i in range(self.D):
            self.marginals[i]+=self.system==self.states[i]

    def compute_error(self,epoch):
 
        y_bar = self.marginals/((epoch-(self._EPOCHS-self.points)+1))


        y_bar -= self.true
        y_bar = y_bar.reshape(self.D,-1)

        return np.mean((np.linalg.norm(y_bar,axis=0)))

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run(self, video=True):
        # calculate initial energy 
        full_energy_x = self.full_energy()
        print("batch size: ",str(self.lam2))

        for epoch in tqdm.trange(self._EPOCHS,desc='Gibbs step'):
            # Randomly select a site on the lattice

            N, M = np.random.randint(0, self.size, 2)
            # read current state
            curr = self.system[N, M]

            # calculate energy of each state
            A = self.A[:, self.size*N+M]
            energies = np.zeros((1, len(self.states)))
            s = np.random.poisson(self.lam1*A/self.L)


            for state in self.states:
                energies[0, state-1] = self._energy(N, M, state, s)


            # calculate the transition probabilities from the exponentials
            transition = self.softmax(energies)

            # choose a state based on energy probabilities
            # output is (1 to 10)
            ez = np.random.choice(range(1, self.D+1),
                                  1, p=np.squeeze(transition))[0]

            # most probable energy
            (energy_y) = (energies[0, ez-1])

            # current energy
            (energy_x) = energies[0, curr-1]

            # full energy
            # change the state at N,M to the proposed state and calculate full energy
            self.system[N,M] = ez
            full_energy_y = self.full_energy()

            #change the state back 
            self.system[N,M] = curr

            a = np.exp(full_energy_y-energy_y-full_energy_x+energy_x)

            p = min(a, 1)

            # print('p',p)
            # if likely, accept
            if np.random.random() <= p:
                # accept the change
                self.system[N, M] = ez
                # replace the old estimate of the energy with the new one
                full_energy_x = full_energy_y
            else:
                # reject
                self.system[N, M] = curr
            
            self.calc_marginals()
            error = self.compute_error(epoch)
            self.errors.append(error)
   

        print("...done")
        print('final error: ',self.errors[-1])
        self.q.put([[self.lam2]+self.errors])
        print(self.system)
        print(self.psi**2)
        print(self.lam2)
        return True


if __name__ == "__main__":

    # variable to store the errors from marginals
    global args
    args = parser.parse_args()

    errors = []
    tasks = []
    q = multiprocessing.Manager().Queue()

    print('no output ',args.no_output)
    factors = [1,2,3,4]
    # temperatures = [0.2,0.6,0.8,1,2]
    for i in (factors):
        p = PottsLattice(q, temp=0.15, lam1 = 100, lam2 = i, states = args.states, epoch =args.epoch, initial_state="r", size=(args.size, args.size))
        p.start()
        print('start')
        tasks.append(p)
    
    print('one')

    try:
        for task in tasks:
            print('two')
            task.join()
    # handle keyboard interrupts to avoid unclosed processes 
        print('three')
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
            errors.append(item)
        except :
            break
    

    if args.no_output == False:
        with open('DoubleMin_errors_{}_{}_{}.csv'.format(args.states,args.size,args.epoch),'wb') as f:
            for i in errors:
                np.savetxt(f, i,delimiter=',')
