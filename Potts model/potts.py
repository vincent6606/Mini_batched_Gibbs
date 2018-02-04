

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
                    help='factor or lamba')

parser.add_argument('--states', default=10, type=int,
                    help='factor or lamba')

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

    

    def __init__(self,q, inv_tempature, lam,states, epoch=1e6,initial_state='r', size=(100, 100), show=False):
        super(PottsLattice, self).__init__()

        self.sqr_size = size
        self.size = size[0]
        self.T = inv_tempature
        self._build_system(initial_state)

        # marginals
        self.create_A()
        # 10 states
        self.D = states
        self.states = range(1, self.D+1)

        # L is the maximum sum of factors
        self.L = self.A.sum(axis=0).max()
        # factor of lambda
        
        self.lam = lam*(self.L**2)



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
        eu = (self.L/self.lam)*np.dot(s,mask)
        # print(eu)
        return eu



    # @profile
    def full_energy(self, N, M, state):
        mask = self.system.flatten() == state
        A = self.A[:, self.size*N+M]
        A = mask*A
        energy = np.dot(mask, A)
        return energy

    # @property
    def internal_energy(self):
        return -4*self.system.flatten().dot((self.A).dot(self.system.flatten()))

    # @property
    def magnetization(self):
        """Find the overall magnetization of the system
        """
        return np.abs(np.sum(self.system)/self.size**2)
    # @profile
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


    def run(self, video=True):


        print("batch size: ",str(self.lam))
        """Run the simulation
        """

        # FFMpegWriter = manimation.writers['ffmpeg']
        # writer = FFMpegWriter(fps=10)

        # plt.ion()
        # fig = plt.figure()

        # with writer.saving(fig, "Potts_mini.mp4", 100):
        for epoch in tqdm.trange(self._EPOCHS,desc='Gibbs step'):
            # Randomly select a site on the lattice

            N, M = np.random.randint(0, self.size, 2)
            # read current state
            curr = self.system[N, M]

            # calculate energy of each state
            A = self.A[:, self.size*N+M]
            energies = np.zeros((1, len(self.states)))
            s = np.random.poisson(self.lam*A/self.L,self.size**2)


            for state in self.states:
                energies[0, state-1] = self._energy(N, M, state, s)

            # calculate the transition probabilities from the exponentials
            energies -= np.max(energies)
            energies = np.exp(energies)
            sum_ez = np.sum(energies)
            transition = energies/sum_ez

            # choose a state based on energy probabilities
            # out put is (1 to 10)
            ez = np.random.choice(range(1, self.D+1),
                                  1, p=np.squeeze(transition))
            ez = ez[0]
#                 print(np.squeeze(transition))
#                 print('ez',ez)
#                 print('trans',transition)

            # most probable ernergy
            (energy_y) = np.exp(energies[0, ez-1])
#                 print(energy_y)

            # current energy
            (energy_x) = energies[0, curr-1]
#                 print(energy_x)

            # full energies
            (full_energy_x) = self.full_energy(N, M, curr)
            (full_energy_y) = self.full_energy(N, M, ez)

            # if epoch % (self._EPOCHS//75) == 0:
                # print(full_energy_y, energy_y, full_energy_x, energy_x)
            a = np.exp(full_energy_y-energy_y-full_energy_x+energy_x)

            p = min(a, 1)
#                 print('p',p)

            # if likely, accept
            if np.random.random() <= p:
                # accept the change
                self.system[N, M] = ez
            else:
                # reject
                self.system[N, M] = curr
            # hack to record errors for 200 iterations
            
            self.calc_marginals()
            error = self.compute_error(epoch)
            self.errors.append(error)
   

        print("...done")

        # plt.close('all')
        
        print('final error: ',self.errors[-1])
        self.q.put([[self.lam]+self.errors])
        # print(self.q)





        # with open('errors1.csv','wb') as f:
        #     np.savetxt(f, [[self.lam]+self.errors],delimiter=',')
        return True


if __name__ == "__main__":

    # variable to store the errors from marginals
    global args
    args = parser.parse_args()

    errors = []
    tasks = []
    q = multiprocessing.Manager().Queue()


    factors = [2,10,30,50,80]
    for i in (factors):
        p = PottsLattice(q, inv_tempature=11,lam = (i), states = args.states,epoch =args.epoch, initial_state="r", size=(args.size, args.size))
        p.start()
        print('start')
        tasks.append(p)
    
    print('one')

    try:
        for task in tasks:
            print('two')
            task.join()
    # handle keyboard interrupts to avoid un-closed processes 
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
    print(errors[-1])

    with open('MGPMH_errors_{}_{}_{}.csv'.format(args.states,args.size,args.epoch),'wb') as f:
        for i in errors:
            np.savetxt(f, i,delimiter=',')
