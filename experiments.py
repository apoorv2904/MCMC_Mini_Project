
# coding: utf-8

# In[1]:


from utils import *
from mcmc import *

class Experiment:
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        self.n = n
        self.alpha = alpha
        self.m = int(alpha * n)
        self.N = N
        self.sampler = sampler

        self.show_plot = show_plot
        self.print_statistics = print_statistics

        self.mcmc = MCMC(self.n, self.m, sampler=self.sampler)
        self.x_v = np.zeros((1, n))  # state vector
        self.h_v = np.zeros(1)       # energy vector
        self.a_v = np.zeros(1)       # acceptance probability vector
        self.s_v = np.zeros(1)       # acceptance result vector
        self.e_v = np.zeros(1)       # error vector
        self.b_v = b_v               # beta vector

    def set_observations( self, Y, W ):
        self.mcmc.set_observations( Y, W )

    def run(self):
        raise NotImplementedError("To implement in subsclasses")
        
    def final_error(self):
        return self.e_v[-1]
    
    def final_energy(self):
        return self.h_v[-1]
    
    def min_energy(self):
        return self.h_v.min()
    
    def error_at_min_energy(self):
        return self.e_v[self.h_v.argmin()]
    
    def min_error(self):
        return self.e_v.min()
    
    def energy_at_min_error(self):
        return self.h_v[self.e_v.argmin()]
    
    def print_error_energy_statistics(self):
        
        '''print('%35s %.4f ' %('Energy Last Sample:', self.final_energy()))
        print('%35s %.4f ' %('Minimum Energy Sample:', self.min_energy()))
        print('%35s %.4f ' %('Energy for Minimum Error Sample:',
                           self.error_at_min_energy()))
        
        print('%35s %.4f ' %('Error Last Sample:', self.final_error()))
        print('%35s %.4f ' %('Error for Minimum Energy Sample:',
                           self.error_at_min_energy()))
        print('%35s %.4f ' %( 'Minimum Error Sample:', self.min_error()))
        print('')'''
        
        print('%35s %.4f, %35s %.4f ' %('Energy Last Sample:', self.final_energy(), 
                                        'Error Last Sample:', self.final_error()))
        print('%35s %.4f, %35s %.4f ' %('Minimum Energy Sample:', self.min_energy(), 
                                        'Error for Minimum Energy Sample:',
                                        self.error_at_min_energy()))
        print('%35s %.4f, %35s %.4f ' %('Energy for Minimum Error Sample:',
                                        self.energy_at_min_error(), 'Minimum Error Sample:',
                                        self.min_error()))
        print('')
        
        

    def plot_energy_error(self, title=None):
        if title is None:
            title = "n = " + self.n + ", alpha = " + self.alpha
        plot_energy_error([self.h_v], [self.e_v], title)
    
    def plot_acceptance_trend(self, title=None):
        if title is None:
            title = "n = " + self.n + ", alpha = " + self.alpha
        plot_acceptance_trend(self.a_v, self.s_v, title)

    def plot_beta_schedule(self):
        plot_beta_schedule(self.b_v)
    
    def plot_all_results(self, title=None):
        if title is None:
            title = "n = " + str(self.n) + ", alpha = " + str(self.alpha)
        self.plot_energy_error(title)
        self.plot_acceptance_trend("")
        self.plot_beta_schedule()

        
class StandardExperiment(Experiment):
    """The StandardExperiment stores all interesting data for all timesteps,
    this could take a lot of memory."""
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        super().__init__(n, alpha, N, b_v, sampler, show_plot, print_statistics)

        self.x_v = np.zeros((N + 1, n))  # state vector
        self.h_v = np.zeros(N + 1)       # energy vector
        self.a_v = np.zeros(N + 1)       # acceptance probability vector
        self.s_v = np.zeros(N + 1)       # acceptance result vector
        self.e_v = np.zeros(N + 1)       # error vector

        self.h_min = 10000.0                   # minimum energy explored
        self.x_h_min = np.zeros((1, self.n ))         # output corresponding to  minimum energy
    def run(self):
        self.x_v[0], self.h_v[0], self.a_v[0], self.s_v[0] = self.mcmc.get_initial_state()
        self.e_v[0] = self.mcmc.error(self.x_v[0])
        self.mcmc.set_beta(self.b_v[0])
        self.h_min = self.h_v[0]
        self.x_h_min = self.x_v[0]

        for i in range(1, self.N + 1):
            if( self.h_min <= 1e-10 ):
                self.h_v = self.h_v[0:i]
                self.a_v = self.a_v[0:i]
                self.s_v = self.s_v[0:i]
                self.e_v = self.e_v[0:i]
                self.x_v = self.x_v[0:i]
                
                print( 'Reached this condition' )
                break
                
            self.x_v[i], self.h_v[i], self.a_v[i], self.s_v[i] = self.mcmc.draw_sample(
                self.x_v[i-1])
            self.e_v[i] = self.mcmc.error(self.x_v[i])
            if self.h_v[i] < self.h_min:
                self.h_min = self.h_v[i]
                self.x_h_min = self.h_v[i]
                
            if i < self.N and self.b_v[i] != self.b_v[i-1]:
                self.mcmc.set_beta(self.b_v[i])
                #print( 'Setting beta to %.2f, %.2f' %(self.b_v[i], self.mcmc.beta))

        if self.show_plot:
            self.plot_all_results()

        if self.print_statistics:
            self.print_error_energy_statistics()

        return self
    
class LeanExperiment(Experiment):
    """The LeanExperiment stores all interesting data for all timesteps except the chain.
    this should save a lot of memory."""
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        super().__init__(n, alpha, N, b_v, sampler, show_plot, print_statistics)

        self.h_v = np.zeros(N + 1)       # energy vector
        self.a_v = np.zeros(N + 1)       # acceptance probability vector
        self.s_v = np.zeros(N + 1)       # acceptance result vector
        self.e_v = np.zeros(N + 1)       # error vector

        self.h_min = 10000.0                   # minimum energy explored
        self.x_h_min = np.zeros((1, self.n ))         # output corresponding to  minimum energy
        
    def run(self):
        self.x_v[0], self.h_v[0], self.a_v[0], self.s_v[0] = self.mcmc.get_initial_state()
        self.e_v[0] = self.mcmc.error(self.x_v[0])
        self.mcmc.set_beta(self.b_v[0])
        self.h_min = self.h_v[0]
        self.x_h_min = self.x_v[0]

        for i in range(1, self.N + 1):
            if( self.h_min <= 1e-10 ):
                self.h_v = self.h_v[0:i]
                self.a_v = self.a_v[0:i]
                self.s_v = self.s_v[0:i]
                self.e_v = self.e_v[0:i]
                print( 'Reached this condition' )

                break
            self.x_v[0], self.h_v[i], self.a_v[i], self.s_v[i] = self.mcmc.draw_sample(
                self.x_v[0])
            self.e_v[i] = self.mcmc.error(self.x_v[0])
            
            if self.h_v[i] < self.h_min:
                self.h_min = self.h_v[i]
                self.x_h_min = self.x_v[0]
            
            if i < self.N and self.b_v[i] != self.b_v[i-1]:
                self.mcmc.set_beta(self.b_v[i])
                #print( 'Setting beta to %.2f, %.2f' %(self.b_v[i], self.mcmc.beta))

        if self.show_plot:
            self.plot_all_results()

        if self.print_statistics:
            self.print_error_energy_statistics()

        return self
        
class MultiExperiments:
    def __init__(self, n_vector, alpha_vector, N, b_v, n_exp, sampler=MCMC.metropolis,
               show_plot=False, print_statistics=False, seed=123, lean=True,
                ):
        """Creates and runs MCMC experiments. <n> and <alpha> can be lists,
           experiments for all combinations of their parameters will be run.
           N and b_v should (currently) stay constant. The seed is set only once
           before running all experiments, so it's also possible to run multiple
           times with the same parameters."""
        if seed != -1:
            self.set_seed(seed)
        
        self.parameters = np.array(
            np.meshgrid(n_vector, alpha_vector)).T.reshape(-1, 2)
        print("Running %d experiments." % len(self.parameters))
        
        #To ensure same data when comparing against different N
        self.seeds = np.random.randint(1000000, size=len(self.parameters)*n_exp)
        self.N = N
        self.b_v = b_v

        self.experiments = defaultdict(list)
        self.n_exp = n_exp
        
        for indx, [n, alpha] in enumerate( self.parameters ):
            n = int(n)
            if lean:
                Exp = LeanExperiment
            else:
                Exp = StandardExperiment
            
            
            for i in range(n_exp):
                #To ensure same data when comparing against different N
                self.set_seed( self.seeds[(indx* n_exp) + i] )
                self.experiments[indx].append( Exp(n, alpha, N, b_v, sampler, show_plot, print_statistics).run())
            

    def set_seed(self, seed):
        np.random.seed(seed)
    
    def __getitem__(self, key):
        n, alpha = self.parameters[key]
        return self.experiments[key]
  
    def __len__(self):
        return len(self.parameters)
  
    class _exp_iter:
        def __init__(self, experiments, parameters):
            self.experiments = experiments
            self.parameters = parameters
            self.cur = 0

        def __next__(self):
            i = self.cur
            if i >= len(self.parameters):
                raise StopIteration
            self.cur += 1
            n, alpha = self.parameters[i]
            return self.experiments[i]

    def __iter__(self):
        return MultiExperiments._exp_iter(self.experiments, self.parameters)
      
    def get_mean(self, function):
        v = np.ones(len(self.parameters))
        for i, experiments in enumerate(self):
            v[i] = np.mean(list(map(function, experiments)))
        return v
      
    def get_std(self, function):
        v = np.ones(len(self.parameters))
        for i, experiments in enumerate(self):
            v[i] = np.std(list(map(function, experiments)))
        return v
      
    def get_stats(self, name):
        f_dict = {"final_errors": Experiment.final_error,
                  "min_energy_errors": Experiment.error_at_min_energy,
                  "min_errors": Experiment.min_error,
                  "final_energies": Experiment.final_energy,
                  "min_energies": Experiment.min_energy,
                  "min_error_energies": Experiment.energy_at_min_error}
        f = f_dict[name]
        return self.get_mean(f), self.get_std(f)
    
    def final_errors(self):
        e_v = np.ones((len(self.parameters), self.n_exp))
        for i, experiment in enumerate(self):
            for j, exp in enumerate( experiment ):
                e_v[i,j] = exp.final_error()
        if len(self.parameters) == 1:
            e_v = np.reshape(e_v,-1)
        return e_v
    
    def min_energy_errors(self):
        e_v = np.ones((len(self.parameters), self.n_exp))
        for i, experiment in enumerate(self):
            for j, exp in enumerate( experiment ):
                e_v[i,j] = exp.error_at_min_energy()
        if len(self.parameters) == 1:
            e_v = np.reshape(e_v,-1)
        return e_v
    
    def min_errors(self):    
        e_v = np.ones((len(self.parameters), self.n_exp))
        for i, experiment in enumerate(self):
            for j, exp in enumerate( experiment ):
                e_v[i,j] = exp.min_error()
        if len(self.parameters) == 1:
            e_v = np.reshape(e_v,-1)
        
        return e_v
    
    def print_mean_statistics( self ):
        
        mean_final_errors, std_final_errors = self.get_stats('final_errors')
        mean_min_energy_errors, std_min_energy_errors = self.get_stats('min_energy_errors')
        mean_min_errors, std_min_errors = self.get_stats('min_errors')
        mean_final_energies, std_final_energies = self.get_stats('final_energies')
        mean_min_energies, std_min_energies = self.get_stats('min_energies')
        mean_min_error_energies, std_min_error_energies = self.get_stats('min_error_energies')
        
        
        
        for i in range( len(self.parameters)):
            
            n_cur, alpha_cur = self.parameters[i]
            print('\n n=%d, alpha=%.2f' %(n_cur, alpha_cur))

            print( '%35s: %.3f, %35s: %.3f' %('Mean Final Energy', mean_final_energies[i], 
                                              'Mean Final Error', mean_final_errors[i]))

            print( '%35s: %.3f, %35s: %.3f' %('Mean Minimum Energy', mean_min_energies[i], 
                                              'Mean Error at Minimum Energy', mean_min_energy_errors[i]))

            print( '%35s: %.3f, %35s: %.3f' %('Mean Energy at Minimum Error', mean_min_error_energies[i], 
                                              'Mean Minimum Error', mean_min_errors[i] ))

            print( '%35s: %.3f, %35s: %.3f' %('Std Final Energy', std_final_energies[i], 
                                              'Std Final Error', std_final_errors[i]))

            print( '%35s: %.3f, %35s: %.3f' %('Std Minimum Energy', std_min_energies[i], 
                                              'Std Error at Minimum Energy', std_min_energy_errors[i]))

            print( '%35s: %.3f, %35s: %.3f' %('Std Energy at Minimum Error', std_min_error_energies[i], 
                                              'Std Minimum Error', std_min_errors[i]))
    

