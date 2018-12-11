
# coding: utf-8

# In[ ]:


from utils import *

class Schedule:

    def __init__(self, beta_0 = 1.0):
        self.beta_0 = beta_0

    def get_schedule(self, N):
        """Should return a numpy vector of length N that specifies the beta
        value for each time step."""
        raise NotImplementedError("To implement in subclasses.")

class ConstantSchedule(Schedule):
    """Constant schedule. Beta never changes."""

    def __init__(self, beta_0 ):
        self.beta_0 = beta_0
        self.name = 'Constant Schedule %.3f' %(beta_0)

    def get_schedule(self, N):
        return np.ones(N) * self.beta_0
    


class ExponentialMultiplicativeSchedule(Schedule):
    """Exponential schedule. T_k = T_start*(alpha^k). Usually ( 0.8 < alpha < 0.99 )"""
    def __init__(self, T_start, T_freeze, alpha ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.alpha = alpha
        self.name = 'Exponential Multiplicative Schedule T_S:%.2f T_F:%2.f Alpha:%.2f' %(T_start, T_freeze, alpha)

        #super(BinaryAnnealer, self).__init__(1/self.T0)
    
    def get_nsteps(self, dwell):
        # T_freeze = T_start*(alpha^k)
        # k  = log(T_freeze/T_start) / log(alpha)
        return np.ceil(dwell * ( np.log(self.T_freeze/self.T_start) / np.log(self.alpha) ))
    
    def get_dwell_period(self, N):
        # T_freeze = T_start*(alpha^k)
        # k  = log(T_freeze/T_start) / log(alpha)
        dwell = int(N/( np.log(self.T_freeze/self.T_start) / np.log(self.alpha) )) 
        return dwell
    
    def get_schedule(self, N ):
        self.N = N
        self.dwell = self.get_dwell_period(N)
        n_steps = np.ceil(N/self.dwell)
        T_schedule = self.T_start * np.power(self.alpha,np.arange(n_steps))
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:int(N)]
        return 1.0/T_schedule



class LogarithmicMultiplicativeSchedule(Schedule):
    """Logarithmic Multiplicative schedule. T_k = T_start/ ( 1 + alpha*log(1+k)). Usually ( alpha > 1.0 )"""
    def __init__(self, T_start, T_freeze, alpha ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.alpha = alpha
        self.name = 'Logarithmic Multiplicative Schedule T_S:%.2f T_F:%2.f Alpha:%.2f' %(T_start, T_freeze, alpha)

        #super(BinaryAnnealer, self).__init__(1/self.T0)
    
    def get_nsteps(self, dwell):
        # T_freeze = T_start/( 1 + alpha*log(1+k))
        # k  = exp( (T_start/T_freeze - 1)/alpha ) - 1 
        return np.ceil(dwell * ( np.exp( ((self.T_start/self.T_freeze) - 1 )/ self.alpha ) -1))
    
    def get_dwell_period(self, n_steps):
        dwell = int( n_steps / self.get_nsteps(1) ) 
        return dwell
    
    
    def get_schedule(self, N ):
        self.N = N
        self.dwell = self.get_dwell_period(N)
        n_steps = np.ceil(N/self.dwell)
        
        T_schedule = self.T_start / (1.0 + self.alpha*np.log( np.arange( n_steps ) + 1.0 ) )
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:int(N)]
        return 1.0/T_schedule

class LinearMultiplicativeSchedule(Schedule):
    """Linear Multiplicative schedule. T_k = T_start/ ( 1.0 + alpha*k). Usually ( alpha > 0.0 )"""
    def __init__(self, T_start, T_freeze, alpha ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.alpha = alpha
        #super(BinaryAnnealer, self).__init__(1/self.T0)
        self.name = 'Linear Multiplicative Schedule T_S:%.2f T_F:%2.f Alpha:%.2f' %(T_start, T_freeze, alpha)

    def get_nsteps(self, dwell):
        # T_freeze = T_start/(1 + alpha*k)
        # k  = ((T_start/T_freeze) - 1) / alpha
        return np.ceil(dwell * ( ((self.T_start/self.T_freeze) - 1.0) / self.alpha) )
    
    def get_dwell_period(self, n_steps):
        dwell = int( n_steps / self.get_nsteps(1) ) 
        return dwell
    
    def get_schedule(self, N ):
        self.N = N
        self.dwell = self.get_dwell_period(N)
        n_steps = np.ceil(N/self.dwell)
        T_schedule = self.T_start / (1.0 + self.alpha*np.arange(n_steps))
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:int(N)]
        return 1.0/T_schedule

class QuadraticMultiplicativeSchedule(Schedule):
    """Linear Multiplicative schedule. T_k = T_start/ ( 1.0 + alpha*(k^2)). Usually ( alpha > 0.0 )"""
    def __init__(self, T_start, T_freeze, alpha ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.alpha = alpha
        self.name = 'Quadratic Multiplicative Schedule T_S:%.2f T_F:%2.f Alpha:%.2f' %(T_start, T_freeze, alpha)

        #super(BinaryAnnealer, self).__init__(1/self.T0)
    
    def get_nsteps(self, dwell):
        # T_freeze = T_start/(1 + alpha*(k^2))
        # k  = sqrt( ((T_start/T_freeze) - 1) / alpha )
        return np.ceil(dwell * np.sqrt( ((self.T_start/self.T_freeze) - 1.0) / self.alpha) )
    
    def get_dwell_period(self, n_steps):
        dwell = int( n_steps / self.get_nsteps(1) ) 
        return dwell
    
    def get_schedule(self, N ):
        self.N = N
        self.dwell = self.get_dwell_period(N)
        n_steps = np.ceil(N/self.dwell)
        k = np.arange(n_steps)
        T_schedule = self.T_start / (1.0 + self.alpha*(k*k))
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:int(N)]
        return 1.0/T_schedule

    

class LinearAdditiveSchedule(Schedule):
    """Linear additive schedule. T_k = T_start - k*step_size. """
    def __init__(self, T_start, T_freeze, dwell ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.dwell = dwell
        self.name = 'Linear Additive Schedule T_S:%.2f T_F:%2.f Dwell:%.2f' %(T_start, T_freeze, dwell)

        #super(BinaryAnnealer, self).__init__(1/self.T0
    
    def get_schedule(self, N ):
        n_steps = np.ceil(N/self.dwell )
        T_schedule = np.linspace(self.T_start,self.T_freeze, n_steps)
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:N]
        return 1.0/T_schedule

    
class QuadraticAdditiveSchedule(Schedule):
    """Quadratic additive schedule. T_k = T_freeze + (T_start - T_freeze)*(1-k/N)^2."""
    def __init__(self, T_start, T_freeze, dwell ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.dwell = dwell
        #super(BinaryAnnealer, self).__init__(1/self.T0
        self.name = 'Quadratic Additive Schedule T_S:%.2f T_F:%2.f Dwell:%.2f' %(T_start, T_freeze, dwell)
        
    def get_schedule(self, N ):
        n_steps = np.ceil(N/self.dwell )
        steps = (1.0 - np.arange( n_steps)/n_steps )
        T_schedule = self.T_freeze + (( self.T_start - self.T_freeze ) * steps * steps )
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:N]
        return 1.0/T_schedule
    
    
class ExponentialAdditiveSchedule( Schedule ):
    """Exponential additive schedule. T_k = T_freeze + (T_start - T_freeze)* (1.0/(1 + exp( (2*ln(Ts-Tf)/n)*(k-n/2) )))."""
    def __init__(self, T_start, T_freeze, dwell ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.dwell = dwell
        self.name = 'Exponential Additive Schedule T_S:%.2f T_F:%2.f Dwell:%.2f' %(T_start, T_freeze, dwell)
        #super(BinaryAnnealer, self).__init__(1/self.T0
    
    def get_schedule(self, N ):
        n_steps = np.floor(N/self.dwell )
        steps = np.arange( n_steps) - (n_steps/2.0 )
        scaling_factor = 2*np.log( self.T_start - self.T_freeze )/n_steps 
        denominator = 1.0 + np.exp( scaling_factor * steps )
        
        T_schedule = self.T_freeze + (( self.T_start - self.T_freeze ) * (1.0/denominator) )
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:N]
        return 1.0/T_schedule
        
class TrignometricAdditiveSchedule( Schedule ):
    """Trignometric additive schedule. T_k = T_freeze + 0.5*(T_start - T_freeze)* (1 + cos( k*pi/n))."""
    def __init__(self, T_start, T_freeze, dwell ):
        self.T_start = T_start
        self.T_freeze = T_freeze
        self.dwell = dwell
        self.name = 'Trignometric Additive Schedule T_S:%.2f T_F:%2.f Dwell:%.2f' %(T_start, T_freeze, dwell)
        #super(BinaryAnnealer, self).__init__(1/self.T0
    
    def get_schedule(self, N ):
        n_steps = np.ceil(N/self.dwell )
        steps = np.arange( n_steps) * (np.pi / n_steps )
        
        
        T_schedule = self.T_freeze + (0.5*( self.T_start - self.T_freeze ) * (1.0+ np.cos(steps )))
        T_schedule = np.repeat( T_schedule, self.dwell )
        T_schedule = T_schedule[0:N]
        return 1.0/T_schedule
    
class CustomBetaSchedule(Schedule):
    """Boltzmann schedule. T_k = T_start/(1+log(1+k)). Need an exponential number of samples to converge"""

    def __init__(self, beta_v, steps_v ):
        self.beta_v = beta_v
        self.steps_v = steps_v

    def get_schedule(self):
        n_steps = np.sum(steps_v)
        beta_schedule = np.zeros(n_steps)
        
        cur_steps_cnt = 0
        for steps, beta in zip(steps_v, beta_v):
            beta_schedule[cur_steps_cnt: cur_steps_cnt+steps ] = beta
            cur_steps_cnt = cur_steps_cnt + steps
            
        return beta_schedule
    

