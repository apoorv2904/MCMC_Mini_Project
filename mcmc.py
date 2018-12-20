
# coding: utf-8

# In[1]:


from utils import *

class MCMC:
    
    def __init__(self, n, m, sampler):
        self.n = n
        self.m = m
        self.sampler = sampler
        self.beta = 1.0

        self.W = np.random.randn(self.m, self.n)
        self.X = self.random_vector()
        self.Y = self.get_y(self.X)

    def set_observations(self, Y, W ):
        
        self.n = W.shape[1]
        self.m = W.shape[0]
        self.Y = Y
        self.W = W
        self.X = self.random_vector()
        
    def draw_sample(self, x):
        return self.sampler(self, x)

    def metropolis(self, x):
        # For x_next, flip sign of one random element of x.
        index = np.random.randint(self.n)
        x_next = np.copy(x)
        x_next[index] *= -1
        h_next = self.energy(x_next)
        h_cur = self.energy(x)
        
        acceptance_prob = min(1.0, np.exp(-self.beta * (h_next - h_cur)))
        if np.random.uniform() <= acceptance_prob:
            return x_next, h_next, acceptance_prob, 1
        else:
            return x, h_cur, acceptance_prob, 0

    def glauber(self, x):
        index = np.random.randint(self.n)
        x_next, x_flip = np.copy(x), np.copy(x)
        x_flip[index] *= -1
        h_flip = self.energy(x_flip)
        h_cur = self.energy(x)
        
        prob_plus = (1 + x[index] * np.tanh(self.beta * (h_flip - h_cur)))/2
        if np.random.uniform() <= prob_plus:
            x_next[index] = 1
        else:
            x_next[index] = -1

        # The rest is just to also make h_next, acceptance_prob, and s available
        if x_next[index] == x[index]:
            h_next = h_cur
            s = 0
        else:
            h_next = h_flip
            s = 1

        # Still consider acceptance_prob to be the probability of flipping
        # x[index], so either p+ or p- depending on the current value.
        if x[index] == -1:
            acceptance_prob = prob_plus
        else:
            acceptance_prob = 1 - prob_plus
        return x_next, h_next, acceptance_prob, s

    def random_vector(self):
        return np.random.choice([-1, 1], self.n)

    def ReLU(self, x):
        return np.maximum(x, 0)

    def get_y(self, x):
        return self.ReLU(np.dot(self.W, x) / np.sqrt(self.n))

    def energy(self, x):
        residual = self.Y - self.get_y(x)
        return np.dot(residual, residual)

    def set_beta(self, beta):
        self.beta = beta

    def error(self, x):
        # The error is defined as: np.linalg.norm(x - self.X)**2 / (4 * self.n)
        # This can be simplified to: np.sum((x - self.X)**2) / (4 * self.n)
        # Which is just the fraction of elements that differ in x and self.X:
        return np.count_nonzero(x != self.X) / self.n

    def get_initial_state(self):
        x0 = self.random_vector()
        h0 = self.energy(x0)
        a0 = 1
        s0 = 1
        return x0, h0, a0, s0
    
# Automatic Parameter estimation for annealing
def auto( n, alpha, sampler=MCMC.metropolis, minutes=2, steps=2000, acceptance_rate=0.98, min_acceptance_rate=0.0):
    
    """Explores the annealing energy surface and
    estimates optimal temperature settings."""
    def run(state, energy, T, n_steps):
        
        mcmc.set_beta(1.0/T)
        prev_state = copy.deepcopy( state )
        prev_energy = copy.deepcopy( energy )
        n_accepts = 0.0
        n_improves = 0.0
        
        for i in range( n_steps ):
            state, energy, p_accept, b_accept = mcmc.draw_sample( state )
            delta_energy = energy - prev_energy
            
            n_accepts = n_accepts + b_accept
            if delta_energy < 0.0:
                n_improves = n_improves + 1
            
            prev_state = copy.deepcopy( state )
            prev_energy = copy.deepcopy( energy )
        return state, energy, float(n_accepts) / n_steps, float(n_improves) / n_steps
    
    
    m = int(alpha * n)
    mcmc = MCMC(n, m, sampler)
    state, energy, p_accept, b_accept = mcmc.get_initial_state()
        
    start_energy = copy.deepcopy(energy)
    start_state = copy.deepcopy(state)
    
    step = 0
    start = time.time()

    # Attempting automatic simulated anneal...
    # Find an initial guess for temperature
    T = 0.0
    
    while T == 0.0:
        step += 1
        state, energy, p_accept, b_accept = mcmc.draw_sample( state )
        T = abs( energy - start_energy)

    # Search for Tmax - a temperature that gives 98% acceptance
    state, energy, acceptance, improvement = run(state, energy, T, steps)

    step += steps
    while acceptance > acceptance_rate:
        T = np.round(T / 1.5, 2)
        state, energy, acceptance, improvement = run(state, energy, T, steps)
        step += steps
    while acceptance < acceptance_rate:
        T = np.round(T * 1.5, 2)
        state, energy, acceptance, improvement = run(state, energy, T, steps)
        step += steps
    Tmax = T

    # Search for Tmin - a temperature that gives 0% improvement
    while improvement > min_acceptance_rate:
        T = np.round(T / 1.5, 2)
        state, energy, acceptance, improvement = run(state, energy, T, steps)
        step += steps
    Tmin = T

   
    # Don't perform anneal, just return params
    return {'tmax': Tmax, 'tmin': Tmin, 'nexplore': step }

