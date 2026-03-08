import numpy as np
import random

class GovernmentAgent:
    def __init__(self, lr=0.1, discount=0.95, epsilon=1.0):
        # On définit une liste d'actions mixtes : (Type, Valeur)
        # Type 0 = Taxe, Type 1 = Quota
        self.actions = [ (0, v) for v in [0, 1, 2, 3, 4, 5] ] + \
                       [ (1, v) for v in [10, 8, 6, 4, 2] ]
        
        self.lr = lr
        self.discount = discount # Gamma
        self.epsilon = epsilon
        self.eps_decay = 0.9995
        self.q_table = {} 

    def discretize_state(self, pollution):
        """
        On simplifie l'état à la pollution seule pour stabiliser l'apprentissage au début.
        Si tu passes un vecteur [taxe, prod, pollution], on ne prend que la pollution.
        """
        if isinstance(pollution, np.ndarray):
            val = pollution[-1] # On prend le dernier élément (pollution)
        else:
            val = pollution
            
        return int(np.clip(val / 5, 0, 19)) # 20 paliers

    def get_action(self, state):
        s = self.discretize_state(state)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(self.actions))

        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            return np.argmax(self.q_table[s])

    def learn(self, state, action_idx, reward, next_state):
        s = self.discretize_state(state)
        ns = self.discretize_state(next_state)
        
        if ns not in self.q_table:
            self.q_table[ns] = np.zeros(len(self.actions))
            
        old_value = self.q_table[s][action_idx]
        next_max = np.max(self.q_table[ns])
        
        # Mise à jour Q-Value
        self.q_table[s][action_idx] = old_value + self.lr * (reward + self.discount * next_max - old_value)
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)

class FirmAgent:
    def __init__(self, lr=0.1, discount=0.9, epsilon=1.0):
        # Actions : Quantités possibles à produire (ex: de 0 à 20)
        self.actions = np.linspace(0, 20, 21) 
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.eps_decay = 0.999
        self.q_table = {}

    def get_state_key(self, action_type, action_value):
        # On crée un état combinant le type de politique et sa valeur
        # Exemple : (0, 2.0) pour une taxe de 2.0
        return (action_type, round(action_value, 1))

    def get_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return np.argmax(self.q_table[state_key])

    def learn(self, s_key, a_idx, r, ns_key):
        if ns_key not in self.q_table:
            self.q_table[ns_key] = np.zeros(len(self.actions))
        
        old_val = self.q_table[s_key][a_idx]
        next_max = np.max(self.q_table[ns_key])
        
        # La firme apprend à maximiser son propre profit intertemporel
        self.q_table[s_key][a_idx] = old_val + self.lr * (r + self.discount * next_max - old_val)
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)