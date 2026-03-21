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
        self.eps_decay = 0.9999 # Réduction de l'exploration
        self.q_table = {} 

    def discretize_state(self, pollution):
        """
        On simplifie l'état à la pollution seule pour stabiliser l'apprentissage.
        """
        if isinstance(pollution, np.ndarray):
            val = pollution[-1] 
        else:
            val = pollution
            
        return int(np.clip(val / 5, 0, 19)) # 20 paliers de pollution

    def get_action(self, state):
        s = self.discretize_state(state)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(self.actions))
        
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        return np.argmax(self.q_table[s])

    def learn(self, s, a_idx, r, ns):
        s_dis = self.discretize_state(s)
        ns_dis = self.discretize_state(ns)
        
        if ns_dis not in self.q_table:
            self.q_table[ns_dis] = np.zeros(len(self.actions))
            
        # Q-Learning Target
        best_next_action = np.max(self.q_table[ns_dis])
        td_target = r + self.discount * best_next_action
        
        # Mise à jour de la Q-Table
        self.q_table[s_dis][a_idx] += self.lr * (td_target - self.q_table[s_dis][a_idx])

    def decay_epsilon(self):
        """Réduit l'exploration du gouvernement (à appeler dans le main)"""
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)


class FirmAgent:
    def __init__(self, lr=0.1, epsilon=1.0):
        # Actions : Quantités possibles à produire (de 0 à 15 pour plus de précision)
        self.actions = np.linspace(0, 15, 16) 
        self.lr = lr
        self.epsilon = epsilon
        self.eps_decay = 0.9999
        self.q_table = {}

    def get_action(self, action_type, action_value):
        """La firme observe la politique (état) avant de décider"""
        state_key = (int(action_type), round(float(action_value), 1))
        
        if state_key not in self.q_table:
            # INITIALISATION OPTIMISTE : on commence à 20 pour forcer l'exploration
            self.q_table[state_key] = np.ones(len(self.actions)) * 20.0
        
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        return np.argmax(self.q_table[state_key])

    def learn(self, action_type, action_value, a_idx, r):
        """Apprentissage basé sur le profit immédiat (discount=0)"""
        state_key = (int(action_type), round(float(action_value), 1))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(len(self.actions)) * 20.0
            
        # La firme apprend si l'action choisie a rapporté le profit r attendu
        self.q_table[state_key][a_idx] += self.lr * (r - self.q_table[state_key][a_idx])

    def decay_epsilon(self):
        """Réduit l'exploration de la firme (à appeler dans le main)"""
        self.epsilon = max(0.01, self.epsilon * self.eps_decay)