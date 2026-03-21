import numpy as np

class EcoSystem:
    def __init__(self, n_firms=5, rho=0.9, alpha=0.5, price=15.0): #tout à ajuster 
        self.n_firms = n_firms
        self.rho = rho      
        self.alpha = alpha  
        self.p = price      
        self.a_i = np.linspace(1, 5, n_firms) 
        self.b = 0.5
        self.lambda_env = 1 #tester différentes valeurs 
        self.reset()

    def reset(self):
        self.pollution = 0.0
        self.total_q = 0.0
        self.prev_policy_val = 0.0
        return self._get_state()

    def _get_state(self):
        # L'état est principalement défini par le stock de pollution
        return np.array([self.prev_policy_val, self.total_q, self.pollution])

    def step_multi_firms(self, action_type, action_value, q_list):
        individual_profits = []
        current_q_total = 0
        total_tax = 0 #correction welfare 
        
        for i in range(self.n_firms):
            q_chosen = q_list[i]
            
            if action_type == 0: # TAXE
                tax_cost = action_value * self.alpha * q_chosen
                q_eff = q_chosen
            else: # QUOTA
                tax_cost = 0
                q_eff = min(q_chosen, action_value)

            total_tax += tax_cost #correction welfare
            
            cost = self.a_i[i] * q_eff + self.b * (q_eff**2)
            profit = (self.p * q_eff) - cost - tax_cost
            
            individual_profits.append(profit)
            current_q_total += q_eff

        # Mise à jour de la pollution
        self.pollution = self.rho * self.pollution + self.alpha * current_q_total
        self.total_q = current_q_total
        
        # Calcul du Bien-être Social (Social Welfare)

        #comparer les deux, linéaire et quadratique 
        env_damage = self.lambda_env * self.pollution 
        #env_damage = self.lambda_env * (self.pollution ** 2)
        social_welfare = sum(individual_profits) + total_tax - env_damage
        #social_welfare = sum(individual_profits) - env_damage
        
        return self._get_state(), social_welfare, individual_profits