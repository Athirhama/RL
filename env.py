import numpy as np

class EcoSystem:
    def __init__(self, n_firms=5, rho=0.9, alpha=0.5, price=10.0):
        self.n_firms = n_firms
        self.rho = rho      
        self.alpha = alpha  
        self.p = price      
        self.a_i = np.linspace(1, 5, n_firms) 
        self.b = 0.5
        self.reset()

    def reset(self):
        self.pollution = 0.0
        self.total_q = 0.0
        self.prev_policy_val = 0.0
        return self._get_state()

    def _get_state(self):
        # L'état observé par l'agent : [Dernière Valeur, Production Totale, Pollution]
        return np.array([self.prev_policy_val, self.total_q, self.pollution])

    def step(self, action_type, action_value):
        """
        action_type : 0 pour Taxe, 1 pour Quota
        action_value : Valeur de la taxe ou plafond du quota
        """
        # 1. Réaction des firmes selon le type de politique
        if action_type == 0:  # CAS TAXE
            tax = action_value
            # q_i = (p - a_i - tax * alpha) / (2 * b)
            q_is = (self.p - self.a_i - (tax * self.alpha)) / (2 * self.b)
            q_is = np.maximum(0, q_is)
            tax_cost = tax * self.alpha * q_is
        else:  # CAS QUOTA
            # q_i sans taxe = (p - a_i) / (2 * b), mais limité par action_value
            q_is_optimal = (self.p - self.a_i) / (2 * self.b)
            q_is = np.maximum(0, np.minimum(q_is_optimal, action_value))
            tax_cost = 0

        self.total_q = np.sum(q_is)
        
        # 2. Calcul des profits individuels
        costs = self.a_i * q_is + self.b * (q_is**2)
        profits = (self.p * q_is) - costs - tax_cost
        
        # 3. Mise à jour de la pollution
        self.pollution = self.rho * self.pollution + self.alpha * self.total_q
        
        # 4. Calcul du Bien-être Social (Social Welfare)
        social_welfare = np.sum(profits) - self.pollution
        
        # 5. Mise à jour de l'état
        self.prev_policy_val = action_value
        state = self._get_state()
        
        # Note : on retourne 3 valeurs (state, reward, info) pour coller à ton Notebook
        # Si ton Notebook attend 'next_state, r, done', ajoute un booléen ici.
        return state, social_welfare, {"profits": profits, "pollution": self.pollution}
    def step_with_firm(self, action_type, action_value, q_chosen):
        """
        Version où la production 'q_chosen' est dictée par l'agent firme.
        """
        # 1. Calcul des coûts et profits avec la production choisie
        # On garde la logique : Taxe si type 0, Quota si type 1
        tax_cost = (action_value * self.alpha * q_chosen) if action_type == 0 else 0
        
        # Si c'est un quota et que la firme dépasse, on force le respect du quota
        if action_type == 1:
            q_effective = min(q_chosen, action_value)
        else:
            q_effective = q_chosen

        self.total_q = q_effective # Pour l'instant une seule firme
        
        costs = self.a_i[0] * q_effective + self.b * (q_effective**2)
        firm_profit = (self.p * q_effective) - costs - tax_cost
        
        # 2. Mise à jour pollution
        self.pollution = self.rho * self.pollution + self.alpha * self.total_q
        
        # 3. Social Welfare (Reward Gouvernement)
        social_welfare = firm_profit - self.pollution
        
        return self.pollution, social_welfare, firm_profit
    def step_multi_firms(self, action_type, action_value, q_list):
        """
        q_list : liste des quantités produites par les 5 firmes.
        action_type : 0 (Taxe), 1 (Quota).
        action_value : valeur de la politique.
        """
        individual_profits = []
        current_q_total = 0
        
        for i in range(self.n_firms):
            q_chosen = q_list[i]
            
            # Application de la politique (Taxe ou Quota)
            if action_type == 0: # TAXE
                tax_cost = action_value * self.alpha * q_chosen
                q_eff = q_chosen
            else: # QUOTA
                tax_cost = 0
                q_eff = min(q_chosen, action_value)
            
            # Coût spécifique à la firme i : c_i(q) = a_i*q + b*q^2
            cost = self.a_i[i] * q_eff + self.b * (q_eff**2)
            profit = (self.p * q_eff) - cost - tax_cost
            
            individual_profits.append(profit)
            current_q_total += q_eff

        # Mise à jour du stock de pollution
        self.total_q = current_q_total
        self.pollution = self.rho * self.pollution + self.alpha * self.total_q
        
        # Reward du gouvernement : Bien-être social = Somme des profits - Pollution
        social_welfare = sum(individual_profits) - self.pollution
        
        return self.pollution, social_welfare, individual_profits