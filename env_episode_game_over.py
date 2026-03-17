import numpy as np


class EcoSystem:

    def __init__(self, n_firms=5, rho=0.9, alpha=0.5, price=10.0,
                 pollution_threshold=4000.0, max_steps=40):
        #pollution_threshold=4000.0 et max_steps=40 à ajuster 
        self.n_firms = n_firms
        self.rho = rho
        self.alpha = alpha
        self.p = price
        self.a_i = np.linspace(1, 5, n_firms)
        self.b = 0.5
        self.pollution_threshold = pollution_threshold
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.pollution = 0.0
        self.total_q = 0.0
        self.prev_policy_val = 0.0
        self.t = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.prev_policy_val,
            self.total_q,
            self.pollution
        ])

    def step(self, action_type, action_value):

        """
        action_type : 0 = taxe, 1 = quota
        action_value : valeur de la politique
        """

        if action_type == 0:  # TAXE
            tax = action_value
            q_is = (self.p - self.a_i - (tax * self.alpha)) / (2 * self.b)
            q_is = np.maximum(0, q_is)
            tax_cost = tax * self.alpha * q_is

        else:  # QUOTA
            q_is_optimal = (self.p - self.a_i) / (2 * self.b)
            q_is = np.maximum(0, np.minimum(q_is_optimal, action_value))
            tax_cost = 0

        self.total_q = np.sum(q_is)

        costs = self.a_i * q_is + self.b * (q_is ** 2)
        profits = (self.p * q_is) - costs - tax_cost

        self.pollution = self.rho * self.pollution + self.alpha * self.total_q

        social_welfare = np.sum(profits) - self.pollution

        self.t += 1 #update temps 

        #condition de fin d'épisode 
        done = False

        # catastrophe écologique
        if self.pollution >= self.pollution_threshold:

            done = True

            social_welfare -= 100 #reward en cas de catastrophe, à ajuster 

        # limite temporelle
        if self.t >= self.max_steps:

            done = True

        self.prev_policy_val = action_value
        state = self._get_state()


        info = {
            "profits": profits,
            "pollution": self.pollution
        }


        return state, social_welfare, done, info


    def step_with_firm(self, action_type, action_value, q_chosen):

        tax_cost = (action_value * self.alpha * q_chosen) if action_type == 0 else 0

        if action_type == 1:
            q_effective = min(q_chosen, action_value)
        else:
            q_effective = q_chosen


        self.total_q = q_effective


        costs = self.a_i[0] * q_effective + self.b * (q_effective ** 2)

        firm_profit = (self.p * q_effective) - costs - tax_cost


        # pollution
        self.pollution = self.rho * self.pollution + self.alpha * self.total_q


        # reward gouvernement
        social_welfare = firm_profit - self.pollution


        self.t += 1

        done = False

        if self.pollution >= self.pollution_threshold:
            done = True
            social_welfare -= 50000 #reward en cas de catastrophe, à ajuster

        if self.t >= self.max_steps:
            done = True


        return self.pollution, social_welfare, firm_profit, done


    def step_multi_firms(self, action_type, action_value, q_list):

        individual_profits = []

        current_q_total = 0


        for i in range(self.n_firms):

            q_chosen = q_list[i]

            if action_type == 0:  # TAXE

                tax_cost = action_value * self.alpha * q_chosen

                q_eff = q_chosen

            else:  # QUOTA

                tax_cost = 0

                q_eff = min(q_chosen, action_value)


            cost = self.a_i[i] * q_eff + self.b * (q_eff ** 2)

            profit = (self.p * q_eff) - cost - tax_cost

            individual_profits.append(profit)

            current_q_total += q_eff


        self.total_q = current_q_total


        self.pollution = self.rho * self.pollution + self.alpha * self.total_q


        social_welfare = sum(individual_profits) - self.pollution


        self.t += 1

        done = False

        if self.pollution >= self.pollution_threshold:
            done = True
            social_welfare -= 50000 #reward en cas de catastrophe, à ajuster

        if self.t >= self.max_steps:
            done = True


        return self.pollution, social_welfare, individual_profits, done