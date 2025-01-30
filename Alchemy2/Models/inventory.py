from itertools import combinations_with_replacement
import os
os.chdir('/Github/LLMs_game/Alchemy2')
import json

class Inventory():
    def __init__(self):

        self.parents = {int(k):v for k,v in json.load(open('dataset/alchemy2ParentTable.json', 'r')).items()}
        self.combination_table = json.load(open('dataset/alchemy2CombinationTable.json', 'r'))
        # Initialize inventory attributes
        self.inventory_used = {0, 1, 2, 3}  # Initialize
        self.inventory_total = {0, 1, 2, 3} 

    def reset(self):
        """Resets inventory to basic elements.
        """
        self.inventory_used = {0,1,2,3}
        self.inventory_total = {0,1,2,3}
        self.update_success_list()


    def update(self, combination_results):
        """Updates inventory for given combination and results. Stores inventory length for given run and step.

        Args:
            combination (list): List of two element indices involved in last combination.
            combination_results (list): List of element indices that resulted from combination.
        """
        # store current inventory
        inventory_used_temp = self.inventory_used.copy()
        inventory_total_temp = self.inventory_total.copy()

        # update total inventory
        if combination_results == [-1]:
            combination_results = []
        self.inventory_total.update(combination_results)
        condition_elements = self.check_conditions()
        self.inventory_total.update(condition_elements)

        # update used inventory
        self.inventory_used = self.inventory_total.copy()

        #they should be the same
        new_results_non_final = list(self.inventory_used.difference(inventory_used_temp))
        new_results_total = list(self.inventory_total.difference(inventory_total_temp))

        return (new_results_non_final, new_results_total)

    def check_conditions(self):
        """Returns elements for which conditions are fulfilled.

        Returns:
            list: List of elements indices for which conditions are fulfilled.
        """
        condition_elements = list()
        inventory_size = len(self.inventory_total)

        # check for size conditions
        if (inventory_size >= 50) and (35 not in self.inventory_total):
            condition_elements.append(35)
        if (inventory_size >= 100) and (40 not in self.inventory_total):
            condition_elements.append(40)
        if (inventory_size >= 150) and (605 not in self.inventory_total):
            condition_elements.append(605)
        if (inventory_size >= 150) and (606 not in self.inventory_total):
            condition_elements.append(606)
        if (inventory_size >= 150) and (566 not in self.inventory_total):
            condition_elements.append(566)
        if (inventory_size >= 300) and (627 not in self.inventory_total):
            condition_elements.append(627)

        # check for set conditions
        light_set = {367, 114, 239, 280, 593, 136, 125, 21, 107, 176, 434, 25, 46, 109, 133, 284}

        if (121 not in self.inventory_total) and (len(self.inventory_total.intersection(light_set)) >= 5):
            condition_elements.append(121)

        small_set = {165, 325, 332, 484, 574, 570, 485, 422, 429, 419, 339, 222}
        if (572 not in self.inventory_total) and (len(self.inventory_total.intersection(small_set)) >= 5):
            condition_elements.append(572)

        motion_set = {569, 265, 19, 150, 375, 15, 12, 16, 20, 22, 25, 32, 40, 45, 248}
        if (571 not in self.inventory_total) and (len(self.inventory_total.intersection(motion_set)) >= 5):
            condition_elements.append(571)

        return condition_elements

    def update_success_list(self):
        """Updates the lists of successful and unsuccessful combinations."""
        self.inventory_successfull = []  # Clear previous values
        self.inventory_not_successfull = []  # Clear previous values

        for (combo) in combinations_with_replacement(self.inventory_used, 2):
            if str(combo[0]) in self.combination_table and str(combo[1]) in self.combination_table[str(combo[0])]:
                self.inventory_successfull.append(combo)
            else:
                self.inventory_not_successfull.append(combo)

# test update_success_list
#inventory = Inventory()
#inventory.reset()  # Reset inventory before updating success list
#inventory.update_success_list()
#print(inventory.inventory_successfull)
#print(inventory.inventory_not_successfull)


