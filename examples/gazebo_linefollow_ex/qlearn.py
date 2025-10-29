import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # Load Q values from pickle file
        with open(filename+".pickle", 'rb') as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # Save Q values to pickle file
        with open(filename+".pickle", 'wb') as f:
            pickle.dump(self.q, f)
        
        # Save Q values to CSV file for easy viewing
        with open(filename+".csv", 'w') as f:
            for key, value in self.q.items():
                f.write("{}, {}\n".format(key, value))

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # Implement epsilon-greedy exploration vs exploitation
        # if random value < epsilon, take random action (exploration)
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            if return_q:
                return action, self.getQ(state, action)
            else:
                return action
        else:
            # exploitation: choose action with highest Q value
            # get Q values for all actions in current state
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # handle edge case: multiple actions with same max Q value
            # collect all actions that have the max Q value
            best_actions = [self.actions[i] for i in range(len(self.actions)) 
                           if q_values[i] == max_q]
            
            # randomly choose among best actions to break ties
            action = random.choice(best_actions)
            
            if return_q:
                return action, max_q
            else:
                return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        
        # Get current Q value for (state1, action1)
        # If not in dictionary, getQ returns 0.0 (edge case handled)
        old_q = self.getQ(state1, action1)
        
        # Find max Q value for all actions in state2
        # Get Q values for all possible actions from state2
        future_q_values = [self.getQ(state2, a) for a in self.actions]
        
        # Handle edge case: if state2 is terminal or no actions, max Q is 0
        if future_q_values:
            max_future_q = max(future_q_values)
        else:
            max_future_q = 0.0
        
        # Update Q value using Bellman equation
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        
        # Store updated Q value
        self.q[(state1, action1)] = new_q
