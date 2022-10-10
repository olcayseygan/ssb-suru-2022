
class EvaluationAgent():

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        pass

    def act(self, observation):
        
        return self.action_space.sample()
    

