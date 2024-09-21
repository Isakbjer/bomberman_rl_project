from .my_tabQ import MyRLAgent

def setup(self):
    """
    Initialize the agent.
    """
    self.agent = MyRLAgent()
    self.agent.setup(training=self.train)

def act(self, game_state):
    """
    Choose an action using the agent.
    """
    return self.agent.act(game_state)
