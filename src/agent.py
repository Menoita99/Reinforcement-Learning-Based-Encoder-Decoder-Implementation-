from enviroment import Environment
from enviroment import Actions
from collections import deque


class Agent:

    def __init__(self):
        self.env = Environment()
        self.memory = deque()

        self.policyNet # = policyNet()
        self.targetNet # = policyNet.deepCopy()
        self.encoder

        self.epsilon = 0.99975
        self.updateSteps = 100




    def tain(self,episode):
        # TODO
        envEnd = False
        #self.currentState = env.initState()
        for i in range(episode):
            print("training")
            self.env.reset()
            state = self.env.initState()

            while not envEnd:
                envEnd = not envEnd
                # act

                #nextState,reward, flag = self.env.step(action)
                #encode(nextState)

                #cache

                # train

                # if step % updateSteps:
                #   self.target = self.poly.deepCopy()


    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        # action = None
        # if random:
        #    action = Action.random()
        # else:
        #    action = policyNet.foward?
        # TODO
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        #deque.append(currentState,action,reward,nextState)
        # TODO
        pass

    def recall(self):
        """Sample experiences from memory"""
        # deque.sample()
        # TODO
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        # recall
        # train policy
        # TODO
        pass