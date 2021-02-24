"""

"""
import datetime
import os
import numpy
import torch

import pandas as pd

from .abstract_game import AbstractGame

from xgboost import XGBClassifier

import treelite_runtime

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1,1,52) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(26)) # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1)) # List of players. You should only edit the length
        self.stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0 # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 4 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 51 # Maximum number of moves if game is not finished before
        self.num_simulations = 51 # Number of future moves self-simulated
        self.discount = 1 # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.03  # Initial learning rate
        self.lr_decay_rate = 0.75  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 150000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Overtime(seed)

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        #print(observation, reward, done)
        return observation, reward , done


    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Approve 0 Extra Time Off",
            1: "Approve 1 Extra Time Off",
            2: "Approve 2 Extra Time Off",
            3: "Approve 3 Extra Time Off",
            4: "Approve 4 Extra Time Off",
            5: "Approve 5 Extra Time Off",
            6: "Approve -1 Extra Time Off",
            7: "Approve -2 Extra Time Off",
            8: "Approve -3 Extra Time Off",
            9: "Approve -4 Extra Time Off",
            10: "Approve -5 Extra Time Off",
            11: "Approve -6 Extra Time Off",
            12: "Approve -7 Extra Time Off",
            13: "Approve -8 Extra Time Off",
            14: "Approve -9 Extra Time Off",
            15: "Approve -10 Extra Time Off",
            16: "Approve -11 Extra Time Off",
            17: "Approve -12 Extra Time Off",
            18: "Approve -13 Extra Time Off",
            19: "Approve -14 Extra Time Off",
            20: "Approve -15 Extra Time Off",
            21: "Approve -16 Extra Time Off",
            22: "Approve -17 Extra Time Off",
            23: "Approve -18 Extra Time Off",
            24: "Approve -19 Extra Time Off",
            25: "Approve -20 Extra Time Off"
        }
        return f"{action_number}. {actions[action_number]}"


class Overtime:
    def __init__(self, seed):
        self.data = numpy.load('data.npy')
        self.week_data = pd.read_csv('week.csv')['CB_TOTAL'].to_numpy()
        self.model = treelite_runtime.Predictor('./mymodel.so', verbose=True)
        self.random = numpy.random.RandomState(seed)
        self.week = 0
        self.weeks = numpy.zeros(52)
        self.savings = 0
        self.vac = 0
        self.vacation = 0

    def reset(self):
        self.savings = 0
        self.vac = 0
        self.week = 0
        self.weeks = numpy.zeros(52)
        self.vacation = 0
        return self.get_observation()

    """
    Action: 0 = Reject Extra Time Off
    Action: 1 = Approve Extra Time Off
    """

    def step(self, action):
        try:
            self.vacation = 0
            if action < 6:
                self.vacation = action
            elif action > 5:
                self.vacation = (action - 5)*-1
            done = self.week == 51
            reward = self.get_reward(done)
            observe = self.get_observation()
        except Exception as e:
            reward = -1000
            print(e)
        return observe, reward, done

    def get_observation(self):
        #data = self.get_data(self.week)
        prediction = self.predict()
        self.weeks[self.week]=prediction
        self.week += 1
        return self.weeks.reshape(1,1,52)

    def reward(self):
        try:
            callbacks = self.week_data[self.week]
            delta = 0
            reward = 0
            if callbacks >=0:
                delta = callbacks + self.vacation
                if delta <= 0:
                    #self.savings += callbacks*1000
                    reward = callbacks*1000
                elif self.vacation < 0:
                    reward = self.vacation*-1000
                else:
                    #self.savings += -1*self.vacation*1000  
                    reward = -1*self.vacation*1000
    #         else:
    #             delta = callbacks + self.vacation
    #             if delta >= 0:
    #                 self.vac += callbacks*-1
    #                 reward = callbacks*-1
    #             else:
    #                 self.vac += self.vacation
    #                 reward = self.vacation
    #         if delta <=0:
    #             reward = numpy.abs(self.min - delta)*10/numpy.abs(self.min)
    #         else:
    #             reward = numpy.abs(self.max - delta)*10/numpy.abs(self.max)
        except Exception as e: 
            reward = 0
            print(e)
        return round(reward)
        
    def get_data(self):
        return self.data[self.week]
    
#     def get_actual(self):
        
    
    def predict(self):
        data = self.get_data().reshape(1,21)
        data = treelite_runtime.Batch.from_npy2d(data)
        predict = self.model.predict(data).round().astype(int)
        return numpy.random.randint(2, size =1)
        
    def legal_actions(self):
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

    def get_reward(self, done):
        reward = self.reward()
        return reward

    def render(self):
        print("Vacation: " + str(self.vacation))
