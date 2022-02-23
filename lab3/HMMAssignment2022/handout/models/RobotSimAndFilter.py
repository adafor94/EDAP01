
import random
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#

# The robot moves according to the following strategy:
# Pick random start heading h_0. For any new step pick new heading h_t+1 based on the current
# heading h_t according to:
# P( h_t+1 = h_t | not encountering a wall) = 0.7
# P( h_t+1 != h_t | not encountering a wall) = 0.3
# P( h_t+1 = h_t | encountering a wall) = 0.0
# P( h_t+1 != h_t | encountering a wall) = 1.0
# It then moves in the direction h_t+1 by one step in the grid. This means essentially that a) it will
# always move one step and b) it can only move straight.
# In case a new heading is to be found, the new one is randomly chosen from the possible ones
# (facing a wall somewhere along the wall leaves three, facing the wall in a corner leaves two options
# for where to turn, see one example below in hint 1).

class RobotSim:
    def __init__(self, sm : StateModel, tm : TransitionModel, true_state):
        self.sm : StateModel = sm 
        self.tm : TransitionModel = tm
        
        self.true_state = true_state            # true state

        self.rows, self.cols, self.head = self.__sm.get_grid_dimensions()

        print("Hello World")

    def next_state(self) -> int :
        x,y,h = self.sm.state_to_pose(self.true_state)          # current pose
        prev_h = h

        #new coordinates if moving forward
        x -= h == 0
        y += h == 1
        x += h == 2
        y -= h == 3
        
        if x < 0 or x > self.rows or y < 0 or y > self.cols:            #if encountering a wall
            while h == prev_h:
                h = random.randint(0,4)                         # loop until new h
        else:
            probability = random.random()
            if probability < 0.3:
                while h == prev_h:
                    h = random.randint(0,4)
        
        newState = self.sm.pose_to_state(x,y,h)
        self.true_state = newState
        
        return newState

    


#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self):
        print("Hello again, World")

        
        
        
