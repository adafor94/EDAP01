
from operator import pos
import random
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#

class RobotSim:
    def __init__(self, sm : StateModel, tm : TransitionModel, true_state):
        self.sm : StateModel = sm 
        self.tm : TransitionModel = tm
        self.true_state = true_state            # true state

        self.rows, self.cols, self.head = self.sm.get_grid_dimensions()

        print("Hello World / RobotSim")

    def possibleHeadings(self, x, y):
            possible_heading = []
            if x > 0:
                possible_heading.append(0)
            if x < self.rows-1:
                possible_heading.append(2)
            if y > 0:
                possible_heading.append(3)
            if y < self.cols-1:
                possible_heading.append(1)

            return possible_heading

    def next_state(self) -> int :
        x,y,h = self.sm.state_to_pose(self.true_state)          # current pose
        possible_headings = self.possibleHeadings(x,y)

        if h in possible_headings:              #No wall in front of robot
            prob = random.random()
            if prob < 0.3:
                prev = h
                while prev == h:
                    h = random.choice(possible_headings)
        else:
            h = random.choice(possible_headings)
       
        x -= h == 0
        y += h == 1
        x += h == 2
        y -= h == 3
        newState = self.sm.pose_to_state(x,y,h)
        self.true_state = newState

        return newState

    def robot_sensing(self,x, y):               # returns position or None
        Ls = [(x+a, y+b) for a,b in [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0, -1), (1, -1)]]
        Ls2 = [(x+a, y+b) for a,b in [(2,-2), (2,-1), (2,0), (2,1), (2,2), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (1,2), (0,2), (-1,2), (1,-2), (0,-2), (-1,-2)]]
        
        inside_grid = lambda xy: xy[0] >= 0 and xy[0] < self.rows and xy[1] >= 0 and xy[1] < self.cols 
        Ls = list(filter(inside_grid, Ls))
        Ls2 = list(filter(inside_grid, Ls2))

        # print("x y:", x,y)
        # print("LS:", Ls)
        # print("Ls2:", Ls2)

        prob = random.random()

        trueLocation_prob = 0.1
        ls_prob = len(Ls) * 0.05
        ls2_prob = len(Ls2) * 0.025
        if prob <= trueLocation_prob:
            return self.sm.state_to_position(self.true_state)
        elif prob <= trueLocation_prob + ls_prob:
            return random.choice(Ls)
        elif prob <= trueLocation_prob + ls_prob + ls2_prob:
            return random.choice(Ls2)
        else:
            return None

#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self, sm, tm, ob):
        self.sm : StateModel = sm
        self.tm : TransitionModel = tm
        self.ob : ObservationModel = ob
        print("Hello again, World / HMMFilter")
    
    def filtering(self, sense, probs):
        senseReading = self.sm.position_to_reading(sense[0], sense[1]) if sense else None

        T_trans = self.tm.get_T_transp()
        O = self.ob.get_o_reading(senseReading)
        res = np.matmul(np.matmul(O, T_trans), probs)       
        # print("Probs shape", probs.shape, "o shape", O.shape, "Ttran shape:", T_trans.shape)
        # print("res shape", res.shape)
        # print("O:", O)
        res = (1.0 / sum(res) ) * res
        print("Max:", max(res))
        return res

        
        
        
