#===============================================================================================
#   This is a class to generalize the game board model -- it summarize info that algorithms
#   must use to work.
#===============================================================================================

class GameModel:
    def __init__(self, valid_actions, dig_action, board_lenght, depot, has_gold_model, relevant_squares):
        self.valid_actions = valid_actions
        self.dig_action = dig_action
        self.board_lenght = board_lenght
        self.depot = depot
        self.has_gold_model = has_gold_model
        self.relevant_squares = relevant_squares
        
    def get_valid_actions(self):
        return self.valid_actions
    
    def get_dig_action(self):
        return self.dig_action
    
    def get_board_lenght(self):
        return self.board_lenght
    
    def get_depot(self):
        return self.depot
    
    def get_has_gold_model(self):
        return self.has_gold_model
    
    def get_relevant_squares(self):
        return self.relevant_squares
    
    def get_dug_lenght(self):
        return len(self.relevant_squares)
    
    def get_action_space_lenght(self):
        return len(self.valid_actions)