import random as rd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#
# Definitions
#
# Order of layers: Red, Green, Blue, Purple
#
ACTIONS_DIMENSION = 142

REGULAR_PLATE = 0.1
BONUS_PLATE = 0.4

COLORS = { 0: "red", 1: "lightgreen", 2: "cyan", 3: "purple"}

moves = {1: ((0, 0), (1, 0)), 2: ((0, 1), (1, 1)), 3: ((0, 2), (1, 2)), 4: ((0, 3), (1, 3)), 5: ((0, 4), (1, 4)), 
         6: ((0, 5), (1, 5)), 7: ((1, 0), (2, 0)), 8: ((1, 0), (0, 0)), 9: ((1, 1), (2, 1)), 10: ((1, 1), (0, 1)), 
         11: ((1, 2), (2, 2)), 12: ((1, 2), (0, 2)), 13: ((1, 3), (2, 3)), 14: ((1, 3), (0, 3)), 15: ((1, 4), (2, 4)), 
         16: ((1, 4), (0, 4)), 17: ((1, 5), (2, 5)), 18: ((1, 5), (0, 5)), 19: ((2, 0), (3, 0)), 20: ((2, 0), (1, 0)), 
         21: ((2, 1), (3, 1)), 22: ((2, 1), (1, 1)), 23: ((2, 2), (3, 2)), 24: ((2, 2), (1, 2)), 25: ((2, 3), (3, 3)), 
         26: ((2, 3), (1, 3)), 27: ((2, 4), (3, 4)), 28: ((2, 4), (1, 4)), 29: ((2, 5), (3, 5)), 30: ((2, 5), (1, 5)), 
         31: ((3, 0), (4, 0)), 32: ((3, 0), (2, 0)), 33: ((3, 1), (4, 1)), 34: ((3, 1), (2, 1)), 35: ((3, 2), (4, 2)), 
         36: ((3, 2), (2, 2)), 37: ((3, 3), (4, 3)), 38: ((3, 3), (2, 3)), 39: ((3, 4), (4, 4)), 40: ((3, 4), (2, 4)), 
         41: ((3, 5), (4, 5)), 42: ((3, 5), (2, 5)), 43: ((4, 0), (5, 0)), 44: ((4, 0), (3, 0)), 45: ((4, 1), (5, 1)), 
         46: ((4, 1), (3, 1)), 47: ((4, 2), (5, 2)), 48: ((4, 2), (3, 2)), 49: ((4, 3), (5, 3)), 50: ((4, 3), (3, 3)), 
         51: ((4, 4), (5, 4)), 52: ((4, 4), (3, 4)), 53: ((4, 5), (5, 5)), 54: ((4, 5), (3, 5)), 55: ((5, 0), (6, 0)), 
         56: ((5, 0), (4, 0)), 57: ((5, 1), (6, 1)), 58: ((5, 1), (4, 1)), 59: ((5, 2), (6, 2)), 60: ((5, 2), (4, 2)), 
         61: ((5, 3), (6, 3)), 62: ((5, 3), (4, 3)), 63: ((5, 4), (6, 4)), 64: ((5, 4), (4, 4)), 65: ((5, 5), (6, 5)), 
         66: ((5, 5), (4, 5)), 67: ((6, 0), (5, 0)), 68: ((6, 1), (5, 1)), 69: ((6, 2), (5, 2)), 70: ((6, 3), (5, 3)), 
         71: ((6, 4), (5, 4)), 72: ((6, 5), (5, 5)), 73: ((0, 0), (0, 1)), 74: ((1, 0), (1, 1)), 75: ((2, 0), (2, 1)), 
         76: ((3, 0), (3, 1)), 77: ((4, 0), (4, 1)), 78: ((5, 0), (5, 1)), 79: ((6, 0), (6, 1)), 80: ((0, 1), (0, 0)), 
         81: ((0, 1), (0, 2)), 82: ((1, 1), (1, 0)), 83: ((1, 1), (1, 2)), 84: ((2, 1), (2, 0)), 85: ((2, 1), (2, 2)), 
         86: ((3, 1), (3, 0)), 87: ((3, 1), (3, 2)), 88: ((4, 1), (4, 0)), 89: ((4, 1), (4, 2)), 90: ((5, 1), (5, 0)), 
         91: ((5, 1), (5, 2)), 92: ((6, 1), (6, 0)), 93: ((6, 1), (6, 2)), 94: ((0, 2), (0, 1)), 95: ((0, 2), (0, 3)), 
         96: ((1, 2), (1, 1)), 97: ((1, 2), (1, 3)), 98: ((2, 2), (2, 1)), 99: ((2, 2), (2, 3)), 100: ((3, 2), (3, 1)), 
         101: ((3, 2), (3, 3)), 102: ((4, 2), (4, 1)), 103: ((4, 2), (4, 3)), 104: ((5, 2), (5, 1)), 105: ((5, 2), (5, 3)), 
         106: ((6, 2), (6, 1)), 107: ((6, 2), (6, 3)), 108: ((0, 3), (0, 2)), 109: ((0, 3), (0, 4)), 110: ((1, 3), (1, 2)), 
         111: ((1, 3), (1, 4)), 112: ((2, 3), (2, 2)), 113: ((2, 3), (2, 4)), 114: ((3, 3), (3, 2)), 115: ((3, 3), (3, 4)), 
         116: ((4, 3), (4, 2)), 117: ((4, 3), (4, 4)), 118: ((5, 3), (5, 2)), 119: ((5, 3), (5, 4)), 120: ((6, 3), (6, 2)), 
         121: ((6, 3), (6, 4)), 122: ((0, 4), (0, 3)), 123: ((0, 4), (0, 5)), 124: ((1, 4), (1, 3)), 125: ((1, 4), (1, 5)), 
         126: ((2, 4), (2, 3)), 127: ((2, 4), (2, 5)), 128: ((3, 4), (3, 3)), 129: ((3, 4), (3, 5)), 130: ((4, 4), (4, 3)), 
         131: ((4, 4), (4, 5)), 132: ((5, 4), (5, 3)), 133: ((5, 4), (5, 5)), 134: ((6, 4), (6, 3)), 135: ((6, 4), (6, 5)), 
         136: ((0, 5), (0, 4)), 137: ((1, 5), (1, 4)), 138: ((2, 5), (2, 4)), 139: ((3, 5), (3, 4)), 140: ((4, 5), (4, 4)), 
         141: ((5, 5), (5, 4)), 142: ((6, 5), (6, 4))}


def color_fits_3D(field, i, j, depth):
    """
    Checks if two items to the left or two colors to the bottom or two colors to the right are NOT of the same color as the new item.
    
    Input:
    - field: battfield, numpy array (7, 6, 4)
    - i, j, depth: position on the new item, int, within field.shape
    Output:
    - boolean: True, if the new item is ok
    """
    # Check two colors to the left
    if (j > 1):
        if (field[i, j - 2, depth] > 0) and (field[i, j - 1, depth] > 0):
            return False
        
    # Check two colors to the right
    if (j < 4):
        if (field[i, j + 2, depth] > 0) and (field[i, j + 1, depth] > 0):
            return False
    
    # Check two colors to the bottom
    if (i < 5):
        if (field[i + 2, j, depth] > 0) and (field[i + 1, j, depth] > 0):
            return False
    
    return True


def initialize_field_3D(field):
    """
    Initialization of the battle field.
    Move from bottom left corner and add new elements.
    
    Input: 
    - field: numpy array of zeros, shape = (7, 6, 4)
    Output:
    - field: numpy array of floats, shape = (7, 6, 4)
    """
    rd.seed()
    
    for i in reversed(range(field.shape[0])):
        for j in range(field.shape[1]):      
            new_color = rd.randrange(4)
            
            while not color_fits_3D(field, i, j, new_color):
                new_color = rd.randrange(4)
                
            field[i, j, new_color] = REGULAR_PLATE
    
    return field


def visualize_field_3D(field):
    """
    Visualizes the battle field in colored circles
    Handles bonus plates
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    Output:
    - None
    """
    fig, ax = plt.subplots(figsize=(5, 7))

    ax.set_xlim((0, 10))
    ax.set_ylim((0, 13))

    circles = []

    for ii in range(7):
        for jj in range(6):
            clr = COLORS[field[ii, jj, :].argmax()]

            #if (field[ii, jj] // 1 == 1.0):
            #    circles.append( mpatches.RegularPolygon((jj + 1, 7 - ii), numVertices=4, radius=0.4, color=clr) )
            #else:
            #    circles.append( mpatches.Circle((jj + 1, 7 - ii), radius=0.4, color=clr) )
             
            #
            # DEBUG
            #
            #if (field[ii, jj].sum() == 0.1):
            #    circles.append( mpatches.RegularPolygon((jj + 1, 7 - ii), numVertices=3, radius=0.2, color="black") )
            #elif (field[ii, jj] // 1 == 1.0):
            #    circles.append( mpatches.RegularPolygon((jj + 1, 7 - ii), numVertices=4, radius=0.4, color=clr) )
            #else:
            #    circles.append( mpatches.Circle((jj + 1, 7 - ii), radius=0.4, color=clr) )
            
            if (field[ii, jj].sum() == 0.1):
                # Regular plate
                circles.append( mpatches.Circle((jj + 1, 7 - ii), radius=0.4, color=clr) )
            elif (field[ii, jj].sum() == 0.4):
                # Bonus plate of 4
                circles.append( mpatches.RegularPolygon((jj + 1, 7 - ii), numVertices=4, radius=0.4, color=clr) )
            else:
                # Premium plate of 5
                circles.append( mpatches.RegularPolygon((jj + 1, 7 - ii), numVertices=3, radius=0.2, color="black") )

    for circ in circles:
        ax.add_artist(circ)
        
    return


def make_move_v2_3D(field, move, moves):
    """
    Physically moves plates according to the move description
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    - move: particular move to make, 1<=move<=142
    - moves: dictionary of all possible moves defined above
    Output:
    - new_field: updated field with two swapped plates
    - plate_start: coordinates of the plate that started the move, tuple (row, column)
    - plate_end: cooredinates of the plate that ended the move, tuple (row, column)
    """
    (start_row, start_col), (end_row, end_col) = moves[move]

    # Swap two plates and create new (modified) field
    new_field = field.copy()
    temp_color = field[end_row, end_col, :].copy()
    new_field[end_row, end_col, :] = field[start_row, start_col, :]
    new_field[start_row, start_col, :] = temp_color
        
    return new_field, (start_row, start_col), (end_row, end_col)


def fill_field_3D(field):
    """
    Fills the field after burning the sets
    Moves the plates downward, filling the upper row so that it doesn't have "easy" sets of three
    Starts from the left lower corner, in order to reuse color_fits_3D()
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    Output:
    - updated field: numpy array of floats, (7, 6, 4)
    """  
    for ii in reversed(range(field.shape[0])):
        for jj in range(field.shape[1]):
            while (field[ii, jj, :].sum() == 0.):
                # Put downward by one
                # Not needed if we are in the uppermost row
                if (ii != 0):
                    for iii in reversed(range(ii)):
                        field[iii + 1, jj, :] = field[iii, jj, :].copy()
                    
                    field[0, jj, :] = 0.

                # Fill the top
                new_color = rd.randrange(4)

                while not color_fits_3D(field, 0, jj, new_color):
                    new_color = rd.randrange(4)

                field[0, jj, new_color] = REGULAR_PLATE 
                
    return field


def get_sets_3D(field):
    """
    Finds all sets and all bonus plates included into those sets
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    Output:
    - list of sets coordinates: list of tuples (row_start, column_start, set_length, direction, color), counting from top left corner
      Direction is either 0 (horizontal) or 1 (vertical)
    - list of bonus plates included into sets: list of tuples (row, column, type). 
      Type is either 4 or 5 (reserved for future)
    """
    output_bonus_plates = []
    output_sets = []

    # Find all 3+ sets in horizontal row
    for ii in range(field.shape[0]):
        temp_bonus_plates = []
        jj = 0
        len = 1
        while (jj < field.shape[1]):
            if (jj > 0):
                if (((field[ii, jj, :] > 0) == (field[ii, jj - 1, :] > 0)).all()):
                    len = len + 1
                else:
                    if (len >= 3):
                        # Add temp list of bonus plates to the permanent list of bonus plates
                        output_bonus_plates = output_bonus_plates + temp_bonus_plates
                        
                        # Add to permanent list of sets
                        output_sets.append((ii, jj - len, len, 0, field[ii, jj - 1, :].argmax()))
                        
                    temp_bonus_plates = []
                    len = 1
            
            if (field[ii, jj, :].sum() > 0.1):
                # Add to temp list of bonus plates
                temp_bonus_plates.append((ii, jj, 4))
            
            jj = jj + 1
            
        if (len >= 3):
            # Add temp list of bonus plates to the permanent list of bonus plates
            output_bonus_plates = output_bonus_plates + temp_bonus_plates

            # Add to permanent list of sets
            output_sets.append((ii, jj - len, len, 0, field[ii, jj - 1, :].argmax()))

    # Find all 3+ sets in vertical columns
    for jj in range(field.shape[1]):
        temp_bonus_plates = []
        ii = 0
        len = 1
        while (ii < field.shape[0]):
            if (ii > 0):
                if (((field[ii, jj, :] > 0) == (field[ii - 1, jj, :] > 0)).all()):
                    len = len + 1
                else:
                    if (len >= 3):
                        # Add temp list of bonus plates to the permanent list of bonus plates
                        output_bonus_plates = output_bonus_plates + temp_bonus_plates
                        
                        # Add to permanent list of sets
                        output_sets.append((ii - len, jj, len, 1, field[ii - 1, jj, :].argmax()))
                        
                    temp_bonus_plates = []
                    len = 1
            
            if (field[ii, jj, :].sum() > 0.1):
                # Add to temp list of bonus plates
                temp_bonus_plates.append((ii, jj, 4))
            
            ii = ii + 1
            
        if (len >= 3):
            # Add temp list of bonus plates to the permanent list of bonus plates
            output_bonus_plates = output_bonus_plates + temp_bonus_plates

            # Add to permanent list of sets
            output_sets.append((ii - len, jj, len, 1, field[ii - 1, jj, :].argmax()))
            
    return output_sets, output_bonus_plates


def plate_in_set_3D(plate, row, col, length, direction):
    """
    Checks whether plate is in the set given by row, col, length, direction
    
    Input:
    - plate: plate location, tuple (row, column)
    - row: row where the set starts
    - col: column where the set starts
    - length: the set's length
    - direction: the set's direction
    Output:
    - True if the plate is within the set, False otherwise
    """
    if (direction == 0):
        # Horizontal set
        if ((plate[0] != row) or (plate[1] < col) or (plate[1] > (col + length - 1))):
            return False
    else:
        # Vertucal set
        if ((plate[1] != col) or (plate[0] < row) or (plate[0] > (row + length - 1))):
            return False
        
    return True


def calculate_score_v2_3D(field, plate_from, plate_to):
    """
    Calculates the score in the field. 
    Replaces all sets with zeros.
    Handles bonus plates: replaces required rows with zeros (Type 4)
    Puts bonus plates, should any set be of the length of 4
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    - plate_from: coordinates of the plate where the move starts, tuple (row, column)
    - plate_to: coordinates of the plate where the move ends, tuple (row, column)
    Output:
    - score: int, 0+
    - field: modified field, (7, 6, 4)
    """
    # Get all sets with possible bonus plates
    sets, bonus_plates = get_sets_3D(field)

    # Set all requires plates to zero
    #
    # First handle sets
    for st in sets:
        row = st[0]
        col = st[1]
        lng = st[2]
        drc = st[3]

        if (drc == 0):
            field[row, col:(col + lng), :] = 0.
        else:
            field[row:(row + lng), col, :] = 0.
    #       
    # Then handle bonus plates/rows
    for pl in bonus_plates:
        row = pl[0]
        col = pl[1]
        typ = pl[2]

        if (typ == 4):
            field[row, :, :] = 0.

    # Calculate score
    score = (field.sum(axis=2) == 0.).sum()

    # Put new bonus plates. Specially care for the move coordinates
    for st in sets:
        row = st[0]
        col = st[1]
        lng = st[2]
        drc = st[3]
        clr = st[4]

        if (lng >= 4):
            if (plate_in_set_3D(plate_from, row, col, lng, drc)):
                # Move start plate in set. Put new bonus plate according to the move coordinates
                field[plate_from[0], plate_from[1], clr] = BONUS_PLATE
                
                #
                # DEBUG
                #
                #print("DEBUG: set of 4+ was made!")
                
            elif (plate_in_set_3D(plate_to, row, col, lng, drc)):
                # Move end plate in set. Put new bonus plate according to the move coordinates
                field[plate_to[0], plate_to[1], clr] = BONUS_PLATE
                
                #
                # DEBUG
                #
                #print("DEBUG: set of 4+ was made!")
                
            else:
                # Just put the new bonus plate at the very right/bottom of the set
                # This CANNOT happen during the manual move!
                # It CAN ONLY HAPPEN when the field is randomly filled with new plates
                if (drc == 0):
                    field[row, col + lng - 1, clr] = BONUS_PLATE
                else:
                    field[row + lng - 1, col, clr] = BONUS_PLATE

    return score, field


def predict_max_score_3D(field, aero_cnn, number_of_moves, moves, debug_flag=False):
    """
    Predicts best move for the given field.
    It makes every move, stacks resulting fields and predicts.
    
    Input:
    - field: numpy array of floats, (7, 6, 4)
    - aero_cnn: CNN
    - number_of_moves: int, number of possible moves in the game (normally, 142)
    - moves: dict, explains every move 
    - debug_flag: boolean, whether to output predictions
    Output:
    - best_move_score: float, best predicted score
    - best_move_number: int, number of the corresponding move
    """
    X_data = ()
           
    for move in range(1, number_of_moves + 1):
        swapped, _, _ = make_move_v2_3D(field.copy(), move, moves)
        X_data = X_data + ( swapped, )
    
    # Now select the most successful move
    X_data = np.stack(X_data, axis=0)   
    prediction = aero_cnn.predict(X_data)
    
    # If needed to debug, output prediction
    if (debug_flag):
        print("DEBUG", prediction)
    
    best_move_score = prediction.max()
    best_move_number = prediction.argmax() + 1
    
    return best_move_score, best_move_number


