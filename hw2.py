# BFS
def BFS(TREE):
    if not isinstance(TREE, tuple):
        return (TREE, )
    
    q = [TREE]
    output = []
    while q:
        node = q.pop(0)
        if isinstance(node, tuple):
            for child in node: # left to right
                q.append(child)
        else:
            output.append(node) # only leaves

    return tuple(output)

def DFS(TREE):
    out = []

    def go(node):
        if not isinstance(node, tuple):
            out.append(node)
            return
        for child in node: # left to right
            go(child)

    go(TREE)
    return tuple(out)

#DFID
def DFID(TREE, Depth):
    output = []
    def DLS(node, depth, limit):
        if not isinstance(node, tuple):
            if depth <= limit:
                output.append(node)
            return
        if depth == limit:
            return # cutoff
        # right to left expansion
        for child in reversed(node):
            DLS(child, depth + 1, limit)

    for limit in range(Depth + 1):
        DLS(TREE, 0, limit)
    return tuple(output)
    

# These functions implement a depth-first solver for the homer-baby-dog-poison
# problem. In this implementation, a state is represented by a single tuple
# (homer, baby, dog, poison), where each variable is True if the respective entity is
# on the west side of the river, and False if it is on the east side.
# Thus, the initial state for this problem is (False False False False) (everybody
# is on the east side) and the goal state is (True True True True).

# The main entry point for this solver is the function DFS_SOL, which is called
# with (a) the state to search from and (b) the path to this state. It returns
# the complete path from the initial state to the goal state: this path is a
# list of intermediate problem states. The first element of the path is the
# initial state and the last element is the goal state. Each intermediate state
# is the state that results from applying the appropriate operator to the
# preceding state. If there is no solution, DFS_SOL returns [].
# To call DFS_SOL to solve the original problem, one would call
# DFS_SOL((False, False, False, False), [])
# However, it should be possible to call DFS_SOL with any intermediate state (S)
# and the path from the initial state to S (PATH).

# First, we define the helper functions of DFS_SOL.

# FINAL_STATE takes a single argument S, the current state, and returns True if it
# is the goal state (True, True, True, True) and False otherwise.
def FINAL_STATE(S):
    return S == (True, True, True, True)


# NEXT_STATE returns the state that results from applying an operator to the
# current state. It takes two arguments: the current state (S), and which entity
# to move (A, equal to "h" for homer only, "b" for homer with baby, "d" for homer
# with dog, and "p" for homer with poison).
# It returns a list containing the state that results from that move.
# If applying this operator results in an invalid state (because the dog and baby,
# or poisoin and baby are left unsupervised on one side of the river), or when the
# action is impossible (homer is not on the same side as the entity) it returns [].
# NOTE that NEXT_STATE returns a list containing the successor state (which is
# itself a tuple)# the return should look something like [(False, False, True, True)].
def NEXT_STATE(S, A):
    homer, baby, dog, poison = S

    # homer always moves
    nhomer = not homer
    nbaby, ndog, npoison = baby, dog, poison

    if A == "h":
        pass # homer moves alone
    elif A == "b":
        if baby != homer:
            return [] # can't move baby if it's not on the same side as homer
        nbaby = not baby
    elif A == "d":
        if dog != homer:
            return [] # can't move dog if it's not on the same side as homer
        ndog = not dog
    elif A == "p":
        if poison != homer:
            return [] # can't move poison if it's not on the same side as homer
        npoison = not poison
    else:
        return [] # invalid action
    
    new_state = (nhomer, nbaby, ndog, npoison)
    # check for invalid states
    homer2, baby2, dog2, poison2 = new_state
    if (baby2 == dog2 and baby2 != homer2) or (baby2 == poison2 and baby2 != homer2):
        return [] # invalid state: baby is left with dog or poison without homer
    return [new_state]


# SUCC_FN returns all of the possible legal successor states to the current
# state. It takes a single argument (S), which encodes the current state, and
# returns a list of each state that can be reached by applying legal operators
# to the current state.
def SUCC_FN(S):
    successors = []
    for action in ["h", "b", "d", "p"]:
        next_states = NEXT_STATE(S, action)
        successors.extend(next_states)
    return successors


# ON_PATH checks whether the current state is on the stack of states visited by
# this depth-first search. It takes two arguments: the current state (S) and the
# stack of states visited by DFS (STATES). It returns True if S is a member of
# STATES and False otherwise.
def ON_PATH(S, STATES):
    return S in STATES


# MULT_DFS is a helper function for DFS_SOL. It takes two arguments: a list of
# states from the initial state to the current state (PATH), and the legal
# successor states to the last, current state in the PATH (STATES). PATH is a
# first-in first-out list of states# that is, the first element is the initial
# state for the current search and the last element is the most recent state
# explored. MULT_DFS does a depth-first search on each element of STATES in
# turn. If any of those searches reaches the final state, MULT_DFS returns the
# complete path from the initial state to the goal state. Otherwise, it returns
# [].
def MULT_DFS(STATES, PATH):
    for s in STATES:
        result = DFS_SOL(s, PATH + [s])
        if result:
            return result
    return []


# DFS_SOL does a depth first search from a given state to the goal state. It
# takes two arguments: a state (S) and the path from the initial state to S
# (PATH). If S is the initial state in our search, PATH is set to []. DFS_SOL
# performs a depth-first search starting at the given state. It returns the path
# from the initial state to the goal state, if any, or [] otherwise. DFS_SOL is
# responsible for checking if S is already the goal state, as well as for
# ensuring that the depth-first search does not revisit a node already on the
# search path (i.e., S is not on PATH).
def DFS_SOL(S, PATH):
    if PATH == []:
        PATH = [S]
    if FINAL_STATE(S):
        return PATH
    if ON_PATH(S, PATH[:-1]): # avoid revisiting states on the current path
        return []
    
    successors = [new_state for new_state in SUCC_FN(S) if not ON_PATH(new_state, PATH)]
    return MULT_DFS(successors, PATH)



# Test cases 
# 1) BFS tests
def test_BFS():
    assert BFS("ROOT") == ("ROOT",)
    assert BFS(((("L", "E"), "F"), "T")) == ("T", "F", "L", "E")
    assert BFS(("R", ("I", ("G", ("H", "T"))))) == ("R", "I", "G", "H", "T")
    assert BFS((("A", ("B",)), ("C",), "D")) == ("D", "A", "C", "B")
    assert BFS(("T", ("H", "R", "E"), "E")) == ("T", "E", "H", "R", "E")
    assert BFS(("A", (("C", (("E",), "D")), "B"))) == ("A", "B", "C", "D", "E")

# 2) DFS tests
def test_DFS():
    assert DFS("ROOT") == ("ROOT",)
    assert DFS(((("L", "E"), "F"), "T")) == ("L", "E", "F", "T")
    assert DFS(("R", ("I", ("G", ("H", "T"))))) == ("R", "I", "G", "H", "T")
    assert DFS((("A", ("B",)), ("C",), "D")) == ("A", "B", "C", "D")
    assert DFS(("T", ("H", "R", "E"), "E")) == ("T", "H", "R", "E", "E")
    assert DFS(("A", (("C", (("E",), "D")), "B"))) == ("A", "C", "E", "D", "B")

# 3) DFID tests
def test_DFID():
    assert DFID("ROOT", 0) == ("ROOT",)
    assert DFID(((("L", "E"), "F"), "T"), 3) == ("T", "T", "F", "T", "F", "E", "L")
    assert DFID(("R", ("I", ("G", ("H", "T")))), 4) == (
        "R", "I", "R", "G", "I", "R",
        "T", "H", "G", "I", "R"
    )
    assert DFID(((("A", ("B",)), ("C",), "D")), 3) == (
        "D", "D", "C", "A",
        "D", "C", "B", "A"
    )
    assert DFID(("T", ("H", "R", "E"), "E"), 2) == (
        "E", "T",
        "E", "E", "R", "H", "T"
    )
    assert DFID(("A", (("C", (("E",), "D")), "B")), 5) == (
        "A",
        "B", "A",
        "B", "C", "A",
        "B", "D", "C", "A",
        "B", "D", "E", "C", "A"
    )

# 4) Homer helper methods tests
def test_Homer_helpers():
    # FINAL_STATE
    assert FINAL_STATE((True, True, True, True)) is True
    assert FINAL_STATE((False, False, False, False)) is False

    # NEXT_STATE from initial state
    S0 = (False, False, False, False)

    # Only valid move from start is taking baby
    assert NEXT_STATE(S0, "h") == []
    assert NEXT_STATE(S0, "b") == [(True, True, False, False)]
    assert NEXT_STATE(S0, "d") == []
    assert NEXT_STATE(S0, "p") == []

    # SUCC_FN from start
    assert SUCC_FN(S0) == [(True, True, False, False)]

    # ON_PATH
    path = [(False, False, False, False), (True, True, False, False)]
    assert ON_PATH((True, True, False, False), path) is True
    assert ON_PATH((True, False, True, False), path) is False

# 5) Homer solver tests
def test_Homer_solver():
    sol = DFS_SOL((False, False, False, False), [])

    # Solution exists
    assert sol != []

    # Correct start and goal
    assert sol[0] == (False, False, False, False)
    assert sol[-1] == (True, True, True, True)

    # Every step must be legal
    for i in range(len(sol) - 1):
        s = sol[i]
        s2 = sol[i + 1]
        assert any(NEXT_STATE(s, a) == [s2] for a in ["h", "b", "d", "p"]), (s, s2)


if __name__ == "__main__":
    test_BFS()
    print("All BFS tests passed!")
    test_DFS()
    print("All DFS tests passed!")
    test_DFID()
    print("All DFID tests passed!")
    test_Homer_helpers()
    print("All Homer helper tests passed!")
    test_Homer_solver()
    print("All Homer solver tests passed!")

    print("All tests passed")