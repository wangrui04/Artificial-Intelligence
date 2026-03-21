from hw2-skeleton import *

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
