def PAD(N: int) -> int:
    if N <= 2:
        return 1
    pad = [1, 1, 1]
    for i in range(3, N + 1):
        pad.append(pad[i - 2] + pad[i - 3])
    return pad[N]

def SUMS(N: int) -> int:
    if N <= 2:
        return 0
    return SUMS(N - 2) + SUMS(N - 3) + 1

def ANON(TREE):
    if type(TREE) is not tuple:
         return '?'
    output = []
    for subtree in TREE:
        output.append(ANON(subtree))
    return tuple(output)

def TREE_HEIGHT(TREE):
    if type(TREE) is not tuple:
        return 0 
    heights = [TREE_HEIGHT(subtree) for subtree in TREE]
    return 1 + max(heights, default=0)

def TREE_ORDER(TREE):
    if type(TREE) is not tuple:
        return (TREE,)
    L, m, R = TREE
    return TREE_ORDER(L) + TREE_ORDER(R) + (m,)

def main():
    N = int(input("Enter a positive integer N: "))
    print(f"PAD({N}) = {PAD(N)}")
    print(f"SUMS({N}) = {SUMS(N)}")
    tree_input = eval(input("Enter a nested tuple representing a tree: "))
    print(ANON(tree_input))
    print(TREE_HEIGHT(tree_input))
    print(TREE_ORDER(tree_input))

if __name__ == "__main__":
    main() 