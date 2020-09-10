
DEBUG = False

def merge(match_scheme, sum_match_scheme):
    # print(match_scheme, sum_match_scheme)
    for match_item, sum_match_item in zip(match_scheme, sum_match_scheme):
        for node in match_item:
            if node not in sum_match_item:
                sum_match_item.append(node)
    return sum_match_scheme

def bfs(node1, node2, graph1, graph2):
    if DEBUG:
        print(node1, node2)
    adj_nodes1 = graph1[node1]
    adj_nodes2 = graph2[node2]
    if len(adj_nodes2) > len(adj_nodes1) or not adj_nodes2 and adj_nodes1:
        return None
    matchs1 = [False for _ in range(len(adj_nodes1))]
    sum_match_scheme = [[] for _ in range(len(graph1))]
    for node2_id in range(len(adj_nodes2)):
        for node1_id in range(len(adj_nodes1)):
            match_scheme = bfs(adj_nodes1[node1_id], adj_nodes2[node2_id], graph1, graph2)
            if not matchs1[node1_id] and match_scheme:
                matchs1[node1_id] = True
                if DEBUG:
                    print("match {} {}".format(adj_nodes1[node1_id], adj_nodes2[node2_id]))
                    print("match_scheme", match_scheme)
                break
        if not match_scheme:
            return None
        else:
            sum_match_scheme = merge(match_scheme, sum_match_scheme)
    matchs1_id_list = []
    for i in range(len(matchs1)):
        if matchs1[i]:
            matchs1_id_list.append(adj_nodes1[i])
    sum_match_scheme[node1] = matchs1_id_list
    if DEBUG:
        print("sum_match_scheme", sum_match_scheme)
    return sum_match_scheme

def minus(graph, sub_graph):
    result = []
    for a, b in zip(graph, sub_graph):
        result.append(list(set(a)-set(b)))
    return result

def subtraction(graph1, graph2):
    # graph1(DAG), graph2(DAG)
    # graph1 = graph2 + some edges,
    # but their encode way maybe not same
    # so you must shield their encode way by your code
    # you can assume the first node is root of DAG
    # so you don't have to find the root by your self
    # return: graph1 - graph2 (the additive edges)
    assert len(graph1) == len(graph2), "node_num of two graphs do not equal, but you are {} and {}".format(len(graph1), len(graph2))
    node1, node2 = 0, 0
    match_scheme_in_graph1_pattern = bfs(node1, node2, graph1, graph2)
    additive_edge = minus(graph1, match_scheme_in_graph1_pattern)
    return additive_edge


if __name__ == '__main__':
    # graph2 = [[1], [2, 4, 5], [3], [6], [3], [3], []]
    # graph2 = [[4], [2], [6], [2], [1, 3, 5], [2], []]
    graph2 = [[6], [4], [4], [], [3], [4], [2, 5, 1]]
    graph1 = [[1, 2], [2, 4, 5, 3, 6], [3, 6], [6], [3], [3], []]
    additive_edge = subtraction(graph1, graph2)
    print(additive_edge)
