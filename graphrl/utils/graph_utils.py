

def calculate_node_degree(adj_matrix, node):
    degree = 0
    for neighbour in adj_matrix[node]:
        degree += int(adj_matrix[node][neighbour] != 0) 
    return degree