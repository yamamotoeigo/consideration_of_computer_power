#!/usr/bin/env python
# coding: utf-8

# In[1044]:


import networkx as nx
import random
import block_division_func as bd
import copy


# In[1045]:


def get_random_nodes(graph, num_pcs: int) -> list:
    """ネットワークトポロジーからランダムにノードをn個選択する

    Args:
        graph (_type_): トポロジ
        node_num (int): 取得するノードの個数

    Returns:
        list: 取得したノード番号のリスト
    """    
    random_nodes = random.sample(list(graph.nodes()), num_pcs)
    random_nodes = [int(node) for node in random_nodes]
    return random_nodes


# In[1046]:


def generate_random_numbers(n: int, total: int) -> list:
    """計算機にキャパシティを付与

    Args:
        n (int): _description_
        total (int): _description_

    Returns:
        list: _description_
    """    
    numbers = [0] * n

    # Generate n-1 random numbers
    for i in range(n-1):
        num = random.randint(1, total//(10^(len(str(total)))))
        numbers[i] = num

    # Calculate the remaining value needed to reach the total
    if total < sum(numbers):
        remaining = abs(total - sum(numbers))
        for i in range(n):
            random_node = random.randint(0, n-1)
            numbers[random_node] += remaining // random.randint(total//(10^(len(str(total)))), total//(10^(len(str(total)))-1))
    else:
        remaining = random.randint(1, total)

    # Add the remaining value as the last number
    numbers[-1] = (remaining//10)

    return numbers


# In[1047]:


def generate_exchange_matrix(assigned_matrix: list, num_pcs: int, G) -> list:
    exchange_matrix = [[0] * num_pcs for _ in range(num_pcs)]
    
    for i in range(num_pcs):
        for j in range(num_pcs):
            if i == j:
                continue
            exchange_matrix[i][j] = nx.dijkstra_path_length(G, str(assigned_matrix[i]), str(assigned_matrix[j]))
            exchange_matrix[j][i] = nx.dijkstra_path_length(G, str(assigned_matrix[i]), str(assigned_matrix[j]))
            
    return exchange_matrix


# In[1048]:


def get_top_n_indices(lst: list, n: int) -> list:
    """ノードの中でキャパが上位n個の要素のインデックスを取得する

    Args:
        lst (list): 探索対象のリスト
        n (int): 何個取るか

    Returns:
        list: _description_
    """    
    # Use enumerate() and sorted() to get the indices of the top n elements
    indices = [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:n]]
    return indices


# In[1049]:


def convert_node_number(original_nodes: list, node_list: list) -> list:
    """ノード番号を文字列から数字に変換する

    Args:
        node_list (list): ノード番号のリスト

    Returns:
        list: ノード番号のリスト
    """    
    node_list = [int(original_nodes[node]) for node in node_list]
    return node_list


# In[1050]:


def evaluate_communication_cost(block_list, assigned_matrix, converter, linked_blocks, G):
    """割り当て結果から順伝播させた時の通信コストを計算する

    Args:
        block_list (list): ブロック分割のリスト
        assigned_matrix (list): ニューロンに計算機を割り当てたリスト
        linked_blocks (list): ブロックの接続関係を表すリスト

    Returns:
        float: 総通信コスト
    """    
    total_cost = 0
    # Process each block
    for temp_label in block_list:
        # Get the computer (PC) assigned to the block
        initial_pc = bd.get_pc_by_block_label(temp_label, assigned_matrix)
        
        # Get the list of labels of other blocks connected to the current block
        linked_list = linked_blocks[temp_label]

        # Get the list of computers (PCs) assigned to the connected blocks
        target_pcs = [bd.get_pc_by_block_label(linked_block, assigned_matrix) for linked_block in linked_list]

        # Remove duplicate computer numbers
        target_pcs = list(set(target_pcs))
        
        # Calculate the total communication cost
        temp_cost = 0
        # print(target_pcs_convert)
        for target in target_pcs:
            if target is not None:
                # target_pc = converted_assigned_matrix[target]
                length = nx.dijkstra_path_length(G, str(converter[initial_pc]), str(converter[target]))
                # print(f'inital: {initial_pc}, target: {target}, length: {length}')
                # print(f'inital: {converter[initial_pc]}, target: {converter[target]}, length: {length}\n')
                temp_cost += length

        total_cost += temp_cost

    return total_cost


# In[1051]:


def generate_commcost_and_slots_greedy_assignment(num_pcs: int, num_blocks: int, capacities: list, exchange_matrix: list) -> list:
    """通信コストとスロット数を考慮してグリーディーに割り当てを実施する

    Args:
        num_pcs (int): 計算機の第数
        num_blocks (int): ブロックの数
        capacities (list): 計算機のキャかシティ
        exchange_matrix (list): 交流行列

    Returns:
        list: 割り当て結果
    """    
    assignment = [-1] * num_blocks  # 割り当てを表すリスト
    used_pcs = []  # 割り当て済み(使用済み)の計算機のリスト

    # 1. キャパの一番大きいPCを探す
    max_capacity = max(capacities)
    count = 0  # 割り当て済みのブロック数
    
    initial_pc = capacities.index(max_capacity)
    used_pcs.append(initial_pc)  # 最大キャパシティの計算機を使用済みリストに追加
    # print(f'最大キャパシティのPC: {initial_pc}')
    # print(max_capacity)
    
    # 2. (1)のPCを始点としてキャパ分のブロックを割り当てる
    for i in range(max_capacity):
        if count < num_blocks:
            assignment[i] = initial_pc
            count += 1
    

    # 3. 4. をすべてのブロックをなくなるまで繰り返す
    while count < num_blocks:
        # next_pc = bd.get_positive_min(exchange_matrix[initial_pc], used_pcs=used_pcs)  # 通信コストが低いPC
        # print(next_pc)
        next_pc = None
        max_value = -1

        # next_pc は, キャパシティを通信コストで割った値が最大となるPCを探索する
        for pc in range(num_pcs):
            if pc not in used_pcs:
                communication_cost = exchange_matrix[initial_pc][pc]
                if communication_cost != 0:
                    value = capacities[pc] / communication_cost
                    if value > max_value:
                        max_value = value
                        next_pc = pc
                        
        if next_pc is None:
            # 計算機の探索が不可能な場合、未使用のPCの中で最大の容量を持つPCを現在のPCとする
            remaining_pcs = set(range(num_pcs)) - set(used_pcs)
            next_pc = max(remaining_pcs, key=lambda pc: capacities[pc])

        used_pcs.append(next_pc)  # 使用済みリストに追加

        # 4. (3)のPCにブロックをキャパ分、割り当てる
        for i in range(capacities[next_pc]):
            if count >= num_blocks:
                break
            assignment[count] = next_pc
            count += 1

    return assignment


def generate_comm_cost_greedy_assignment(num_pcs: int, num_blocks: int, capacities: list, exchange_matrix: list) -> list:
    """通信コストだけを見てグリーディーに割り当てを実施する

    Args:
        num_pcs (int): 計算機の第数
        num_blocks (int): ブロックの数
        capacities (list): 計算機のキャかシティ
        exchange_matrix (list): 交流行列

    Returns:
        list: 割り当て結果
    """    
    assignment = [-1] * num_blocks  # 割り当てを表すリスト
    used_pcs = []  # 割り当て済み(使用済み)の計算機のリスト

    # 1. キャパの一番大きいPCを探す
    max_capacity = max(capacities)
    count = 0  # 割り当て済みのブロック数
    
    initial_pc = capacities.index(max_capacity)
    used_pcs.append(initial_pc)  # 最大キャパシティの計算機を使用済みリストに追加
    # print(f'最大キャパシティのPC: {initial_pc}')
    # print(max_capacity)
    
    # 2. (1)のPCを始点としてキャパ分のブロックを割り当てる
    for i in range(max_capacity):
        if count < num_blocks:
            assignment[i] = initial_pc
            count += 1
    

    # 3. 4. をすべてのブロックをなくなるまで繰り返す
    while count < num_blocks:
        next_pc = bd.get_positive_min(exchange_matrix[initial_pc], used_pcs=used_pcs)  # 通信コストが低いPC
        # print(next_pc)
        # next_pc = None
        # max_value = -1

        # next_pc は, キャパシティを通信コストで割った値が最大となるPCを探索する
        # for pc in range(num_pcs):
        #     if pc not in used_pcs:
        #         communication_cost = exchange_matrix[initial_pc][pc]
        #         if communication_cost != 0:
        #             value = capacities[pc] / communication_cost
        #             if value > max_value:
        #                 max_value = value
        #                 next_pc = pc
                        
        if next_pc is None:
            # 計算機の探索が不可能な場合、未使用のPCの中で最大の容量を持つPCを現在のPCとする
            remaining_pcs = set(range(num_pcs)) - set(used_pcs)
            next_pc = max(remaining_pcs, key=lambda pc: capacities[pc])

        used_pcs.append(next_pc)  # 使用済みリストに追加

        # 4. (3)のPCにブロックをキャパ分、割り当てる
        for i in range(capacities[next_pc]):
            if count >= num_blocks:
                break
            assignment[count] = next_pc
            count += 1

    return assignment


def generate_slots_greedy_assignment(num_pcs: int, num_blocks: int, capacities: list, exchange_matrix: list) -> list:
    """キャパシティだけを考えて割り当てを実施する

    Args:
        num_pcs (int): 計算機の第数
        num_blocks (int): ブロックの数
        capacities (list): 計算機のキャかシティ
        exchange_matrix (list): 交流行列

    Returns:
        list: 割り当て結果
    """    
    assignment = [-1] * num_blocks  # 割り当てを表すリスト
    used_pcs = []  # 割り当て済み(使用済み)の計算機のリスト

    # 1. キャパの一番大きいPCを探す
    max_capacity = max(capacities)
    count = 0  # 割り当て済みのブロック数
    
    initial_pc = capacities.index(max_capacity)
    used_pcs.append(initial_pc)  # 最大キャパシティの計算機を使用済みリストに追加
    # print(f'最大キャパシティのPC: {initial_pc}')
    # print(max_capacity)
    
    # 2. (1)のPCを始点としてキャパ分のブロックを割り当てる
    for i in range(max_capacity):
        if count < num_blocks:
            assignment[i] = initial_pc
            count += 1
    

    # 3. 4. をすべてのブロックをなくなるまで繰り返す
    while count < num_blocks:
        next_pc = None
        max_capacity = -1

        for pc in range(num_pcs):
            if pc not in used_pcs:
                capacity = capacities[pc]
                if capacity > max_capacity:
                    max_capacity = capacity
                    next_pc = pc
        
        if next_pc is None:
            # 計算機の探索が不可能な場合、未使用のPCの中で最大の容量を持つPCを現在のPCとする
            remaining_pcs = set(range(num_pcs)) - set(used_pcs)
            next_pc = max(remaining_pcs, key=lambda pc: capacities[pc])

        used_pcs.append(next_pc)  # 使用済みリストに追加

        # 4. (3)のPCにブロックをキャパ分、割り当てる
        for i in range(capacities[next_pc]):
            if count >= num_blocks:
                break
            assignment[count] = next_pc
            count += 1

    return assignment


# In[1052]:


# def experiment(num_blocks):
    # 実験環境
    # GraphML形式のグラフを読み込む
    G = nx.read_edgelist("/home/yamamoto/research/consideration_of_computer_power/src/data/japanese_network.edgelist", data=False)

    # ノード番号を文字列から数字に変換する
    # mapping = {node: i for i, node in enumerate(G.nodes())}
    # G = nx.relabel_nodes(G, mapping)

    pos = nx.spring_layout(G)
    # nx.draw_networkx(G, pos)

    # 分散処理に使用するPCPCの数
    num_pcs = 10
    # 分割するブロックの数
    num_blocks_total = 15000
    # 分割するブロックの数(実験で使用する)
    num_blocks = num_blocks

    # ニューラルネットワークの分割ブロックの構造を表す
    structure_row = 4 # ブロックの行数
    structure_col = num_blocks_total // structure_row # ブロックの列数

    # トポロジにある計算機の数
    num_nodes = G.number_of_nodes()
    # トポロジにある計算機にキャパシティを割り当てる
    capacities = generate_random_numbers(num_nodes, num_blocks_total)

    # ランダムにノードを {num_pcs}個取得
    random_nodes = get_random_nodes(graph=G, num_pcs=num_pcs)
    # ランダムノードのキャパシティ
    random_capacities = [capacities[node] for node in random_nodes]

    # 計算機群からキャパシティ上位{num_pcs}個の計算機を取得
    top_indices = get_top_n_indices(capacities, num_pcs)
    # 上位ノードのキャパシティ
    top_capacities = [capacities[node] for node in top_indices]
    # 上位ノードの交流行列を作成
    exchange_matrix = generate_exchange_matrix(top_indices, num_pcs, G)

    # ニューラルネットワークを分割したブロック
    block_structure = bd.generate_block_structure(row=structure_row, col=structure_col)

    # 通信する必要のあるブロックのリストを作成(ブロックの接続関係)
    linked_blocks = bd.generate_linked_block_list(block_structure)

    # 分割ブロックの番号リスト
    block_list = bd.generate_block_list(num_blocks=num_blocks)

    # ランダムに割り当てる
    random_assigned_matrix = bd.generate_random_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=random_capacities)
    random_random_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=random_assigned_matrix, converter=random_nodes, G=G)

    # ランダムにノードを選択し、グリーディーに割り当て
    random_exchange_matrix = generate_exchange_matrix(random_nodes, num_pcs, G)
    random_greedy_assigned_matrix = generate_greedy_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=random_capacities, exchange_matrix=random_exchange_matrix)
    random_greedy_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=random_greedy_assigned_matrix, converter=random_nodes, G=G)
    
    # 上位のノードを選択し、ランダムに割り当てる
    top_random_assigned_matrix = bd.generate_random_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=top_capacities)
    top_random_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=top_random_assigned_matrix, converter=top_indices, G=G)

    # 上位のノードを選択しグリーディーに割り当てる
    greedy_assigned_matrix = generate_greedy_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=top_capacities, exchange_matrix=exchange_matrix)
    top_greedy_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=greedy_assigned_matrix, converter=top_indices, G=G)

    greedy_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=greedy_assigned_matrix, converter=top_indices, G=G)
    print(f'ランダムにノードを選択しランダムに割り当てた場合の総通信コスト: {random_random_total_cost}')
    print(f'ランダムにノードを選択し提案手法で割り当てた場合の総通信コスト: {random_greedy_total_cost}')
    print(f'上位のノードを選択しランダムに割り当てた場合の総通信コスト: {top_random_total_cost}')
    print(f'上位のノードを選択し提案手法で割り当てた場合の総通信コスト: {top_greedy_total_cost}\n')
    
    return random_random_total_cost, random_greedy_total_cost, top_random_total_cost, top_greedy_total_cost


# In[1053]:


# # 実験環境
# # GraphML形式のグラフを読み込む
# G = nx.read_edgelist("/home/yamamoto/research/consideration_of_computer_power/src/data/japanese_network.edgelist", data=False)

# # ノード番号を文字列から数字に変換する
# # mapping = {node: i for i, node in enumerate(G.nodes())}
# # G = nx.relabel_nodes(G, mapping)

# pos = nx.spring_layout(G)
# # nx.draw_networkx(G, pos)

# # 分散処理に使用するPCPCの数
# num_pcs = 10
# # 分割するブロックの数
# num_blocks_total = 15000
# # 分割するブロックの数(実験で使用する)
# num_blocks = 100

# # ニューラルネットワークの分割ブロックの構造を表す
# structure_row = 2 # ブロックの行数
# structure_col = num_blocks_total // structure_row # ブロックの列数

# # トポロジにある計算機の数
# num_nodes = G.number_of_nodes()
# # トポロジにある計算機にキャパシティを割り当てる
# capacities = generate_random_numbers(num_nodes, num_blocks_total)

# # ランダムにノードを {num_pcs}個取得
# random_nodes = get_random_nodes(graph=G, num_pcs=num_pcs)
# # ランダムノードのキャパシティ
# random_capacities = [capacities[node] for node in random_nodes]

# # 計算機群からキャパシティ上位{num_pcs}個の計算機を取得
# top_indices = get_top_n_indices(capacities, num_pcs)
# # 上位ノードのキャパシティ
# top_capacities = [capacities[node] for node in top_indices]
# # 上位ノードの交流行列を作成
# exchange_matrix = generate_exchange_matrix(top_indices, num_pcs, G)

# # ニューラルネットワークを分割したブロック
# block_structure = bd.generate_block_structure(row=structure_row, col=structure_col)

# # 通信する必要のあるブロックのリストを作成(ブロックの接続関係)
# linked_blocks = bd.generate_linked_block_list(block_structure)

# # 分割ブロックの番号リスト
# block_list = bd.generate_block_list(num_blocks=num_blocks)

# # ランダムに割り当てる
# random_assigned_matrix = bd.generate_random_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=random_capacities)
# random_assigned_matrix_converted = convert_node_number(random_nodes, random_assigned_matrix)

# print(f'計算機のキャパシティ:\n{capacities}')
# print(f'目標値: {num_blocks}, 結果: {sum(capacities)}\n')

# print(f'ランダムに選択したノード: {random_nodes}')
# print(f'ランダムに選んだノードのキャパシティ: {random_capacities}(合計値: {sum(random_capacities)})\n')

# print(f'キャパシティ上位{num_pcs}個のノード: {top_indices}')
# print(f'キャパシティ上位{num_pcs}個のノードのキャパシティ: {top_capacities}(合計値: {sum(top_capacities)})\n')
# print("交流行列")
# for row in exchange_matrix:
#     print(row)
# print(f'\n分割ブロックの番号リスト{block_list}\n')


# In[1054]:


# # ランダムに割り当てる
# random_assigned_matrix = bd.generate_random_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=random_capacities)
# random_assigned_matrix_converted = convert_node_number(random_nodes, random_assigned_matrix)

# # ランダムにノードを選択し、グリーディーに割り当て
# random_exchange_matrix = generate_exchange_matrix(random_nodes, num_pcs, G)
# random_greedy_assigned_matrix = generate_greedy_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=random_capacities, exchange_matrix=random_exchange_matrix)
# random_greedy_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=random_greedy_assigned_matrix, converter=random_nodes, G=G)


# print("\nPCが担当するブロックの割り当てを表す行列")
# print(f'Group_random = {random_assigned_matrix}')
# print(f'変換後: {random_assigned_matrix_converted}\n')

# random_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=random_assigned_matrix, converter=random_nodes, G=G)
# print(f'ランダムに割り当て:\n{random_assigned_matrix}')
# print(f'ランダムに割り当てた場合の総通信コスト: {random_total_cost}\n')


# # In[1055]:


# # グリーディーに割り当てる
# greedy_assigned_matrix = generate_greedy_assignment(num_pcs=num_pcs, num_blocks=num_blocks, capacities=top_capacities, exchange_matrix=exchange_matrix)
# greedy_assigned_matrix_converted = convert_node_number(top_indices, greedy_assigned_matrix)


# greedy_total_cost = evaluate_communication_cost(block_list=block_list, linked_blocks=linked_blocks, assigned_matrix=greedy_assigned_matrix, converter=top_indices, G=G)
# print(f'ランダムにノードを選択しランダムに割り当てた場合の総通信コスト: {random_total_cost}\n')
# print(f'ランダムにノードを選択し提案手法で割り当てた場合の総通信コスト: {random_greedy_total_cost}\n')
# print(f'上位のノードを選択し提案手法で割り当てた場合の総通信コスト: {greedy_total_cost}\n')

