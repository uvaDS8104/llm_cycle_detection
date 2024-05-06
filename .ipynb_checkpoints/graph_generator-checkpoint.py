import os
from random import randint, shuffle, random
import networkx as nx
import argparse
from tqdm import tqdm

class Generator:
    def __init__(self, num_of_nodes=10, edge_probability=0.35, max_weight=4):
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability
        self.max_weight = max_weight

    def generate_graph(self):
        l = randint(2, 6)
        while True:
            idx = list(range(self.num_of_nodes))
            shuffle(idx)
            G = nx.Graph()
            G.add_nodes_from(range(self.num_of_nodes))
            for u in G.nodes():
                for v in G.nodes():
                    if u < v and random() < self.edge_probability:
                        weight = randint(1, self.max_weight)
                        G.add_edge(idx[u], idx[v], weight=weight)
            if nx.is_connected(G):
                for u in G.nodes():
                    for v in G.nodes():
                        if u != v and not G.has_edge(idx[u], idx[v]) and nx.shortest_path_length(G, source=idx[u], target=idx[v]) >= l:
                            q = [idx[u], idx[v]]
                            return G, q

    def generate(self):
        G, q = self.generate_graph()
        return G, q

def main():
    parser = argparse.ArgumentParser(description="Data generation")
    parser.add_argument('--mode', type=str, default="easy", help='Mode (default: easy)')
    args = parser.parse_args()

    assert args.mode in ["easy", "hard"]

    p_list = [0.5, 0.7, 0.9] if args.mode == "easy" else [0.2, 0.25]
    n_min, n_max = (5, 10) if args.mode == "easy" else (11, 20)
    g_num, max_weight = (10, 4) if args.mode == "easy" else (10, 10)

    newpath = os.path.join('data', args.mode)
    os.makedirs(newpath, exist_ok=True)

    graph_index = 0
    for num in tqdm(range(n_min, n_max + 1)):
        for edge_probability in p_list:
            for _ in range(g_num):
                generator = Generator(num_of_nodes=num, edge_probability=edge_probability, max_weight=max_weight)
                Graph, q = generator.generate()
                edge = list(Graph.edges())
                file_path = os.path.join(newpath, f"graph{graph_index}.txt")
                with open(file_path, "w") as f:
                    f.write(f"{Graph.number_of_nodes()} {Graph.number_of_edges()}\\n")
                    for e in edge:
                        e = (e[0], e[1]) if random() < 0.5 else (e[1], e[0])
                        f.write(f"{e[0]} {e[1]} {Graph[e[0]][e[1]]['weight']}\\n")
                    f.write(f"{q[0]} {q[1]}\\n")
                graph_index += 1

if __name__ == "__main__":
    main()
