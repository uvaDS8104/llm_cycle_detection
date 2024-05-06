import openai
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
import random
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

model_list = ["text-davinci-003"]
parser = argparse.ArgumentParser(description="clustering_coefficient")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--mode', type=str, default="easy", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temperature (default: 0)')
parser.add_argument('--token', type=int, default=400, help='max token (default: 400)')
parser.add_argument('--SC', type=int, default=0, help='self-consistency (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='number of cases for self-consistency (default: 5)')
args = parser.parse_args()
assert args.prompt in ["CoT", "none", "PROGRAM","k-shot","Instruct","Algorithm","hard-CoT"]

def translate(edge, n, args):
    Q = ''
    if args.prompt in ["CoT", "k-shot", "Instruct", "Algorithm", "hard-CoT"]:
        with open("prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = Q + "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\nThe nodes are numbered from 0 to " + str(n-1)+", and the edges are:"
    for i in range(len(edge)):
        Q = Q + ' ('+str(edge[i][0])+','+str(edge[i][1])+')'
    if args.prompt == "Instruct":
        Q = Q + ". Let's construct a graph with the nodes and edges first."
    Q = Q + "\n"
    target_node = random.randint(0, n-1)
    Q = Q + "Q: What is the clustering coefficient of node " + str(target_node) + " in this graph?\nA:"
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's think step by step:"
        case "LTM":
            Q = Q + " Let's break down this problem:"
        case "PROGRAM":
            Q = Q + " Let's solve the problem by a Python program:"

    return Q, target_node

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
def predict(Q, args):
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7
    if 'gpt' in args.model:
        Answer_list = []
        for text in input:
            response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=args.token,
            )
            Answer_list.append(response["choices"][0]["message"]["content"])
        return Answer_list
    response = openai.Completion.create(
    model=args.model,
    prompt=input,
    temperature=temperature,
    max_tokens=args.token,
    )
    Answer_list = []
    for i in range(len(input)):
        Answer_list.append(response["choices"][i]["text"])
    return Answer_list

def log(Q, res, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = 'log/clustering_coefficient/'+args.model+'-'+args.mode+'-'+time+ '-' + args.prompt
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath+"res.npy", res)
    np.save(newpath+"answer.npy", answer)
    with open(newpath+"prompt.txt","w") as f:
        f.write(Q)
        f.write("\n")
        f.write("Acc: " + str(res.sum())+'/'+str(len(res)) + '\n')
        print(args, file=f)
        
def runtime_analysis(args):
    runtimes = {"easy": [], "medium": [], "hard": []}
    for mode in ["easy", "medium", "hard"]:
        args.mode = mode
        match args.mode:
            case "easy":
                g_num = 150
            case "medium":
                g_num = 600
            case "hard":
                g_num = 400
        
        for j in range(g_num):
            with open(f"data/clustering_coefficient/{args.mode}/standard/{j}.txt", "r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f:
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q, target_node = translate(edge, n, args)
                start_time = time.time()
                predict([Q], args)
                end_time = time.time()
                runtime = end_time - start_time
                runtimes[mode].append(runtime)
    
    for mode in runtimes:
        print(f"Average runtime for {mode} graphs: {np.mean(runtimes[mode]):.2f} seconds")
        
def paraphrase_graph_question(edge, n, args):
    Q = ''
    if args.prompt in ["CoT", "k-shot", "Instruct", "Algorithm", "hard-CoT"]:
        with open("prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = Q + "Consider an undirected graph where an edge between nodes i and j is represented as (i,j).\nThe graph has nodes labeled from 0 to " + str(n-1) + ", and the following edges:"
    random.shuffle(edge)
    for i in range(len(edge)):
        Q = Q + ' (' + str(edge[i][0]) + ',' + str(edge[i][1]) + ')'
    if args.prompt == "Instruct":
        Q = Q + ". As a first step, let's build the graph using the provided nodes and edges."
    Q = Q + "\n"
    target_node = random.randint(0, n-1)
    Q = Q + "Q: Calculate the clustering coefficient for node " + str(target_node) + " in this graph.\nA:"
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's approach this step-by-step:"
        case "LTM":
            Q = Q + " To solve this, let's break it down:"
        case "PROGRAM":
            Q = Q + " Here's a Python program to solve the problem:"

    return Q, target_node

def robustness_analysis(args):
    res_original, res_paraphrased = [], []
    g_num = 150
    
    for j in range(g_num):
        with open(f"data/clustering_coefficient/easy/standard/{j}.txt", "r") as f:
            n, m = [int(x) for x in next(f).split()]
            edge = []
            for line in f:
                edge.append([int(x) for x in line.split()])
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for k in range(m):
                G.add_edge(edge[k][0], edge[k][1])
            
            Q_original, target_node_original = translate(edge, n, args)
            Q_paraphrased, target_node_paraphrased = paraphrase_graph_question(edge, n, args)
            
            original_answer = predict([Q_original], args)[0]
            paraphrased_answer = predict([Q_paraphrased], args)[0]
            
            true_cc_original = nx.clustering(G, target_node_original)
            true_cc_paraphrased = nx.clustering(G, target_node_paraphrased)
            
            try:
                pred_cc_original = float(original_answer)
                pred_cc_paraphrased = float(paraphrased_answer)
                if abs(pred_cc_original - true_cc_original) < 1e-6:
                    res_original.append(1)
                else:
                    res_original.append(0)
                if abs(pred_cc_paraphrased - true_cc_paraphrased) < 1e-6:
                    res_paraphrased.append(1)
                else:
                    res_paraphrased.append(0)
            except ValueError:
                res_original.append(0)
                res_paraphrased.append(0)
    
    res_original = np.array(res_original)
    res_paraphrased = np.array(res_paraphrased)
    print(f"Accuracy on original questions: {res_original.mean():.2f}")
    print(f"Accuracy on paraphrased questions: {res_paraphrased.mean():.2f}")
    
def classical_clustering_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    num_neighbors = len(neighbors)
    if num_neighbors <= 1:
        return 0
    num_edges = 0
    for i in range(num_neighbors):
        for j in range(i+1, num_neighbors):
            if G.has_edge(neighbors[i], neighbors[j]):
                num_edges += 1
    possible_edges = num_neighbors * (num_neighbors - 1) / 2
    return num_edges / possible_edges



def main():
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res, answer = [], []
    match args.mode:
        case "easy":
            g_num = 150
        case "medium":
            g_num = 600
        case "hard":
            g_num = 400

    batch_num = 20
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, target_nodes = [], [], []
        for j in range(i*batch_num, min(g_num, (i+1)*batch_num)):
            with open("data/clustering_coefficient/"+args.mode+"/standard/"+str(j)+".txt","r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f: # read rest of lines
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q, target_node = translate(edge, n, args)
                Q_list.append(Q)
                G_list.append(G)
                target_nodes.append(target_node)
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote = 0
            for k in range(sc):
                ans, G = sc_list[k][j].lower(), G_list[j]
                answer.append(ans.lower())
                target_node = target_nodes[j]
                true_cc = nx.clustering(G, target_node)
                try:
                    pred_cc = float(ans)
                    if abs(pred_cc - true_cc) < 1e-6:
                        vote += 1
                except ValueError:
                    pass
            if vote * 2 >= sc:
                res.append(1)
            else:
                res.append(0)    
            
    res = np.array(res)
    answer = np.array(answer)
    log(Q, res, answer, args)
    print(res.sum())

def main():
    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res, answer = [], []
    match args.mode:
        case "easy":
            g_num = 150
        case "medium":
            g_num = 600
        case "hard":
            g_num = 400

    batch_num = 20
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        G_list, Q_list, target_nodes = [], [], []
        for j in range(i*batch_num, min(g_num, (i+1)*batch_num)):
            with open(f"data/clustering_coefficient/{args.mode}/standard/{j}.txt", "r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f:
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q, target_node = translate(edge, n, args)
                Q_list.append(Q)
                G_list.append(G)
                target_nodes.append(target_node)
        sc = 1
        if args.SC == 1:
            sc = args.SC_num
        sc_list = []
        for k in range(sc):
            answer_list = predict(Q_list, args)
            sc_list.append(answer_list)
        for j in range(len(Q_list)):
            vote = 0
            for k in range(sc):
                ans, G = sc_list[k][j].lower(), G_list[j]
                answer.append(ans.lower())
                target_node = target_nodes[j]
                true_cc = nx.clustering(G, target_node)
                classical_cc = classical_clustering_coefficient(G, target_node)
                try:
                    pred_cc = float(ans)
                    if abs(pred_cc - true_cc) < 1e-6:
                        vote += 1
                except ValueError:
                    pass
            if vote * 2 >= sc:
                res.append(1)
            else:
                res.append(0)    
            
    res = np.array(res)
    answer = np.array(answer)
    log(Q, res, answer, args)
    print(f"Accuracy: {res.sum()}/{len(res)}")
    
    # Runtime analysis
    print("Running runtime analysis...")
    runtime_analysis(args)
    
    # Robustness analysis
    print("Running robustness analysis...")
    robustness_analysis(args)

if __name__ == "__main__":
    main()