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
parser = argparse.ArgumentParser(description="cycle")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--mode', type=str, default="easy", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temprature (default: 0)')
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
    #character = [chr(65+i) for i in range(26)] + [chr(65+i)+chr(65+i) for i in range(26)]
    for i in range(len(edge)):
        Q = Q + ' ('+str(edge[i][0])+','+str(edge[i][1])+')'
    if args.prompt == "Instruct":
        Q = Q + ". Let's construct a graph with the nodes and edges first."
    Q = Q + "\n"
    if args.prompt == "Recitation":
        Q = Q + "Q1: Are node "+str(edge[0][0])+" and node " +str(edge[0][1])+" connected with an edge?\nA1: Yes.\n"
        u = -1
        for i in range(n):
            if u != -1:
                break
            for j in range(n):
                if [i,j] not in edge:
                    u, v = i, j
                    break
        Q = Q + "Q2: Are node "+str(u)+" and node " +str(v)+" connected with an edge?\nA2: No.\n"
    Q = Q + "Q: Is there a cycle in this graph?\nA:"
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's think step by step:"
        case "LTM":
            Q = Q + " Let's break down this problem:"
        case "PROGRAM":
            Q = Q + " Let's solve the problem by a Python program:"

    return Q

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
    newpath = 'log/cycle/'+args.model+'-'+args.mode+'-'+time+ '-' + args.prompt
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
        G_list, Q_list = [], []
        for j in range(i*batch_num, min(g_num, (i+1)*batch_num)):
            with open("data/cycle/graph/"+args.mode+"/standard/graph"+str(j)+".txt","r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f: # read rest of lines
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q = translate(edge, n, args)
                Q_list.append(Q)
                G_list.append(G)
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
                result = 0
                pos = max(ans.find("in this case"), ans.find("after running the algorithm"))
                if pos == -1:
                    pos = 0
                p1 = ans.find("there is no cycle")  # for codex
                p2 = ans.find("there is a cycle")  # for codex
                p1 = 1000000 if p1 == -1 else p1
                p2 = 1000000 if p2 == -1 else p2
                idx = i * batch_num + j
                if (idx*2 < g_num and p1 < p2) or (idx*2 > g_num and p2 < p1):
                    vote += 1
            if vote * 2 >= sc:
                res.append(1)
            else:
                res.append(0)    
            
    res = np.array(res)
    answer = np.array(answer)
    log(Q, res, answer, args)
    print(res.sum())
    
    # Runtime analysis
    print("Running runtime analysis...")
    runtime_analysis(args)
    
    # Robustness check
    print("Running robustness check...")
    robustness_check(args)
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
            with open(f"data/cycle/graph/{args.mode}/standard/graph{j}.txt", "r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f:
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q = translate(edge, n, args)
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
        Q = Q + ". As a first step, let's construct the graph using the provided nodes and edges."
    Q = Q + "\n"
    Q = Q + "Q: Does this graph contain any cycles?\nA:"
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's approach this step-by-step:"
        case "LTM":
            Q = Q + " To solve this, let's break it down:"
        case "PROGRAM":
            Q = Q + " Here's a Python program to solve the problem:"

    return Q

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
        Q = Q + ". As a first step, let's construct the graph using the provided nodes and edges."
    Q = Q + "\n"
    Q = Q + "Q: Does this graph contain any cycles?\nA:"
    match args.prompt:
        case "0-CoT":
            Q = Q + " Let's approach this step-by-step:"
        case "LTM":
            Q = Q + " To solve this, let's break it down:"
        case "PROGRAM":
            Q = Q + " Here's a Python program to solve the problem:"

    return Q

def robustness_check(args):
    res_original, res_paraphrased = [], []
    g_num = 150
    
    for j in range(g_num):
        with open(f"data/cycle/graph/easy/standard/graph{j}.txt", "r") as f:
            n, m = [int(x) for x in next(f).split()]
            edge = []
            for line in f:
                edge.append([int(x) for x in line.split()])
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for k in range(m):
                G.add_edge(edge[k][0], edge[k][1])
            
            Q_original = translate(edge, n, args)
            Q_paraphrased = paraphrase_graph_question(edge, n, args)
            
            original_answer = predict([Q_original], args)[0]
            paraphrased_answer = predict([Q_paraphrased], args)[0]
            
            result_original = 0
            result_paraphrased = 0
            
            if "there is no cycle" in original_answer.lower():
                result_original = 1 if j < g_num // 2 else 0
            elif "there is a cycle" in original_answer.lower():
                result_original = 0 if j < g_num // 2 else 1
            
            if "there is no cycle" in paraphrased_answer.lower():
                result_paraphrased = 1 if j < g_num // 2 else 0
            elif "there is a cycle" in paraphrased_answer.lower():
                result_paraphrased = 0 if j < g_num // 2 else 1
            
            res_original.append(result_original)
            res_paraphrased.append(result_paraphrased)
    
    res_original = np.array(res_original)
    res_paraphrased = np.array(res_paraphrased)
    print(f"Accuracy on original questions: {res_original.mean():.2f}")
    print(f"Accuracy on paraphrased questions: {res_paraphrased.mean():.2f}")
            
            
if __name__ == "__main__":
    main()