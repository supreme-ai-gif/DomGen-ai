"""
neuralic_3_0.py
Neuralic 3.0 — Reasoning Brain AI
Features:
- Embedding-based neurons (concepts as vectors)
- Sparse dynamic graph (connections with weights)
- Memory & reasoning
- Learning from text files or direct input
- JSON persistence
- Console chat + teach + file upload
"""

import json
import os
import time
import math
import random
from collections import defaultdict, deque

# -------------------------
# CONFIG
# -------------------------
STATE_FILE = "neuralic_3_0_state.json"
MAX_NEURONS = 20000
MAX_CONNECTIONS = 50
DECAY_INTERVAL = 200
DECAY_FACTOR = 0.03
AUTOSAVE_INTERVAL = 50
BULK_LEARN_CHUNK_SIZE = 20
EMBED_DIM = 16
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -------------------------
# UTILITIES
# -------------------------
def now_ts():
    return int(time.time())

def normalize_text(s: str):
    s = s.lower()
    s = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in s)
    return ' '.join(s.split())

def tokenize(s: str):
    return [t for t in normalize_text(s).split() if t]

def make_id(s: str):
    return str(abs(hash(normalize_text(s))) % (10**12))

def cosine_sim(vec1, vec2):
    dot = sum(a*b for a,b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(a*a for a in vec2))
    if norm1==0 or norm2==0:
        return 0.0
    return dot/(norm1*norm2)

def random_embedding(dim=EMBED_DIM):
    return [random.uniform(-1,1) for _ in range(dim)]

# -------------------------
# NEURON
# -------------------------
class Neuron:
    __slots__ = ("nid","concept","embedding","connections","seen_count","last_seen")
    def __init__(self, concept):
        self.concept = concept
        self.nid = make_id(concept)
        self.embedding = random_embedding()
        self.connections = {}  # target_nid -> weight
        self.seen_count = 0
        self.last_seen = now_ts()

    def connect_to(self, target_nid, weight=1.0):
        if target_nid == self.nid:
            return
        if target_nid in self.connections:
            self.connections[target_nid] = min(self.connections[target_nid]+weight, 100.0)
        else:
            if len(self.connections) >= MAX_CONNECTIONS:
                # remove smallest weight
                smallest = min(self.connections.items(), key=lambda x:x[1])[0]
                del self.connections[smallest]
            self.connections[target_nid] = weight

    def strengthen(self, target_nid, amount=0.2):
        self.connect_to(target_nid, amount)

    def decay(self, factor):
        for k in list(self.connections.keys()):
            self.connections[k] *= (1-factor)
            if self.connections[k]<0.001:
                del self.connections[k]

    def to_dict(self):
        return {
            "nid": self.nid,
            "concept": self.concept,
            "embedding": self.embedding,
            "connections": self.connections,
            "seen_count": self.seen_count,
            "last_seen": self.last_seen
        }

    @classmethod
    def from_dict(cls,d):
        n = cls(d["concept"])
        n.nid = d["nid"]
        n.embedding = d.get("embedding", random_embedding())
        n.connections = {k: float(v) for k,v in d.get("connections", {}).items()}
        n.seen_count = d.get("seen_count",0)
        n.last_seen = d.get("last_seen", now_ts())
        return n

# -------------------------
# BRAIN
# -------------------------
class Neuralic:
    def __init__(self, state_file=STATE_FILE):
        self.neurons = {}          # nid -> Neuron
        self.index = {}            # normalized_concept -> nid
        self.memory = {}           # input -> output
        self.state_file = state_file
        self.interaction_count = 0
        self._bootstrap_seed()

    # ---------- Persistence ----------
    def save_state(self):
        data = {
            "neurons": {nid:n.to_dict() for nid,n in self.neurons.items()},
            "index": self.index,
            "memory": self.memory,
            "interaction_count": self.interaction_count
        }
        with open(self.state_file,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2,ensure_ascii=False)
        print("[save] state written:", self.state_file)

    def load_state(self):
        if not os.path.exists(self.state_file):
            print("[load] no state file, starting fresh")
            return
        with open(self.state_file,"r",encoding="utf-8") as f:
            data = json.load(f)
        self.neurons = {nid:Neuron.from_dict(nd) for nid,nd in data.get("neurons",{}).items()}
        self.index = data.get("index",{})
        self.memory = data.get("memory",{})
        self.interaction_count = data.get("interaction_count",0)
        print(f"[load] loaded {len(self.neurons)} neurons, {len(self.memory)} memories")

    # ---------- Bootstrap ----------
    def _bootstrap_seed(self):
        seeds = [("hello","Hello! I'm Neuralic 3.0.")]
        for inp,out in seeds:
            self._ensure_neuron(inp)
            self._ensure_neuron(out)
            self.link(inp,out)
            self.memory[inp] = out

    # ---------- Neuron Management ----------
    def _ensure_neuron(self,concept):
        key = normalize_text(concept)
        if key in self.index:
            return self.neurons[self.index[key]]
        if len(self.neurons) >= MAX_NEURONS:
            return random.choice(list(self.neurons.values()))
        n = Neuron(concept)
        if n.nid in self.neurons:
            n.nid = n.nid + "_" + str(random.randint(0,9999))
        self.neurons[n.nid] = n
        self.index[key] = n.nid
        return n

    def link(self, from_concept, to_concept, weight=1.0):
        n1 = self._ensure_neuron(from_concept)
        n2 = self._ensure_neuron(to_concept)
        n1.connect_to(n2.nid,weight)
        n2.connect_to(n1.nid,weight*0.25)

    def strengthen(self, from_concept, to_concept, amount=0.2):
        k1,k2 = normalize_text(from_concept), normalize_text(to_concept)
        if k1 in self.index and k2 in self.index:
            self.neurons[self.index[k1]].strengthen(self.index[k2],amount)
            self.neurons[self.index[k2]].strengthen(self.index[k1],amount*0.5)

    # ---------- Decay ----------
    def maybe_decay(self):
        self.interaction_count += 1
        if self.interaction_count % DECAY_INTERVAL == 0:
            self.apply_decay(DECAY_FACTOR)
        if self.interaction_count % AUTOSAVE_INTERVAL == 0:
            self.save_state()

    def apply_decay(self,factor):
        for n in list(self.neurons.values()):
            n.decay(factor)
        to_remove = [nid for nid,n in self.neurons.items() if not n.connections and normalize_text(n.concept) not in self.memory]
        for nid in to_remove:
            del self.neurons[nid]
            self.index.pop(normalize_text(n.concept),None)
        if to_remove:
            print(f"[decay] pruned {len(to_remove)} neurons")

    # ---------- Reasoning ----------
    def find_similar(self,text,top_k=5):
        tokens = tokenize(text)
        scores=[]
        for nid,n in self.neurons.items():
            sim = cosine_sim(random_embedding(), n.embedding)  # placeholder simple embedding
            if sim>0:
                scores.append((sim,nid,n.concept))
        scores.sort(reverse=True,key=lambda x:x[0])
        return scores[:top_k]

    def graph_walk_scores(self,start_tokens,max_steps=3):
        matched=set()
        for nid,n in self.neurons.items():
            matched.add(nid)
        scores=defaultdict(float)
        q=deque()
        for nid in matched:
            q.append((nid,1.0,0))
            scores[nid]+=1.0
        while q:
            nid,strength,depth=q.popleft()
            if depth>=max_steps: continue
            node=self.neurons.get(nid)
            if not node: continue
            for tgt,w in node.connections.items():
                prop = strength*w
                if prop<0.01: continue
                scores[tgt]+=prop
                q.append((tgt,prop,depth+1))
        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        return ranked

    def propose_reply(self,input_text):
        if input_text in self.memory:
            return self.memory[input_text],"memory_exact"
        # graph reasoning
        start_tokens = tokenize(input_text)
        ranked = self.graph_walk_scores(start_tokens,max_steps=3)
        for nid,_ in ranked:
            concept=self.neurons[nid].concept
            if concept in self.memory:
                self.strengthen(input_text,concept,0.1)
                return self.memory[concept],"graph_walk"
        return f"I'm learning '{input_text}'. Teach me with: teach: <reply>","unknown"

    # ---------- Interaction ----------
    def handle_input(self,user_input):
        user_input = user_input.strip()
        if not user_input: return "Say something."
        self.maybe_decay()
        reply,src = self.propose_reply(user_input)
        self._note_seen(user_input)
        return reply + f" [source:{src}]"

    def teach(self,input_text,reply_text):
        self.memory[input_text] = reply_text
        self.link(input_text,reply_text)
        self._note_seen(input_text)
        self._note_seen(reply_text)
        self.save_state()
        return "Learned. Thank you."

    def learn_file(self,filepath):
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"
        with open(filepath,"r",encoding="utf-8",errors="ignore") as f:
            text=f.read()
        sentences=[]
        cur=[]
        for ch in text:
            cur.append(ch)
            if ch in ".!?":
                s=''.join(cur).strip()
                if s: sentences.append(s)
                cur=[]
        if cur: sentences.append(''.join(cur).strip())
        for i in range(0,len(sentences),BULK_LEARN_CHUNK_SIZE):
            chunk = [normalize_text(s) for s in sentences[i:i+BULK_LEARN_CHUNK_SIZE]]
            for s in chunk: self._ensure_neuron(s); self.memory[s]=s
        self.save_state()
        return f"Learned {len(sentences)} sentences."

    def show_stats(self):
        return {"neurons":len(self.neurons),"memories":len(self.memory),"interactions":self.interaction_count}

    def _note_seen(self,concept):
        n=self._ensure_neuron(concept)
        n.seen_count+=1
        n.last_seen=now_ts()

# -------------------------
# CLI
# -------------------------
def repl():
    ai = Neuralic()
    ai.load_state()
    print("\nNeuralic 3.0 — Chat & Teach AI")
    last_input=None
    while True:
        try:
            txt=input("You: ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\nSaving state...")
            ai.save_state()
            break
        if not txt: continue
        if txt.lower() in ("quit","exit"):
            print("Saving and exiting...")
            ai.save_state()
            break
        if txt.lower().startswith("teach:"):
            if not last_input: print("No recent input to teach."); continue
            reply = txt[len("teach:"):].strip()
            if not reply: print("teach: needs reply"); continue
            print(ai.teach(last_input,reply))
            continue
        if txt.lower().startswith("learn_file:"):
            path = txt[len("learn_file:"):].strip()
            print(ai.learn_file(path))
            continue
        if txt.lower()=="stats":
            print(ai.show_stats())
            continue
        last_input=txt
        print("AI:",ai.handle_input(txt))

if __name__=="__main__":
    repl()
