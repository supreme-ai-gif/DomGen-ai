# neuralic_core.py
import json
import os
import time
import math
import random
from collections import deque, defaultdict

# CONFIG
STATE_FILE = "neuralic_state.json"
MAX_CONNECTIONS = 200
DECAY_RATE = 0.995        # multiply old_weight by this each decay step
LEARNING_STEP = 0.12      # amount to strengthen on co-occurrence
ACTIVATION_DECAY = 0.5    # activation lost per hop
MAX_HOPS = 3              # graph walk depth for activation spreading
AUTOSAVE_INTERVAL = 20    # interactions between autosaves
random.seed(42)

def now_ts():
    return int(time.time())

def normalize(text: str):
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).strip()

def tokens(text: str):
    return [t for t in normalize(text).split() if t]

def make_nid(concept: str):
    return f"n_{abs(hash(normalize(concept))) % (10**12)}"

class Neuron:
    def __init__(self, concept: str):
        self.nid = make_nid(concept)
        self.concept = concept
        self.tokens = tokens(concept)
        self.activation = 0.0
        self.seen = 0
        # connections: nid -> weight (0..1)
        self.connections = {}

    def connect(self, other_nid: str, weight: float):
        if other_nid == self.nid:
            return
        if other_nid in self.connections:
            self.connections[other_nid] = min(1.0, self.connections[other_nid] + weight)
        else:
            if len(self.connections) >= MAX_CONNECTIONS:
                # evict weakest
                weakest = min(self.connections.items(), key=lambda x: x[1])[0]
                del self.connections[weakest]
            self.connections[other_nid] = min(1.0, weight)

    def decay_connections(self):
        to_del = []
        for k, v in list(self.connections.items()):
            nv = v * DECAY_RATE
            if nv < 0.001:
                to_del.append(k)
            else:
                self.connections[k] = nv
        for k in to_del:
            del self.connections[k]

    def to_dict(self):
        return {
            "nid": self.nid,
            "concept": self.concept,
            "tokens": self.tokens,
            "activation": self.activation,
            "seen": self.seen,
            "connections": self.connections
        }

    @classmethod
    def from_dict(cls, d):
        n = cls(d["concept"])
        n.nid = d["nid"]
        n.tokens = d.get("tokens", tokens(n.concept))
        n.activation = float(d.get("activation", 0.0))
        n.seen = int(d.get("seen", 0))
        n.connections = {k: float(v) for k, v in d.get("connections", {}).items()}
        return n

class NeuralicBrain:
    def __init__(self, state_file=STATE_FILE):
        self.state_file = state_file
        self.neurons = {}        # nid -> Neuron
        self.index = {}          # normalized concept -> nid
        self.memory = {}         # exact input -> taught reply
        self.interactions = 0
        self._load_state()
        self._bootstrap_seed()

    # ---------- persistence ----------
    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for nid, nd in data.get("neurons", {}).items():
                n = Neuron.from_dict(nd)
                self.neurons[nid] = n
                self.index[normalize(n.concept)] = nid
            self.memory = data.get("memory", {})
            self.interactions = data.get("interactions", 0)
        except Exception as e:
            print("Error loading state:", e)

    def save_state(self):
        data = {
            "neurons": {nid: n.to_dict() for nid, n in self.neurons.items()},
            "memory": self.memory,
            "interactions": self.interactions
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ---------- bootstrap small seed ----------
    def _bootstrap_seed(self):
        seeds = {
            "hello": "Hello, I am Neuralic.",
            "how are you": "I am learning. Teach me more.",
            "what is your name": "My name is Neuralic."
        }
        for k, v in seeds.items():
            self._ensure_neuron(k)
            self._ensure_neuron(v)
            self.link(k, v, weight=0.6)
            self.memory.setdefault(k, v)

    # ---------- neuron helpers ----------
    def _ensure_neuron(self, concept: str):
        if not concept:
            concept = "<empty>"
        key = normalize(concept)
        if key in self.index:
            return self.neurons[self.index[key]]
        n = Neuron(concept)
        # collision-safe insertion
        while n.nid in self.neurons:
            n.nid = n.nid + "_" + str(random.randint(0,9999))
        self.neurons[n.nid] = n
        self.index[key] = n.nid
        return n

    # create or strengthen bidirectional link
    def link(self, a_concept: str, b_concept: str, weight: float = LEARNING_STEP):
        a = self._ensure_neuron(a_concept)
        b = self._ensure_neuron(b_concept)
        a.connect(b.nid, weight)
        b.connect(a.nid, weight * 0.4)
        self._maybe_autosave()

    def strengthen_sequence(self, concepts):
        # strengthen connections between neighbor concepts in a sequence
        nids = []
        for c in concepts:
            n = self._ensure_neuron(c)
            n.seen += 1
            nids.append(n.nid)
        for i in range(len(nids) - 1):
            a = self.neurons[nids[i]]
            b = self.neurons[nids[i+1]]
            a.connect(b.nid, LEARNING_STEP)
            b.connect(a.nid, LEARNING_STEP * 0.4)
        self._maybe_autosave()

    # decay step called occasionally
    def decay_all(self):
        for n in self.neurons.values():
            n.decay_connections()
        self._maybe_autosave()

    # ---------- activation & reasoning ----------
    def _activate_from_input(self, text: str):
        # reset activations
        for n in self.neurons.values():
            n.activation = 0.0

        toks = tokens(text)
        # activate neurons corresponding to individual tokens and whole phrase
        activated = []
        # match tokens and whole phrase
        phrase_key = normalize(text)
        if phrase_key in self.index:
            activated.append(self.index[phrase_key])
            self.neurons[self.index[phrase_key]].activation += 1.0

        for t in toks:
            key = normalize(t)
            if key in self.index:
                nid = self.index[key]
                self.neurons[nid].activation += 0.6
                activated.append(nid)
            else:
                # ensure neuron for token (growth)
                n = self._ensure_neuron(t)
                n.activation += 0.2
                activated.append(n.nid)

        # propagation: BFS-like spreading with decay
        q = deque()
        for nid in set(activated):
            q.append((nid, self.neurons[nid].activation, 0))
        seen = set()
        while q:
            nid, act, depth = q.popleft()
            if depth >= MAX_HOPS or act < 0.02:
                continue
            node = self.neurons.get(nid)
            if not node:
                continue
            # propagate along connections proportional to weight
            for tgt, w in node.connections.items():
                gain = act * w * (1 - ACTIVATION_DECAY * (depth))  # decay over hops
                if gain < 0.01:
                    continue
                self.neurons[tgt].activation += gain
                if tgt not in seen:
                    q.append((tgt, self.neurons[tgt].activation, depth + 1))
                    seen.add(tgt)

    def _top_activated(self, top_k=8):
        items = [(n.activation, n.nid, n.concept) for n in self.neurons.values() if n.activation > 0]
        items.sort(reverse=True, key=lambda x: x[0])
        return items[:top_k]

    def generate_sentence_from_activations(self, top_k=6):
        top = self._top_activated(top_k)
        if not top:
            return None
        # naive sentence builder: pick top concept as subject, then follow strong connections to assemble S-V-O-ish phrase
        subject = top[0][2]
        words = [subject]
        used = set([top[0][1]])
        current_nid = top[0][1]
        # walk connections picking strongest unseen neighbors
        for _ in range(6):
            node = self.neurons.get(current_nid)
            if not node or not node.connections:
                break
            # pick neighbor with highest weight not used
            neighbors = sorted(node.connections.items(), key=lambda x: x[1], reverse=True)
            chosen = None
            for nid, w in neighbors:
                if nid not in used:
                    chosen = nid
                    break
            if not chosen:
                break
            used.add(chosen)
            words.append(self.neurons[chosen].concept)
            current_nid = chosen
        # join into sentence with simple cleaning
        sent = " ".join(words)
        # canonical punctuation/cleanup
        if not sent.endswith("."):
            sent = sent.strip() + "."
        return sent.capitalize()

    # ---------- high-level public API ----------
    def chat(self, text: str):
        text = text.strip()
        if not text:
            return "Say something."
        self.interactions += 1
        # if exact memory (taught reply), return that
        if text in self.memory:
            # strengthen sequence between input and reply
            self.strengthen_sequence([text, self.memory[text]])
            self._maybe_autosave()
            return self.memory[text]

        # else activate and reason
        self._activate_from_input(text)
        sent = self.generate_sentence_from_activations()
        # if sentence is None or too similar to placeholder, return recall or ask to teach
        if sent:
            # store an ephemeral association candidate (not auto-memory)
            return sent + "  [generated]"
        else:
            # create placeholder memory and ask to teach
            self.memory[text] = "I don't know yet — teach me with /teach"
            self._maybe_autosave()
            return self.memory[text]

    def teach(self, input_text: str, reply_text: str):
        input_text = input_text.strip()
        reply_text = reply_text.strip()
        if not input_text or not reply_text:
            return "Input and reply required."
        self.memory[input_text] = reply_text
        # create neurons and a strong link
        self.link(input_text, reply_text, weight=0.6)
        self._maybe_autosave()
        return "Learned."

    def learn_from_text(self, text: str, chunk_size=20):
        # break into sentences (rudimentary)
        sentences = []
        cur = []
        for ch in text:
            cur.append(ch)
            if ch in ".!?":
                s = "".join(cur).strip()
                if s:
                    sentences.append(s)
                cur = []
        if cur:
            s = "".join(cur).strip()
            if s:
                sentences.append(s)

        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i+chunk_size]
            for s in chunk:
                s_norm = normalize(s)
                if s_norm not in self.memory:
                    # create neuron and placeholder memory
                    self.memory[s_norm] = f"I'm learning about: {s_norm}"
                # strengthen sequence of tokens inside sentence
                toks = tokens(s_norm)
                self.strengthen_sequence(toks)
        self._maybe_autosave()
        return f"Learned {len(sentences)} sentences."

    # ---------- utilities ----------
    def _maybe_autosave(self):
        if self.interactions % AUTOSAVE_INTERVAL == 0:
            self.save_state()

    def stats(self):
        return {
            "neurons": len(self.neurons),
            "memory_items": len(self.memory),
            "interactions": self.interactions
        }
