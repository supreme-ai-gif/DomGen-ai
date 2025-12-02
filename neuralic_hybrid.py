# neuralic_hybrid.py
# Neuralic 2.2 Hybrid Brain (symbolic rules + neural graph)
import json, os, time, random, math
from collections import defaultdict, deque

STATE_FILE = os.getenv("NEURALIC_STATE_FILE", "neuralic_state.json")
AUTOSAVE_INTERVAL = 30
MAX_CONN = 200
DECAY = 0.995
LEARN_STEP = 0.12
MAX_HOPS = 3
ACT_DECAY = 0.55

random.seed(42)

# ---------------- utils ----------------
def now_ts(): return int(time.time())
def normalize(s: str):
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in (s or "")).strip()
def tokens(s: str):
    return [t for t in normalize(s).split() if t]
def make_nid(concept: str):
    return f"n_{abs(hash(normalize(concept))) % (10**12)}"

# ---------------- neuron ----------------
class Neuron:
    def __init__(self, concept: str):
        self.nid = make_nid(concept)
        self.concept = concept
        self.tokens = tokens(concept)
        self.activation = 0.0
        self.seen = 0
        self.connections = {}  # nid -> weight

    def connect(self, other_nid, weight):
        if other_nid == self.nid: return
        if other_nid in self.connections:
            self.connections[other_nid] = min(1.0, self.connections[other_nid] + weight)
        else:
            if len(self.connections) >= MAX_CONN:
                # evict weakest
                weakest = min(self.connections.items(), key=lambda x: x[1])[0]
                del self.connections[weakest]
            self.connections[other_nid] = min(1.0, weight)

    def decay(self):
        to_del = []
        for k,v in list(self.connections.items()):
            nv = v * DECAY
            if nv < 0.001:
                to_del.append(k)
            else:
                self.connections[k] = nv
        for k in to_del: del self.connections[k]

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
        n = cls(d.get("concept",""))
        n.nid = d.get("nid", n.nid)
        n.tokens = d.get("tokens", tokens(n.concept))
        n.activation = float(d.get("activation", 0.0))
        n.seen = int(d.get("seen", 0))
        n.connections = {k: float(v) for k,v in d.get("connections", {}).items()}
        return n

# ---------------- rule engine (symbolic) ----------------
class RuleEngine:
    def __init__(self):
        # rules stored simply: rule_name -> dict form
        # example rule types: membership (A is letter), sequence (A->B->C), implication (if X then Y)
        self.rules = {}

    def add_fact(self, subject, predicate, obj):
        # predicate e.g. "is", "type", "position", "property"
        key = f"fact::{normalize(subject)}::{predicate}::{normalize(obj)}"
        self.rules[key] = {"type":"fact","subject":normalize(subject),"predicate":predicate,"object":normalize(obj),"added":now_ts()}

    def add_sequence(self, seq_name, items):
        key = f"seq::{seq_name}::{now_ts()}"
        self.rules[key] = {"type":"sequence","name":seq_name,"items":[normalize(i) for i in items],"added":now_ts()}

    def infer(self, query):
        # simple inference: check facts and sequences for direct answers
        q = normalize(query)
        results = []
        for k,r in self.rules.items():
            if r["type"] == "fact":
                if q in (r["subject"], r["object"]):
                    results.append(r)
            elif r["type"] == "sequence":
                if any(q == item for item in r["items"]):
                    results.append(r)
        return results

    def load(self, d):
        self.rules = d.get("rules", {})

    def dump(self):
        return {"rules": self.rules}

# ---------------- hybrid brain ----------------
class NeuralicHybrid:
    def __init__(self, state_file=STATE_FILE):
        self.state_file = state_file
        self.neurons = {}     # nid -> Neuron
        self.index = {}       # normalized concept -> nid
        self.memory = {}      # exact input -> taught reply
        self.rule_engine = RuleEngine()
        self.interactions = 0
        self._load()
        self._bootstrap()

    # persistence
    def _load(self):
        if not os.path.exists(self.state_file): return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for nid, nd in data.get("neurons", {}).items():
                n = Neuron.from_dict(nd)
                self.neurons[nid] = n
                self.index[normalize(n.concept)] = nid
            self.memory = data.get("memory", {})
            self.interactions = int(data.get("interactions", 0))
            self.rule_engine.load(data.get("rules", {}))
        except Exception as e:
            print("load error:",e)

    def save(self):
        data = {
            "neurons": {nid: n.to_dict() for nid,n in self.neurons.items()},
            "memory": self.memory,
            "interactions": self.interactions,
            "rules": self.rule_engine.dump()
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # bootstrap
    def _bootstrap(self):
        # minimal seed
        if "hello" not in self.memory:
            self.memory["hello"] = "Hello — I am Neuralic (hybrid)."
        if not self.rule_engine.rules:
            # sequence A..Z
            letters = [chr(c) for c in range(65,91)]
            self.rule_engine.add_sequence("alphabet_upper", letters)

    # neuron helpers
    def _ensure_neuron(self, concept):
        key = normalize(concept)
        if not key: key = "<empty>"
        if key in self.index:
            return self.neurons[self.index[key]]
        n = Neuron(concept)
        while n.nid in self.neurons:
            n.nid += "_" + str(random.randint(0,9999))
        self.neurons[n.nid] = n
        self.index[key] = n.nid
        return n

    def link(self, a_concept, b_concept, weight=LEARN_STEP):
        a = self._ensure_neuron(a_concept)
        b = self._ensure_neuron(b_concept)
        a.connect(b.nid, weight)
        b.connect(a.nid, weight * 0.4)

    def strengthen_sequence(self, items):
        nids = []
        for c in items:
            n = self._ensure_neuron(c)
            n.seen += 1
            nids.append(n.nid)
        for i in range(len(nids)-1):
            a = self.neurons[nids[i]]; b = self.neurons[nids[i+1]]
            a.connect(b.nid, LEARN_STEP)
            b.connect(a.nid, LEARN_STEP*0.4)

    def decay_all(self):
        for n in self.neurons.values():
            n.decay()

    # concept extractor + rule learner
    def extract_and_learn(self, text):
        # naive extraction: lines of "X is Y" become facts; sequences of tokens become sequences
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        learned = {"facts":0,"sequences":0,"sentences":0}
        for line in lines:
            learned["sentences"] += 1
            # try pattern "X is Y" or "X are Y"
            parts = line.split(" is ")
            if len(parts) == 2:
                subj = parts[0].strip()
                obj = parts[1].strip().strip(".")
                self.rule_engine.add_fact(subj, "is", obj)
                self.link(subj, obj, weight=0.6)
                learned["facts"] += 1
                continue
            parts = line.split(" are ")
            if len(parts) == 2:
                subj = parts[0].strip()
                obj = parts[1].strip().strip(".")
                self.rule_engine.add_fact(subj, "are", obj)
                self.link(subj, obj, weight=0.6)
                learned["facts"] += 1
                continue
            # treat as sequence of tokens (word-level)
            toks = tokens(line)
            if len(toks) > 1:
                self.strengthen_sequence(toks)
                learned["sequences"] += 1
        self.interactions += 1
        if self.interactions % AUTOSAVE_INTERVAL == 0:
            self.save()
        return learned

    # activation + hybrid reasoning
    def _activate(self, text):
        # reset
        for n in self.neurons.values(): n.activation = 0.0
        phrase_key = normalize(text)
        activated = set()
        if phrase_key in self.index:
            activated.add(self.index[phrase_key])
            self.neurons[self.index[phrase_key]].activation += 1.0
        for t in tokens(text):
            if normalize(t) in self.index:
                nid = self.index[normalize(t)]
                self.neurons[nid].activation += 0.7
                activated.add(nid)
            else:
                n = self._ensure_neuron(t)
                n.activation += 0.2
                activated.add(n.nid)
        # propagate
        q = deque([(nid, self.neurons[nid].activation, 0) for nid in activated])
        seen = set(activated)
        while q:
            nid, act, depth = q.popleft()
            if depth >= MAX_HOPS or act < 0.02: continue
            node = self.neurons.get(nid)
            if not node: continue
            for tgt, w in node.connections.items():
                gain = act * w * (1 - ACT_DECAY * depth)
                if gain < 0.01: continue
                self.neurons[tgt].activation += gain
                if tgt not in seen:
                    seen.add(tgt); q.append((tgt, self.neurons[tgt].activation, depth+1))

    def _top(self, k=8):
        arr = [(n.activation, n.nid, n.concept) for n in self.neurons.values() if n.activation>0]
        arr.sort(reverse=True, key=lambda x:x[0])
        return arr[:k]

    # sentence generator (hybrid): use top symbols + apply simple grammar templates + symbolic checks
    def generate_from_activation(self):
        top = self._top(6)
        if not top: return None
        # use symbolic rules: if top contains sequence from rule engine, return descriptive phrase
        for _,_,concept in top:
            # check for facts related to concept
            infs = self.rule_engine.infer(concept)
            if infs:
                # construct a fact-based sentence
                r = infs[0]
                if r["type"] == "fact":
                    return f"{r['subject'].capitalize()} {r['predicate']} {r['object']}."
                elif r["type"] == "sequence":
                    seq = " ".join(r["items"][:6])
                    return f"Sequence sample: {seq}."
        # fallback: walk connections from top[0] to build phrase
        subject = top[0][2]
        words = [subject]
        used = {top[0][1]}
        cur = top[0][1]
        for _ in range(6):
            node = self.neurons.get(cur)
            if not node or not node.connections: break
            neighbors = sorted(node.connections.items(), key=lambda x:x[1], reverse=True)
            chosen = None
            for nid,_ in neighbors:
                if nid not in used:
                    chosen = nid; break
            if not chosen: break
            used.add(chosen)
            words.append(self.neurons[chosen].concept)
            cur = chosen
        sent = " ".join(words).strip()
        if not sent.endswith("."): sent += "."
        return sent.capitalize()

    # public API: chat, teach, learn_file, stats
    def chat(self, text):
        text = str(text).strip()
        if not text: return "Say something."
        self.interactions += 1
        # exact memory
        if text in self.memory:
            # strengthen link
            self.link(text, self.memory[text], weight=0.6)
            if self.interactions % AUTOSAVE_INTERVAL == 0: self.save()
            return self.memory[text]
        # symbolic quick-check
        rule_res = self.rule_engine.infer(text)
        if rule_res:
            # return first direct inference
            r = rule_res[0]
            if r["type"] == "fact":
                return f"{r['subject'].capitalize()} {r['predicate']} {r['object']}."
            elif r["type"] == "sequence":
                return f"Sequence: {' '.join(r['items'][:10])}."
        # hybrid activation->generate
        self._activate(text)
        out = self.generate_from_activation()
        if out:
            return out + "  [generated]"
        # else placeholder and invitation to teach
        placeholder = "I don't know. Please teach me using /teach."
        self.memory[text] = placeholder
        if self.interactions % AUTOSAVE_INTERVAL == 0: self.save()
        return placeholder

    def teach(self, inp, reply):
        inp = str(inp).strip(); reply = str(reply).strip()
        if not inp or not reply: return "input+reply required"
        self.memory[inp] = reply
        self.link(inp, reply, weight=0.8)
        if self.interactions % AUTOSAVE_INTERVAL == 0: self.save()
        return "Learned."

    def learn_file_text(self, text):
        res = self.extract_and_learn(text)
        if self.interactions % AUTOSAVE_INTERVAL == 0: self.save()
        return res

    def stats(self):
        return {"neurons": len(self.neurons), "memory": len(self.memory), "rules": len(self.rule_engine.rules), "interactions": self.interactions}
