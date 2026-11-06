"""Local HP lattice controls and references, never an Adapt-1 selector."""
from dataclasses import dataclass
from functools import cached_property
import itertools
import math
import random
import time


def directions(dim):
    return tuple(tuple(sign if j == axis else 0 for j in range(dim))
                 for axis in range(dim) for sign in (1, -1))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def distance(a, b):
    return sum(abs(x-y) for x, y in zip(a, b))


@dataclass(frozen=True)
class State:
    sequence: str
    chain: tuple
    dim: int = 3

    def __post_init__(self):
        assert self.dim in (2, 3)
        assert len(self.sequence) >= 2 and set(self.sequence) <= {'H', 'P'}
        assert 2 <= len(self.chain) <= len(self.sequence)
        assert all(len(p) == self.dim and all(type(v) is int for v in p) for p in self.chain)
        assert len(set(self.chain)) == len(self.chain)
        assert all(distance(a,b) == 1 for a,b in zip(self.chain,self.chain[1:]))

    @classmethod
    def initial(cls, sequence, dim=3):
        return cls(sequence, (tuple(0 for _ in range(dim)), (1,) + (0,)*(dim-1)), dim)

    @property
    def complete(self):
        return len(self.chain) == len(self.sequence)

    @cached_property
    def legal(self):
        if self.complete:
            return ()
        occupied=set(self.chain)
        return tuple(a for a in directions(self.dim) if add(self.chain[-1],a) not in occupied)

    @property
    def terminal(self):
        return self.complete or not self.legal

    def step(self, action):
        assert action in self.legal
        return State(self.sequence,self.chain+(add(self.chain[-1],action),),self.dim)

    @cached_property
    def contacts(self):
        return sum(self.sequence[i] == self.sequence[j] == 'H' and distance(a,b) == 1
            for i,a in enumerate(self.chain) for j,b in enumerate(self.chain[:max(0,i-1)]))

    @cached_property
    def upper_bound(self):
        n=len(self.sequence)
        return max(1,sum(2*self.dim-int(i>0)-int(i<n-1) for i,t in enumerate(self.sequence) if t=='H')//2)

    def partial_score(self):
        return 0. if self.terminal and not self.complete else .5+.5*self.contacts/self.upper_bound

    def utility(self):
        assert self.terminal
        return self.partial_score()

    def features(self):
        """Fixed-width normalized summaries; deliberately not a lossless encoding.

        No length, residue index slots, action scores, future rollouts, or exact
        solver values appear. Scaling a chain does not change vector width.
        """
        n=len(self.sequence);placed=len(self.chain);remaining=n-placed
        hs=self.sequence.count('H');hp=self.sequence[:placed].count('H')
        center=tuple(sum(p[k] for p in self.chain)/placed for k in range(self.dim))
        radius=math.sqrt(sum(sum((p[k]-center[k])**2 for k in range(self.dim)) for p in self.chain)/placed)
        occupied=set(self.chain);tip=self.chain[-1]
        nonbond_h=0
        for d in directions(self.dim):
            q=add(tip,d)
            if q in occupied:
                index=self.chain.index(q)
                nonbond_h+=index<placed-2 and self.sequence[index]=='H'
        degrees=[sum(add(p,d) in occupied for d in directions(self.dim)) for p in self.chain]
        values={
            'placed_fraction':placed/n,
            'remaining_fraction':remaining/n,
            'hydrophobic_fraction':hs/n,
            'placed_hydrophobic_fraction':hp/placed,
            'remaining_hydrophobic_fraction':self.sequence[placed:].count('H')/max(1,remaining),
            'contact_capacity_fraction':self.contacts/self.upper_bound,
            'h_contact_degree_fraction':2*self.contacts/max(1,(2*self.dim-2)*hs+2),
            'legal_direction_fraction':len(self.legal)/(2*self.dim),
            'tip_h_neighbor_fraction':nonbond_h/(2*self.dim-1),
            'tip_is_hydrophobic':float(self.sequence[placed-1]=='H'),
            'next_is_hydrophobic':float(remaining>0 and self.sequence[placed]=='H'),
            'mean_occupied_degree_fraction':sum(degrees)/(placed*2*self.dim),
            'radius_over_chain_length':radius/n,
            'end_to_end_over_chain_length':distance(self.chain[0],tip)/n,
        }
        assert all(math.isfinite(v) and -1e-12<=v<=1+1e-12 for v in values.values())
        return values

    def record(self):
        return {'sequence':self.sequence,'dim':self.dim,'chain':[list(p) for p in self.chain],
            'contacts':self.contacts,'complete':self.complete,'terminal':self.terminal,
            'utility':self.utility() if self.terminal else None}


def sequence_key(s):
    return min(s,s[::-1])


def sequences(seed,lengths,excluded=()):
    rng=random.Random(seed);seen=set(excluded);out=[]
    for n in lengths:
        while True:
            s=''.join(rng.choice('HP') for _ in range(n));key=sequence_key(s)
            if key not in seen and n//4<=s.count('H')<=3*n//4:
                seen.add(key);out.append(s);break
    return out


def plan(state,depth,arm='C2',rng=None):
    stats={'nodes':0,'leaves':0}
    def value(s,d):
        stats['nodes']+=1
        if s.terminal:
            stats['leaves']+=1
            return s.utility()
        if d==0:
            stats['leaves']+=1
            return .5 if arm=='C1' else s.partial_score()
        return max(value(s.step(a),d-1) for a in s.legal)
    values=[(a,value(state.step(a),depth-1)) for a in state.legal]
    maximum=max(v for a,v in values);ties=[a for a,v in values if v==maximum]
    return (ties[0] if rng is None else rng.choice(ties)),stats


def episode(sequence,dim,arm,depth=1,seed=None,deadline=None):
    state=State.initial(sequence,dim);states=[state];actions=[];nodes=leaves=0
    rng=random.Random(seed) if seed is not None else None
    started=time.monotonic()
    while not state.terminal:
        if deadline is not None and time.monotonic()>deadline:
            raise TimeoutError('local precheck deadline')
        if arm=='random':
            action=(rng or random.Random(0)).choice(state.legal)
        else:
            action,cost=plan(state,depth,arm,rng);nodes+=cost['nodes'];leaves+=cost['leaves']
        actions.append(action);state=state.step(action);states.append(state)
    return {'sequence':sequence,'dim':dim,'arm':arm,'depth':None if arm=='random' else depth,
        'seed':seed,'actions':[list(a) for a in actions],'final':state.record(),
        'utility':state.utility(),'contacts':state.contacts,'complete':state.complete,
        'dead_end':not state.complete,'decisions':len(actions),'nodes':nodes,'leaves':leaves,
        'seconds':time.monotonic()-started},states


def exact_reference(sequence,dim=3,deadline=None):
    """Enumerate every legal continuation with only the first bond fixed +x."""
    initial=State.initial(sequence,dim);chain=list(initial.chain);occupied={p:i for i,p in enumerate(chain)}
    dirs=directions(dim);nodes=complete=dead=0;best=-1;best_chain=None
    started=time.monotonic()
    def visit(contacts):
        nonlocal nodes,complete,dead,best,best_chain
        nodes+=1
        if nodes%8192==0 and deadline is not None and time.monotonic()>deadline:
            raise TimeoutError('exact reference deadline')
        index=len(chain)
        if index==len(sequence):
            complete+=1
            if contacts>best:best=contacts;best_chain=tuple(chain)
            return
        nexts=[add(chain[-1],d) for d in dirs if add(chain[-1],d) not in occupied]
        if not nexts:dead+=1
        for point in nexts:
            gain=0
            if sequence[index]=='H':
                for d in dirs:
                    previous=occupied.get(add(point,d))
                    if previous is not None and previous<index-1 and sequence[previous]=='H':gain+=1
            occupied[point]=index;chain.append(point);visit(contacts+gain);chain.pop();del occupied[point]
    visit(0)
    witness=State(sequence,best_chain,dim)
    assert witness.contacts==best and witness.complete
    return {'sequence':sequence,'dim':dim,'certified_optimal_contacts':best,'certified_optimal_energy':-best,
        'optimal_utility':witness.utility(),'witness':witness.record(),'enumerated_nodes':nodes,
        'complete_walks':complete,'dead_end_walks':dead,'seconds':time.monotonic()-started,
        'proof':'exhaustive legal continuations, first bond fixed +x by lattice symmetry'}


def canonical_geometry(state):
    """Remove translations, axis permutations and sign flips for coverage only."""
    origin=state.chain[0];centered=[tuple(p[k]-origin[k] for k in range(state.dim)) for p in state.chain]
    return min(tuple(tuple(p[perm[k]]*sign[k] for k in range(state.dim)) for p in centered)
        for perm in itertools.permutations(range(state.dim)) for sign in itertools.product((-1,1),repeat=state.dim))
