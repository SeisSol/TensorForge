# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Structural checks over a generated dump; no reference needed.

    python3 check_structure.py gen/gpulike_subroutine.cpp

Reports, per kernel:
  DROPPED      a multilinear result that is neither stored nor read again
  BIAS REUSED  a register array serving as the bias of two accumulations
  STALE READ   a load overtaken by a store to the tensor it reads
  OOB READ     a store indexing past the end of its accumulator

Each of these was a real defect at some point; they are cheap enough to run
over every dump.
"""
import re, sys, itertools
from collections import Counter
path=sys.argv[1]
L=open(path).read().split('\n')
starts=[(i,re.match(r'\s*kernel_(kernel_\w+)\(',l).group(1)) for i,l in enumerate(L) if re.match(r'\s*kernel_kernel_\w+\(.*\{$',l)]
ends=[i for i,l in enumerate(L) if l.startswith('void launcher_')]
COMPUTE=re.compile(r'//\s*(\w+) = \+\(([^)]*)\) \+ (?:None|name: (\w+))')
STORED=re.compile(r'//\s*\w+ = store\{r>[gs]\}\((?:\w+,\s*)?(\w+)\)')
GLBLOAD=re.compile(r'//\s*(\w+) = load\{g>r\}\((glb_\w+)\);')
GLBSTORE=re.compile(r'//\s*(glb_\w+) = store\{r>g\}\((\w+)\);')
HEAD=GLBSTORE; DECL=re.compile(r'float (r\d+)\[(\d+)\]')
ASSIGN=re.compile(r'int32_t (\w+) = (.+);$'); SRCRD=re.compile(r'float value = (r\d+)\[(\w+)\];')
FOR=re.compile(r'for \(int32_t (\w+) = (-?\d+); \w+ < (-?\d+);')
def expand(e,env):
    for _ in range(14):
        n=re.sub(r'\b(v\d+\w*)\b', lambda m: f'({env[m.group(1)]})' if m.group(1) in env else m.group(1), e)
        if n==e: break
        e=n
    return e
bad=0
for k,(s,name) in enumerate(starts):
    e=min([x for x in ends if x>s]); body=L[s:e]; txt='\n'.join(body)
    produced=[];consumed=set();biases=[];seq=[]
    for l in body:
        t=l.strip()
        m=GLBLOAD.match(t)
        if m: seq.append(('load',m.group(1),m.group(2))); continue
        m=GLBSTORE.match(t)
        if m: seq.append(('store',m.group(1),m.group(2))); continue
        m=COMPUTE.match(t)
        if m:
            rd=set(re.findall(r'\b[rs]\d+\b',m.group(2)))
            if m.group(3): rd.add(m.group(3))
            seq.append(('compute',m.group(1),rd))
    for m in COMPUTE.finditer(txt):
        produced.append(m.group(1)); consumed.update(re.findall(r'\b[rs]\d+\b',m.group(2)))
        if m.group(3): biases.append(m.group(3)); consumed.add(m.group(3))
    consumed.update(STORED.findall(txt))
    lost=[r for r in produced if r not in consumed]
    dup=[r for r,c in Counter(biases).items() if c>1 and r.startswith('r')]
    stale=[]
    for i,(kind,reg,tensor) in enumerate(seq):
        if kind!='load': continue
        for j in range(i+1,len(seq)):
            if seq[j][0]=='compute' and reg in seq[j][2]:
                if any(kk=='store' and tt==tensor for kk,tt,_ in seq[i+1:j]): stale.append((reg,tensor))
                break
    sizes={m.group(1):int(m.group(2)) for m in (DECL.search(x) for x in body) if m}
    oor=[]
    for i,l in enumerate(body):
        if not HEAD.search(l): continue
        env={};rng=[]
        for t in (x.strip() for x in body[i+1:i+700]):
            if HEAD.search(t) or re.match(r'//\s*\w+ = (load|\+\()',t): break
            m=FOR.search(t)
            if m:
                v=re.search(r'int32_t (\w+) =',t).group(1); rng.append((v,int(m.group(2)),int(m.group(3))-1)); continue
            m=ASSIGN.match(t)
            if m: env[m.group(1)]=m.group(2)
            m=SRCRD.match(t)
            if not m: continue
            reg,idx=m.group(1),re.sub(r'\(threadIdx\.x % \d+\)','0',expand(m.group(2),env))
            names=[v for v,_,_ in rng]
            for combo in itertools.product(*[(a,b) for _,a,b in rng]):
                try: val=eval(idx,{'__builtins__':{}},dict(zip(names,combo)))
                except Exception: break
                if reg in sizes and not 0<=val<sizes[reg]: oor.append((reg,val,sizes[reg]))
    p=[]
    if lost: p.append(f"DROPPED {lost[:4]}")
    if dup: p.append(f"BIAS REUSED {dup[:3]}")
    if stale: p.append(f"STALE READ {stale[:3]}")
    if oor: p.append(f"OOB READ {oor[:3]}")
    if p: bad+=1; print(f"  #{k:2} {name:22} {'; '.join(p)}")
print(f"flagged: {bad} of {len(starts)}   [{path}]")
