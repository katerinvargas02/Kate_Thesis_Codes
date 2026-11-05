
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

N = 2
L = 2 #Dimer Number 
t1 = 1.0
U = 2.0
t2s = np.linspace(0,0.2,101)  #Parametro perturbativo

# Hopping a_i^dagger a_j + a_j^dagger a_i
def hopping(S,i,j,t):
    c1=-t*np.sqrt((S[i]+1)*S[j])
    c2=-t*np.sqrt((S[j]+1)*S[i])

    B1=S.copy()
    B2=S.copy()

    B1[i]+=1
    B1[j]-=1

    B2[i]-=1
    B2[j]+=1

    return [(c1,B1),(c2,B2)]

# H_0: hopping inside each dimer + interaction U/2 n_i(n_i-1)
def hamiltonian(S):
    terms=[]
    for k in range(L):
        terms+=hopping(S,2*k,2*k+1,t1)
    for i in range(2*L):
        terms.append(((U/2)*S[i]*(S[i]-1),S))
    return terms

# H_LR: hopping between dimers
def H_LR(S):
    terms=[]
    for k in range(1,L):
        terms+=hopping(S,2*k-1,2*k,1.0)  
    return terms

# Fock basis (una sola vez, compartida por H y V)
basis=[np.array(s) for s in product(range(N+1),repeat=2*L) if sum(s)==N]

M=len(basis)
index={tuple(ket):i for i,ket in enumerate(basis)}

# Hamiltonian H_0
rows=[]
cols=[]
vals=[]

for j,ket in enumerate(basis):
    for c,state in hamiltonian(ket):
        i=index.get(tuple(state))
        if i is not None and c!=0:
            rows.append(i)
            cols.append(j)
            vals.append(c)

H=coo_matrix((vals,(rows,cols)),shape=(M,M)).tocsr()

# Coupling between dimers V = H_LR
rows=[]
cols=[]
vals=[]

for j,ket in enumerate(basis):
    for c,state in H_LR(ket):
        i=index.get(tuple(state))
        if i is not None and c!=0:
            rows.append(i)
            cols.append(j)
            vals.append(c)

V=coo_matrix((vals,(rows,cols)),shape=(M,M)).tocsr()

print("Number of basis states:",M)

# Diagonalization of H_0
E,Vec=np.linalg.eigh(H.toarray())
g=1 #g=1 marca que el estado fundamental es el índice 0

E0=E[0]
GS=Vec[:,0]

print("E0 =",E0)

# First-order correction
E1=GS@(V@GS)
print("E1 (first order) =",E1)

# Second-order correction
me=Vec[:,g:].T@(V@GS)
E2=np.sum(np.abs(me)**2/(E0-E[g:]))

print("E2 =",E2)

# E_0 vs t2
E_pert=E0+t2s*E1             
E_pert2=E0+t2s*E1+t2s**2*E2

plt.figure(figsize=(7,5))
plt.plot(t2s,E_pert,label="First order")
plt.plot(t2s,E_pert2,label="Second order")
plt.xlabel("t2")
plt.ylabel("E0")
plt.title(f"Perturbative ground state energy, N={N}, L={L}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
