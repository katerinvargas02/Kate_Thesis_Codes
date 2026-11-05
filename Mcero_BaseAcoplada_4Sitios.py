import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from V2ClebschGordan import ClebschGordan

t_inicio=time.perf_counter()

N=2
t1=1.0
t2=0.1
U=2.0
js=np.arange(0,N/2+0.5,0.5)

if N%2!=0:
    print("N impar: M=m1+m2 es semientero, no existe el sector M=0")
    raise SystemExit


# -2t Jx for each dimer
def j_x_1(j1,m1,j2,m2):
    c1=-t1*np.sqrt(j1*(j1+1)-m1*(m1+1))
    c2=-t1*np.sqrt(j1*(j1+1)-m1*(m1-1))
    B1=np.array([j1,m1+1,j2,m2])
    B2=np.array([j1,m1-1,j2,m2])
    return [(c1,B1),(c2,B2)]

def j_x_2(j1,m1,j2,m2):
    c1=-t1*np.sqrt(j2*(j2+1)-m2*(m2+1))
    c2=-t1*np.sqrt(j2*(j2+1)-m2*(m2-1))
    B1=np.array([j1,m1,j2,m2+1])
    B2=np.array([j1,m1,j2,m2-1])
    return [(c1,B1),(c2,B2)]

# U/4 Ni^2 + U(Jzi)^2 - U/2 Ni
def interaction_1(j1,m1,j2,m2):
    B=np.array([j1,m1,j2,m2])
    c=(U/4)*(2*j1)**2+U*m1**2-(U/2)*(2*j1)
    return [(c,B)]

def interaction_2(j1,m1,j2,m2):
    B=np.array([j1,m1,j2,m2])
    c=(U/4)*(2*j2)**2+U*m2**2-(U/2)*(2*j2)
    return [(c,B)]

# Hopping between dimers 1 and 2
def creation2_annihilation3(j1,m1,j2,m2):
    c=-t2*np.sqrt(j1+m1+1)*np.sqrt(j2-m2)
    B=np.array([j1+0.5,m1+0.5,j2-0.5,m2+0.5])
    return [(c,B)]

def creation3_annihilation2(j1,m1,j2,m2):
    c=-t2*np.sqrt(j1+m1)*np.sqrt(j2-m2+1)
    B=np.array([j1-0.5,m1-0.5,j2+0.5,m2-0.5])
    return [(c,B)]

# Hamiltonian
def hamiltonian(j1,m1,j2,m2):
    return (j_x_1(j1,m1,j2,m2)+interaction_1(j1,m1,j2,m2)
           +j_x_2(j1,m1,j2,m2)+interaction_2(j1,m1,j2,m2)
           +creation2_annihilation3(j1,m1,j2,m2)
           +creation3_annihilation2(j1,m1,j2,m2))


# Uncoupled Basis

# Uncoupled Basis, M=0
basis=[]

for j1 in js:
    for j2 in js:
        for m1 in np.arange(-j1,j1+1):
            for m2 in np.arange(-j2,j2+1):
                if m1+m2==0:
                    basis.append(np.array([j1,m1,j2,m2]))

M=len(basis)
index={tuple(ket):i for i,ket in enumerate(basis)}

# Uncoupled Hamiltonian
rows=[]
cols=[]
vals=[]

for j,ket in enumerate(basis):
    j1,m1,j2,m2=ket

    if 2*j1+2*j2==N:
        for c,state in hamiltonian(j1,m1,j2,m2):
            i=index.get(tuple(state))
            if i is not None:
                rows.append(i)
                cols.append(j)
                vals.append(c)

H=coo_matrix((vals,(rows,cols)),shape=(M,M)).tocsr()



# Coupled Basis, M=0
coupled_basis=[]

for j1 in js:
    for j2 in js:
        for J in np.arange(abs(j1-j2),j1+j2+1):
            if J%1==0:
                coupled_basis.append(np.array([j1,j2,J,0]))

M_coupled=len(coupled_basis)

print("Uncoupled dimension =",M)
print("Coupled dimension =",M_coupled)
print("Dimension ampliada M=0:",M)
print("Dimensiones iguales:",M==M_coupled)


cg_cache={}
for ket in coupled_basis:
    j1,j2,J,M_J=ket
    if (j1,j2,J) not in cg_cache:
        cg_cache[(j1,j2,J)]=ClebschGordan(j1,j2,J,Mmin=-1)

expansiones=[]
for ket in coupled_basis:
    j1,j2,J,M_J=ket
    expansion=[]
    for cg in cg_cache[(j1,j2,J)]:
        if cg.M==M_J:
            i=index.get((j1,cg.m1,j2,cg.m2))
            if i is not None:
                expansion.append((i,float(cg.Value)))
    expansiones.append(expansion)

bra_de={}
for beta,expansion in enumerate(expansiones):
    for i,c in expansion:
        bra_de.setdefault(i,[]).append((beta,c))

# <beta|H|alpha>
filas=[]
columnas=[]
valores=[]

for alpha in range(M_coupled):

    j1,j2,J,M_J=coupled_basis[alpha]

    if 2*j1+2*j2==N:

        for i_ket,c_ket in expansiones[alpha]:
            j1,m1,j2,m2=basis[i_ket]

            for c_H,estado in hamiltonian(j1,m1,j2,m2):
                i_res=index.get(tuple(estado))

                if i_res is None:
                    continue

                for beta,c_bra in bra_de.get(i_res,[]):
                    filas.append(beta)
                    columnas.append(alpha)
                    valores.append(np.conj(c_bra)*c_H*c_ket)

H_coupled=coo_matrix( (valores,(filas,columnas)),shape=(M_coupled,M_coupled)).tocsr()

print("H shape =",H.shape)
print("H_coupled shape =",H_coupled.shape)

def E0_min(A):
    try:
        return eigsh(A,k=1,which='SA')[0][0]
    except Exception:
        return np.linalg.eigvalsh(A.toarray())[0]

E0_uncoupled=E0_min(H)
E0_coupled=E0_min(H_coupled)
print("Uncoupled energy =",E0_uncoupled)
print("Coupled energy =",E0_coupled)
print("Tiempo total =",time.perf_counter()-t_inicio,"s")
#print("Hamiltoniano desacoplado:")
#print(H.toarray())

#print("Hamiltoniano acoplado:")
#print(H_coupled.toarray())
if M_coupled<=200:
    plt.figure(figsize=(10,10))
    plt.imshow(H_coupled.toarray(), cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Matrix element")
    plt.xlabel("Ket index")
    plt.ylabel("Bra index")
    plt.title(f"M=0 Coupled basis - 4 sites, N={N}", pad=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig( "Mcero_CoupledBasis_N6_L2.png", bbox_inches="tight", pad_inches=0.05)
    plt.show()
    


