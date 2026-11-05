import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from ClebschGordan_4Sitios import ClebschGordan


N=2
t1=1.0
t2=0.1
U=2.0
js=np.arange(0,N/2+0.5,0.5)

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
basis=[]
for j1 in js:
    for m1 in np.arange(-j1,j1+1):
        for j2 in js:
            #if 2*j1+2*j2==N:
                for m2 in np.arange(-j2,j2+1):
                    basis.append(np.array([j1,m1,j2,m2]))

# Uncoupled Hamiltonian
M=len(basis)
index={tuple(ket):i for i,ket in enumerate(basis)}
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


# Coupled Basis

coupled_basis=[]

for j1 in js:
    for j2 in js:
        #if 2*j1+2*j2==N:
            for J in np.arange(abs(j1-j2),j1+j2+1):
                for M_J in np.arange(-J,J+1):
                    coupled_basis.append(np.array([j1,j2,J,M_J]))

M_coupled=len(coupled_basis)

print("Uncoupled dimension =",M)
print("Coupled dimension =",M_coupled)


cg_cache={}
for ket in coupled_basis:
    j1,j2,J,M_J=ket
    if (j1,j2,J) not in cg_cache:
        cg_cache[(j1,j2,J)]=ClebschGordan(j1,j2,J)


expansiones=[]
for ket in coupled_basis:
    j1,j2,J,M_J=ket
    expansion=[]
    for cg in cg_cache[(j1,j2,J)]:
        if cg.M==M_J:
            indice_estado_desacoplado=index.get((j1,cg.m1,j2,cg.m2))
            if indice_estado_desacoplado is not None:
                expansion.append((indice_estado_desacoplado,float(cg.Value)))
    expansiones.append(expansion)

# <beta|H|alpha>
filas=[]
columnas=[]
valores=[]

for alpha in range(M_coupled):

    j1,j2,J,M_J=coupled_basis[alpha]

    if 2*j1+2*j2==N:
        expansion_ket=expansiones[alpha]

        for indice_estado_desacoplado,coeficiente_ket in expansion_ket:
            j1,m1,j2,m2=basis[indice_estado_desacoplado]

            for coeficiente_H,estado_resultado in hamiltonian(j1,m1,j2,m2):
                indice_estado_resultado=index.get(tuple(estado_resultado))
                if indice_estado_resultado is None:
                    continue

                for beta in range(M_coupled):
                    expansion_bra=expansiones[beta]

                    for indice_estado_bra,coeficiente_bra in expansion_bra:
                        if indice_estado_bra==indice_estado_resultado:
                            contribucion=np.conj(coeficiente_bra)*coeficiente_H*coeficiente_ket
                            filas.append(beta)
                            columnas.append(alpha)
                            valores.append(contribucion)

H_coupled=coo_matrix((valores,(filas,columnas)),shape=(M_coupled,M_coupled)).tocsr()

# Clebsch-Gordan transformation matrix
#C=np.zeros((M,M_coupled))

#for alpha in range(M_coupled):
   # for i,coef in expansiones[alpha]:
       # C[i,alpha]=coef

# Compare Hamiltonians
#H_transform=C.T.conj() @ H @ C

#print("Difference between Hamiltonians:",
   #   np.max(np.abs(H_coupled.toarray()-H_transform)))

#print("CG orthogonality:",
 #     np.max(np.abs(C.T.conj() @ C - np.eye(M_coupled))))




print("H shape =",H.shape)
print("H_coupled shape =",H_coupled.shape)


eigenvalues,eigenvectors=eigsh(H,k=1,which='SA') #Eigsh es un metodo de diagonalización, en vez de calcular todos los autovalores y todos los autovectores de la matriz, calcula únicamente los que se necesiten
E0_uncoupled=eigenvalues[0]

eigenvalues_coupled,eigenvectors_coupled=eigsh(H_coupled,k=1,which='SA')
E0_coupled=eigenvalues_coupled[0]

print("Uncoupled energy =",E0_uncoupled)
print("Coupled energy =",E0_coupled)

# Hamiltonian matrix heatmap
plt.figure(figsize=(10,10))
plt.imshow(H_coupled.toarray(), cmap="viridis", interpolation="nearest")
plt.colorbar(label="Matrix element")
plt.xlabel("Ket index")
plt.ylabel("Bra index")
plt.title(f"Coupled basis - 4 sites, N={N}", pad=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig("Hamiltonian_coupled_2Dimers_N4.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
plt.figure(figsize=(10,10))
plt.imshow(H_coupled.toarray(), cmap="viridis", interpolation="nearest")
plt.colorbar(label="Matrix element")
plt.savefig( "CoupledBasis_N2_L2.png", bbox_inches="tight", pad_inches=0.05)