
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from ClebschGordan_4Sitios import ClebschGordan


N=11
t1=1.0
t2s=np.linspace(0,0.2,100) 
U=2.0
js=np.arange(0,N/2+0.5,0.5)

E0_unc_lista=[]
E0_cou_lista=[]

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

M=len(basis)
index={tuple(ket):i for i,ket in enumerate(basis)}


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


# Cycle over t2
for t2 in t2s:

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

    H_coupled=coo_matrix( (valores,(filas,columnas)),shape=(M_coupled,M_coupled)  ).tocsr()

   

    #print("H shape =",H.shape)
    #print("H_coupled shape =",H_coupled.shape)


    eigenvalues,eigenvectors=eigsh(H,k=1,which='SA')
    E0_uncoupled=eigenvalues[0]

    eigenvalues_coupled,eigenvectors_coupled=eigsh(H_coupled,k=1,which='SA')
    E0_coupled=eigenvalues_coupled[0]

    E0_unc_lista.append(E0_uncoupled)
    E0_cou_lista.append(E0_coupled)

    #print("t2 =",t2)
    #print("Uncoupled energy =",E0_uncoupled)
    #print("Coupled energy =",E0_coupled)


# Ground-state energy vs t2
plt.figure(figsize=(8,6))
plt.plot(t2s,E0_unc_lista,label="Uncoupled basis")
plt.plot(t2s,E0_cou_lista,"--",label="Coupled basis")
plt.xlabel(r"$t_2$")
plt.ylabel(r"$E_0$")
plt.title(f"Ground-state energy vs t2, N={N}, t1={t1}, U={U}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


