import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Single Layer Perceptron Lab", layout="wide")
sns.set_theme(style="whitegrid")
st.title("Lab Exercise 5 — Single Layer Perceptron")
st.caption("Logic gates, XOR limitation and sample classification datasets")

class Perceptron:
    def __init__(self, lr=0.1, epochs=100, seed=42):
        self.lr, self.epochs, self.seed = lr, epochs, seed
    def step(self, z):
        return (z >= 0).astype(int)
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.w = rng.uniform(-0.5, 0.5, X.shape[1])
        self.b = 0.0
        self.errors, self.w_history, self.b_history = [], [], []
        self.converged = False
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = int(self.step(np.array(np.dot(xi,self.w)+self.b)))
                delta = self.lr*(target-pred)
                if delta:
                    self.w += delta*xi
                    self.b += delta
                    errors += 1
                self.w_history.append(self.w.copy())
                self.b_history.append(self.b)
            self.errors.append(errors)
            if errors == 0:
                self.converged = True
                break
        return self
    def predict(self, X):
        return self.step(np.dot(X,self.w)+self.b)

Xgate = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
gates = {"AND":np.array([0,0,0,1]), "OR":np.array([0,1,1,1]),
         "NAND":np.array([1,1,1,0]), "XOR":np.array([0,1,1,0])}
st.sidebar.header("Training controls")
gate = st.sidebar.selectbox("Logic gate", list(gates))
lr = st.sidebar.slider("Learning rate", .01, .5, .1, .01)
epochs = st.sidebar.slider("Maximum epochs", 10, 300, 100, 10)
seed = st.sidebar.number_input("Random seed", 0, 999, 7)
model = Perceptron(lr, epochs, seed).fit(Xgate, gates[gate])
pred = model.predict(Xgate)
accuracy = np.mean(pred == gates[gate])*100

a,b,c,d = st.columns(4)
a.metric("Training accuracy", f"{accuracy:.1f}%")
b.metric("Epochs", len(model.errors))
c.metric("Converged", "Yes" if model.converged else "No")
d.metric("Misclassified last epoch", model.errors[-1])

tabs = st.tabs(["Truth table", "Decision boundary", "Training dynamics", "Sample datasets", "Concepts"])
with tabs[0]:
    st.subheader(f"{gate} truth table")
    st.dataframe(pd.DataFrame({"Input 1":Xgate[:,0].astype(int),
        "Input 2":Xgate[:,1].astype(int), "Actual":gates[gate],
        "Predicted":pred, "Correct":pred==gates[gate]}), use_container_width=True)
    st.write("Final weights:", model.w, "| Final bias:", model.b)
    if gate == "XOR":
        st.warning("XOR is not linearly separable. A single straight decision boundary cannot classify all four points correctly.")

with tabs[1]:
    xx,yy=np.meshgrid(np.linspace(-.3,1.3,250),np.linspace(-.3,1.3,250))
    zz=model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.contourf(xx,yy,zz,alpha=.25,levels=[-.5,.5,1.5],cmap="coolwarm")
    ax.scatter(Xgate[:,0],Xgate[:,1],c=gates[gate],cmap="coolwarm",
               edgecolor="black",s=150)
    for i,(x,y) in enumerate(Xgate):
        ax.annotate(f"{tuple(Xgate[i].astype(int))}: {gates[gate][i]}",(x,y),
                    xytext=(6,6),textcoords="offset points")
    ax.set(xlabel="Input 1",ylabel="Input 2",title=f"{gate} learned decision regions")
    st.pyplot(fig)

with tabs[2]:
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(range(1,len(model.errors)+1),model.errors,marker="o")
    ax.set(xlabel="Epoch",ylabel="Misclassified samples",
           title=f"{gate} — errors per epoch")
    st.pyplot(fig)
    hist=np.array(model.w_history)
    fig,ax=plt.subplots(figsize=(7,4))
    if len(hist):
        ax.plot(hist[:,0],label="Weight 1")
        if hist.shape[1]>1: ax.plot(hist[:,1],label="Weight 2")
        ax.plot(model.b_history,label="Bias")
    ax.set(xlabel="Training update",ylabel="Parameter value",
           title="Weight and bias update history")
    ax.legend()
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Scholarship eligibility sample")
    sch=pd.DataFrame({
        "Student":["S1","S2","S3","S4","S5","S6","S7","S8"],
        "Internal Assessment":[85,75,65,45,50,30,90,40],
        "Attendance (%)":[90,85,80,60,55,45,95,50],
        "Eligible":[1,1,1,0,0,0,1,0]})
    st.dataframe(sch,use_container_width=True)
    Xs=sch[["Internal Assessment","Attendance (%)"]].to_numpy(float)
    ys=sch["Eligible"].to_numpy(int)
    mean,std=Xs.mean(0),Xs.std(0)
    std[std==0]=1
    Xss=(Xs-mean)/std
    sm=Perceptron(.1,1000,11).fit(Xss,ys)
    sp=sm.predict(Xss)
    st.write(f"Training accuracy: {np.mean(sp==ys)*100:.1f}%")
    st.dataframe(sch.assign(Predicted=sp,Correct=sp==ys),use_container_width=True)
    fig,ax=plt.subplots(figsize=(7,4))
    for label,name in [(0,"Not eligible"),(1,"Eligible")]:
        mask=ys==label
        ax.scatter(Xs[mask,0],Xs[mask,1],label=name,s=90)
    ax.set(xlabel="Internal assessment",ylabel="Attendance (%)",
           title="Scholarship sample — observed classes")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Candidate selection sample")
    cand=pd.DataFrame({
        "Candidate":["C1","C2","C3","C4","C5","C6","C7","C8"],
        "Technical Test":[1,1,0,0,1,0,1,0],
        "Interview":[1,0,1,0,1,0,0,1],
        "Selection":[1,1,1,0,1,0,1,1]})
    st.dataframe(cand,use_container_width=True)
    Xc=cand[["Technical Test","Interview"]].to_numpy(float)
    yc=cand.Selection.to_numpy(int)
    cm=Perceptron(.1,100,13).fit(Xc,yc)
    cp=cm.predict(Xc)
    st.write(f"Training accuracy: {np.mean(cp==yc)*100:.1f}%")
    st.dataframe(cand.assign(Predicted=cp,Correct=cp==yc),use_container_width=True)
    fig,ax=plt.subplots(figsize=(6,4))
    for label,name in [(0,"Not selected"),(1,"Selected")]:
        mask=yc==label
        ax.scatter(Xc[mask,0],Xc[mask,1],label=name,s=100)
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Technical test",
           ylabel="Interview",title="Candidate selection — observed classes")
    ax.legend()
    st.pyplot(fig)

with tabs[4]:
    st.markdown("""
    - **Weights:** determine the contribution of each input.
    - **Bias:** shifts the decision boundary.
    - **Learning rate:** controls the size of each update.
    - **Activation:** binary step function returns 0 or 1.
    - **Linear separability:** whether one straight line can divide the classes.
    - **XOR:** non-linearly separable; an SLP cannot achieve a perfect solution.
    - **MLP:** a hidden layer and nonlinear activation can represent XOR.
    """)
    st.caption("The two sample datasets are small classroom examples. Their displayed accuracy is training accuracy, not a generalisation estimate.")
