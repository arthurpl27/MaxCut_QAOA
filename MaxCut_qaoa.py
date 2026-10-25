import numpy as np
import networkx as nx
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
import qiskit_aer.noise as noise
from qiskit.quantum_info import Statevector


# ============================================================
#        WEIGHTED RANDOM GRAPH CONSTRUCTION + PLOT
# ============================================================

def random_weighted_graph(N_nodes, max_weight):
    # Maximum possible edges in an undirected simple graph
    max_edges = N_nodes * (N_nodes - 1) // 2

    # Random number of edges
    N_edges = np.random.randint(N_nodes - 1, max_edges + 1)

    # Create and shuffle nodes
    nodes = np.arange(N_nodes)
    np.random.shuffle(nodes)

    edges = []

    for i in range(N_nodes - 1):
        edges.append((nodes[i], nodes[i + 1]))
        N_edges -= 1

    # Add remaining random edges
    edge_set = set(edges)

    while N_edges > 0:
        u = np.random.randint(0, N_nodes)
        v = np.random.randint(0, N_nodes)

        # Simple undirected graph
        if u != v:
            edge = tuple(sorted((u, v)))
            if edge not in edge_set:
                edge_set.add(edge)
                edges.append(edge)
                N_edges -= 1

    # Graph with weights
    G = nx.Graph()
    for u, v in edges:
        weight = np.round(np.random.uniform(0, max_weight), 2)
        G.add_edge(u, v, weight=weight)

    return G

# Plot graph
graph_ref = random_weighted_graph(7, 1)
pos = nx.spring_layout(graph_ref)
nx.draw(graph_ref, pos, with_labels = True, font_weight = 'bold')
labels = nx.get_edge_attributes(graph_ref, 'weight')
nx.draw_networkx_edge_labels(graph_ref, pos, edge_labels = labels)
plt.show()

# ============================================================
#        COST AND MIXER HAMILTONIANS
# ============================================================

def cost_h(graph, gamma):
    cost = QuantumCircuit(len(graph.nodes), name=r"$U_C(\gamma)$")
    for u, v, data in graph.edges(data=True):
        w = data["weight"]
        cost.cx(u, v)
        cost.rz(2 * gamma * w, v)
        cost.cx(u, v)
        cost.barrier()

    return cost


def mixer_h(graph, beta):
    mixer = QuantumCircuit(len(graph.nodes), name=r"$U_M(\beta)$")
    for node in graph.nodes:
        mixer.rx(2 * beta, int(node))

    return mixer

# ============================================================
#        COST UNITARY EXAMPLE CIRCUIT FOR GAMMA = PI/3
# ============================================================

cost_example = cost_h(graph_ref, np.pi/3)
fig, ax = plt.subplots(figsize=(12,8))
cost_example.draw(output='mpl', ax=ax)
plt.show()

# ============================================================
#        MIXER UNITARY EXAMPLE CIRCUIT FOR BETA = PI/3
# ============================================================

mixer_example = mixer_h(graph_ref, np.pi/3)
mixer_example.draw('mpl')
plt.show()

# ============================================================
#        QAOA CIRCUIT CONSTRUCTION
# ============================================================

def qaoa_circuit(graph, params):
    N_layers = int(len(params) / 2)
    gammas = params[:N_layers]  # gamma parameters = first half of parameter list
    betas = params[N_layers:]  # beta parameters = second half of parameter list

    # Initialize Circuit
    qr = QuantumRegister(len(graph.nodes))
    cr = ClassicalRegister(len(graph.nodes))
    qaoa = QuantumCircuit(qr, cr)

    # Initialize qubits with Hadamard gates
    qaoa.h(qr)
    for iteration in range(N_layers):
        # Get Cost and Mixer Unitaries
        cost = cost_h(graph, gammas[iteration])
        mixer = mixer_h(graph, betas[iteration])
        qaoa.append(cost, qr)
        qaoa.append(mixer, qr)

    qaoa.measure(qr, cr)

    return qaoa

# ============================================================
#        QAOA CIRCUIT REFERENCE AT LAYER p=2
# ============================================================

Layer = 2
init_params = [np.random.uniform(0,np.pi) for i in range(2*Layer)]

circuit_ref = qaoa_circuit(graph_ref, init_params)
circuit_ref.draw(output='mpl')
plt.show()

# ============================================================
#        FUNCTION FOR CUT VALUE OF THE GRAPH
# ============================================================

def graph_cut(graph, bit_string):

    # Reverse bitstring
    bit_string = bit_string[::-1]

    cut = 0.0
    weights = nx.get_edge_attributes(graph, 'weight')

    for (u, v), w in weights.items():
        if bit_string[u] != bit_string[v]:
            cut -= w   # negative for minimization

    return cut

# ====================================================================
#        FUNCTION FOR EXPECTATION VALUE OF THE COST HAMILTONIAN
# ====================================================================

def cost_expectation(graph, counts):

    energy = 0.0
    total_shots = 0

    for bit_string, frequency in counts.items():
        energy += frequency * graph_cut(graph, bit_string)
        total_shots += frequency

    return energy / total_shots


# ==========================================================================
#        RUN EXPECTATION VALUE OF COST HAMILTONIAN + MINIMIZATION
# ==========================================================================

shots = 10000
backend = Aer.get_backend("qasm_simulator")

def objective(params):
    # Build circuit
    circuit = qaoa_circuit(graph_ref, params)

    # Run on simulator
    tqc = transpile(circuit, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    return cost_expectation(graph_ref, counts)


# COBYLA - Minimization
solution = minimize(objective, init_params, method="COBYLA", options={"maxiter": 150, "disp": True})

print("Best parameters:", solution.x)
print("Best energy value:", solution.fun)

# ============================================================
#        RUN OPTIMAL CIRCUIT + PLOT BAR CHART
# ============================================================

optimized_params = solution.x
optimal_circuit = qaoa_circuit(graph_ref, optimized_params)

# Run optimal circuit
tqc_opt = transpile(optimal_circuit, backend)
result_opt = backend.run(tqc_opt, shots=shots).result()
counts = result_opt.get_counts()

plt.title("Bit Strings from the QAOA")
plt.bar(counts.keys(), counts.values())
plt.xticks(rotation=45)
plt.show()

# Print counts
for bit_string, frequency in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{bit_string}: {frequency}")

# ============================================================
#                    SOLUTION TO MAX-CUT
# ============================================================

# Maximum occurrence of bitstring
max_string = max(counts, key=counts.get)

# (-1) for a positive maximum cut value
max_cut_value = (-1) * graph_cut(graph_ref, max_string)

print(f"The MaxCut bitstring solution is {max_string} with a cut value of {max_cut_value}")

# ============================================================
#       COLORED GRAPH OF THE SOLUTION
# ============================================================

def graph_color(bit_string):
    colors = []
    for index, bit in enumerate(bit_string):
        if bit == "1":
            colors.append('orange')
        elif bit == "0":
            colors.append('dodgerblue')
    return colors

# Plot colored graph
pos = nx.spring_layout(graph_ref)
nx.draw(graph_ref, pos, with_labels=True, font_weight='bold', node_color=graph_color(max_string))
labels = nx.get_edge_attributes(graph_ref, 'weight')
nx.draw_networkx_edge_labels(graph_ref, pos, edge_labels=labels)
plt.show()

# ============================================================
#                 NOISE MODEL
# ============================================================

layers_list = [1, 2, 3, 4, 5]
p_values = np.linspace(0.0, 0.10, 50)
maxiter_base = 300

Q1_GATES = ["rx", "rz", "x", "sx", "h", "u", "u1", "u2", "u3"]    # 1-qubit gates
Q2_GATES = ["cx"]                                                 # 2-qubit gates

def depolarizing_channel(p):
    nm1 = noise.NoiseModel()
    nm1.add_all_qubit_quantum_error(noise.depolarizing_error(p, 1), Q1_GATES)
    nm1.add_all_qubit_quantum_error(noise.depolarizing_error(p, 2), Q2_GATES)
    return nm1

def dephasing_channel(p):
    nm2 = noise.NoiseModel()
    e1 = noise.phase_damping_error(p)     # 1-qubit dephasing channel
    e2 = e1.tensor(e1)                    # independent dephasing on 2 qubits
    nm2.add_all_qubit_quantum_error(e1, Q1_GATES)
    nm2.add_all_qubit_quantum_error(e2, Q2_GATES)
    return nm2

def bit_flip(p):
    nm3 = noise.NoiseModel()
    e1 = noise.pauli_error([("X", p), ("I", 1 - p)])
    e2 = e1.tensor(e1)                    # independent bit-flip on 2 qubits
    nm3.add_all_qubit_quantum_error(e1, Q1_GATES)
    nm3.add_all_qubit_quantum_error(e2, Q2_GATES)
    return nm3

noises = [("Depolarizing channel", depolarizing_channel), ("Dephasing channel", dephasing_channel),
        ("Bit-flip", bit_flip)]

# ----------------------------
#    NO NOISE AT p=0
# ----------------------------

def objective_exact(params, graph):

    qc = qaoa_circuit(graph, params).remove_final_measurements(inplace=False) # Remove measurements from circuit
    probs = Statevector.from_instruction(qc).probabilities_dict()             # Probabilities
    return sum(prob * graph_cut(graph, bit_string) for bit_string, prob in probs.items())

# -----------------------------------
#    ITERATION OF THE MINIMIZATION
# -----------------------------------

def optimize_layers(graph, layers):

    params_layer = {}
    prev = None

    for L in layers:
        if prev is None:
            x0 = np.random.uniform(0, np.pi, size=2 * L)
        else:
            x0 = np.concatenate([prev, np.array([0.0, 0.0])])

        sol = minimize(lambda x: objective_exact(x, graph), x0, method="COBYLA",
            options={"maxiter": int(maxiter_base * L), "disp": True})

        params_layer[L] = sol.x
        prev = sol.x
        print(f"Layers={L}: max cost = {sol.fun:.6f}")

    return params_layer

params_layer = optimize_layers(graph_ref, layers_list)

# ----------------------------
#   NOISY COST FUNCTION
# ----------------------------

def cost_params_noisy(graph, params, backend, shots=10000):
    qc = qaoa_circuit(graph, params)
    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    return cost_expectation(graph, counts)

# -------------------------------------
#  PLOT FOR EACH NOISY COST FUNCTION
# -------------------------------------

results = {noise_name: {L: [] for L in layers_list} for noise_name, _ in noises}

for noise_name, nm_construction in noises:
    for p in p_values:
        nm = nm_construction(float(p))
        noisy_backend = AerSimulator(noise_model=nm)

        for L in layers_list:
            val = cost_params_noisy(graph_ref, params_layer[L], noisy_backend, shots=shots)
            results[noise_name][L].append(val)

for noise_name, _ in noises:
    plt.figure(figsize=(9, 4.5))
    for L in layers_list:
        plt.plot(p_values, results[noise_name][L], marker="x", label=f"Layers p={L}")
    plt.xlabel(r"Noise parameter $p$")
    plt.ylabel("Cost function")
    plt.title(f"QAOA cost vs noise parameter — {noise_name}")
    plt.grid(True)
    plt.legend()
    plt.show()


