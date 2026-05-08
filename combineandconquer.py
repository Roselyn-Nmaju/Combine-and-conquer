"""
=====================
Quantum state preparation utilities implementing a hybrid Time Encoding (TE)
and Space Encoding  (SE) the divide-and-conquer alogrithm,scheme for loading classical data into quantum circuits.

The core algorithm, `CombineandConquer`, decomposes an arbitrary classical data
vector into a parameterised quantum circuit by:
  1. Padding and normalising the input vector.
  2. Extracting amplitude and phase information from complex values.
  3. Generating RY (amplitude) and RZ (phase) rotation angles via a recursive
     binary-tree decomposition.
  4. Grouping rotation angles into sub-circuits controlled by a binary-tree
     indexing structure (the λ parameter governs sub-tree depth).
  5. Stitching sub-circuits together and applying controlled-SWAP (Fredkin)
     gates to route amplitudes into the correct computational basis states.

Dependencies
------------
- numpy
- qiskit
- Python standard library: math, cmath
"""

import numpy as np
import math as math
import cmath as cmath
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate, RZGate


# ---------------------------------------------------------------------------
# Vector pre-processing helpers
# ---------------------------------------------------------------------------

def ModifyVector(vec):
    """Pad a vector to the next power-of-two length by appending zeros.

    Many quantum encoding schemes require the number of amplitudes to equal
    2^n so that they map onto an n-qubit Hilbert space. This function ensures
    that invariant holds without mutating the original list.

    Parameters
    ----------
    vec : list
        Input data vector (real or complex elements).

    Returns
    -------
    list
        A new list whose length is the smallest power of two >= len(vec).
        If len(vec) is already a power of two the list is returned unchanged.

    Example
    -------
    >>> ModifyVector([1, 2, 3])
    [1, 2, 3, 0]
    """
    n = len(vec)
    # bit_length() of (n-1) gives ceil(log2(n)) for n > 1
    power2 = 2 ** (n - 1).bit_length()
    if n < power2:
        vec += [0] * (power2 - n)
    return vec


def NormVector(vec):
    """Normalise a real-valued vector to unit L2 norm.

    The quantum state |ψ⟩ = Σ aᵢ|i⟩ requires Σ |aᵢ|² = 1, so amplitudes
    must be normalised before angle generation.

    Parameters
    ----------
    vec : list of float
        Real-valued amplitude vector.

    Returns
    -------
    list of float
        Unit-normalised copy of *vec*.

    Raises
    ------
    ZeroDivisionError
        If the input vector is the zero vector.
    """
    absvals = [i**2 for i in vec]
    s = sum(absvals)
    sqrtsum = np.sqrt(s)
    out = [i / sqrtsum for i in vec]
    vec = out
    return vec


# ---------------------------------------------------------------------------
# Rotation angle generators
# ---------------------------------------------------------------------------

def GenYAngle(vec, output=None):
    """Recursively generate RY rotation angles for amplitude encoding.

    Implements the angle-computation step of the Möttönen / tree-based state
    preparation algorithm. The binary tree is traversed bottom-up: at each
    level, adjacent pairs of amplitudes are merged into a single parent node
    whose value is their L2 norm, and the angle required to split the parent
    back into its children is stored.

    The recurrence is:
        parent[k] = sqrt(vec[2k]² + vec[2k+1]²)
        angle[k]  = 2 · arcsin(vec[2k+1] / parent[k])

    Parameters
    ----------
    vec : list of float
        Current level of normalised amplitudes (must have even length > 0).
    output : list, optional
        Accumulator for collected angles; initialised to [] on first call.

    Returns
    -------
    list of float
        Flat list of RY rotation angles in radians, ordered from the root
        angle (index 0) down to the leaf angles, matching the order expected
        by `TimeEncoding`.
    """
    if output is None:
        output = []

    if len(vec) > 1:
        # --- Build the parent level by combining adjacent pairs ---
        newx = [0] * int(len(vec) / 2)
        for k in range(len(newx)):
            # L2 norm of the pair → parent amplitude
            eq1 = np.sqrt((abs(vec[2 * k]) ** 2) + (abs(vec[2 * k + 1]) ** 2))
            newx[k] = eq1

        # --- Recurse upward to collect angles for shallower levels first ---
        GenYAngle(newx, output)

        # --- Compute angles at the current level and append ---
        angles = [0] * int(len(vec) / 2)
        for k in range(len(newx)):
            eq2 = (vec[2 * k + 1]) / (newx[k])
            if newx[k] != 0:
                if newx[k] > 0:
                    # Standard case: angle in [–π, π]
                    angles[k] = 2 * np.arcsin(eq2)
                else:
                    # Negative parent amplitude: reflect the angle
                    angles[k] = 2 * math.pi - 2 * np.arcsin(eq2)
            else:
                # Parent amplitude is zero → rotation angle is undefined; use 0
                angles[k] = 0

        output += angles
        return output


def GenZAngle(vec, output=None):
    """Recursively generate RZ rotation angles for phase encoding.

    Mirrors `GenYAngle` but operates on the phase of each amplitude rather
    than its magnitude. At each level, a parent phase is computed as the
    arithmetic mean of adjacent pairs, and the angle needed to recover each
    child's phase from its parent is recorded.

    The recurrence is:
        parent[k] = (vec[2k] + vec[2k+1]) / 2
        angle[k]  = vec[2k+1] - vec[2k]

    Parameters
    ----------
    vec : list of float
        Phase values (in radians) for the current level of the binary tree.
    output : list, optional
        Accumulator for collected angles; initialised to [] on first call.

    Returns
    -------
    list of float
        Flat list of RZ rotation angles in radians, ordered to match the
        qubit ordering used by `TimeEncoding` with ``datatype=1``.
    """
    if output is None:
        output = []

    if len(vec) > 1:
        # --- Build the parent level as the mean of adjacent phases ---
        newx = [0] * int(len(vec) / 2)
        for k in range(len(newx)):
            newx[k] = (vec[2 * k] + vec[2 * k + 1]) / 2

        # --- Recurse upward ---
        GenZAngle(newx, output)

        # --- Compute phase-difference angles at the current level ---
        angles = [0] * int(len(vec) / 2)
        for k in range(len(newx)):
            # Difference encodes how much the right child deviates from parent
            angles[k] = vec[2 * k + 1] - vec[2 * k]

        output += angles
        return output


# ---------------------------------------------------------------------------
# Binary tree data structure and traversal
# ---------------------------------------------------------------------------

class TreeNode:
    """Node in a binary tree used to organise quantum sub-circuit indices.

    Attributes
    ----------
    val : any
        Value stored at this node (typically an integer index).
    left : TreeNode or None
        Left child node.
    right : TreeNode or None
        Right child node.
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def BuildTree(vec):
    """Build a binary tree from a flat array stored in breadth-first order.

    The input follows the standard heap / array-based binary-tree layout:
        - Index 0 is the root.
        - For a node at index i, left child is at 2i+1, right child at 2i+2.
    None entries in *vec* produce absent (None) child nodes.

    Parameters
    ----------
    vec : list
        Flat array of node values in breadth-first (level) order.

    Returns
    -------
    TreeNode or None
        Root of the constructed binary tree, or None if *vec* is empty.
    """
    def CreateNode(index):
        # Base case: index out of bounds or explicit None sentinel
        if index >= len(vec) or vec[index] is None:
            return None

        node = TreeNode(vec[index])
        node.left = CreateNode(2 * index + 1)   # left child
        node.right = CreateNode(2 * index + 2)  # right child
        return node

    return CreateNode(0)


def PreorderTraversal(tree):
    """Return the values of a binary tree in pre-order (root → left → right).

    Pre-order traversal visits the root before any of its subtrees, which is
    the order in which sub-circuits must be composed onto the main circuit.

    Parameters
    ----------
    tree : TreeNode or None
        Root of the binary tree to traverse.

    Returns
    -------
    list
        Values of all nodes in pre-order sequence.
    """
    result = []

    def Traverse(node):
        if node:
            result.append(node.val)   # visit root
            Traverse(node.left)       # traverse left subtree
            Traverse(node.right)      # traverse right subtree

    Traverse(tree)
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def GenBinaryStrings(length):
    """Generate all binary strings of a given length in lexicographic order.

    These strings are used as control-state specifiers for multi-controlled
    rotation gates (e.g., an RY gate that fires only when the control qubits
    are in state '01').

    Parameters
    ----------
    length : int
        Number of bits; produces 2^length strings.

    Returns
    -------
    list of str
        All binary strings of *length* bits, zero-padded, from '00…0' to '11…1'.

    Example
    -------
    >>> GenBinaryStrings(2)
    ['00', '01', '10', '11']
    """
    binary_strings = []
    max_num = 2 ** length

    for i in range(max_num):
        # zfill pads with leading zeros to guarantee uniform length
        binary_string = format(i, 'b').zfill(length)
        binary_strings.append(binary_string)
    return binary_strings


def ReverseString(stringlist):
    """Reverse each string in a list.

    Qiskit uses little-endian qubit ordering (qubit 0 is the least significant
    bit), so control state strings must be reversed before being passed to
    `RYGate.control` or `RZGate.control`.

    Parameters
    ----------
    stringlist : list of str
        List of bit strings to reverse.

    Returns
    -------
    list of str
        New list with each string reversed.

    Example
    -------
    >>> ReverseString(['01', '10'])
    ['10', '01']
    """
    reversed_list = [string[::-1] for string in stringlist]
    return reversed_list


def ComplexToAP(vec):
    """Decompose a list of complex numbers into amplitudes and phases.

    Uses the polar form z = r · e^{iφ} to separate magnitude (r) from
    phase (φ), which are then encoded independently via RY and RZ gates.

    Parameters
    ----------
    vec : list of complex
        Input state vector (may contain real numbers, which are treated as
        complex with zero imaginary part).

    Returns
    -------
    amplitude : list of float
        Magnitudes |z| for each element.
    phase : list of float
        Phases arg(z) ∈ (–π, π] for each element.
    """
    amplitude = []
    phase = []
    for i in range(len(vec)):
        a, p = cmath.polar(vec[i])
        amplitude.append(a)
        phase.append(p)
    return amplitude, phase


# ---------------------------------------------------------------------------
# Time Encoding (TE) circuit builder
# ---------------------------------------------------------------------------

def TimeEncoding(rotations, datatype=0):
    """Build a Time Encoding (TE) sub-circuit from a list of rotation angles.

    A TE circuit implements the angle-based state preparation for a single
    sub-tree of rotation angles. The first angle is applied unconditionally to
    qubit 0; subsequent angles are applied as multi-controlled rotations whose
    control states correspond to all binary strings of increasing length, as
    generated by `GenBinaryStrings`.

    The number of qubits required is ceil(log2(len(rotations))).

    Parameters
    ----------
    rotations : list of float
        Ordered rotation angles (in radians) for this sub-circuit.
        Length must be a power of two.
    datatype : {0, 1}
        Selects the rotation axis:
          - 0 → RY gates  (amplitude encoding)
          - 1 → RZ gates  (phase encoding)

    Returns
    -------
    QuantumCircuit
        Qiskit circuit implementing the TE sub-circuit.
    """
    if datatype == 0:
        # ----------------------------------------------------------------
        # RY-based Time Encoding (amplitude)
        # ----------------------------------------------------------------
        index = 1  # tracks position in the rotations list
        num_qubits = len(rotations).bit_length()  # ceil(log2) qubit count
        circuit = QuantumCircuit(num_qubits)

        # Root rotation: unconditional RY on qubit 0
        circuit.ry(rotations[0], 0)

        # Controlled rotations for each subsequent level of the binary tree
        for i in range(1, num_qubits):
            binaries = GenBinaryStrings(i)
            binaries = ReverseString(binaries)  # account for Qiskit endianness
            qbits = list(range(0, i + 1))       # [ctrl_0, ..., ctrl_{i-1}, target]

            for string in binaries:
                # Create an i-controlled RY gate with the specified control state
                control_y = RYGate(rotations[index]).control(i, None, string)
                circuit.append(control_y, qbits)
                index += 1

    if datatype == 1:
        # ----------------------------------------------------------------
        # RZ-based Time Encoding (phase)
        # ----------------------------------------------------------------
        index = 1
        num_qubits = len(rotations).bit_length()
        circuit = QuantumCircuit(num_qubits)

        # Root rotation: unconditional RZ on qubit 0
        circuit.rz(rotations[0], 0)

        for i in range(1, num_qubits):
            binaries = GenBinaryStrings(i)
            binaries = ReverseString(binaries)
            qbits = list(range(0, i + 1))

            for string in binaries:
                control_z = RZGate(rotations[index]).control(i, None, string)
                circuit.append(control_z, qbits)
                index += 1

    return circuit


# ---------------------------------------------------------------------------
# Sub-circuit grouping helpers
# ---------------------------------------------------------------------------

def TreeSubLists(treelist):
    """Partition a flat binary-tree list into level-by-level sublists.

    Given a flat array in breadth-first order, this function groups elements
    by their depth in the tree:
        Level 0: [root]                      → sublist of size 1
        Level 1: [left, right]               → sublist of size 2
        Level 2: [ll, lr, rl, rr]            → sublist of size 4
        ...

    Parameters
    ----------
    treelist : list
        Flat breadth-first representation of a binary tree.

    Returns
    -------
    list of list
        Each inner list contains the elements at one depth level of the tree.

    Example
    -------
    >>> TreeSubLists([0, 1, 2, 3, 4, 5, 6])
    [[0], [1, 2], [3, 4, 5, 6]]
    """
    result = []
    n = len(treelist)
    power = 0
    i = 0
    while i < n:
        sublist = []
        sublist_size = 2 ** power  # number of nodes at this depth
        for _ in range(sublist_size):
            if i < n:
                sublist.append(treelist[i])
                i += 1
        result.append(sublist)
        power += 1
    return result


def SplitList(index_list, splits):
    """Divide a list into *splits* roughly equal contiguous sub-lists.

    Distributes any remainder elements among the first sub-lists (one extra
    element each) so that sub-list lengths differ by at most 1.

    Parameters
    ----------
    index_list : list
        The list to partition.
    splits : int
        Number of sub-lists to produce. Must satisfy 1 <= splits <= len(index_list).

    Returns
    -------
    list of list or None
        Partitioned sub-lists, or None if *splits* is out of range.
    """
    if splits <= 0 or splits > len(index_list):
        return None

    sublist_size = len(index_list) // splits
    remainder = len(index_list) % splits

    result = []
    start = 0
    for i in range(splits):
        # Give the first `remainder` sublists one extra element
        sublist_end = start + sublist_size + (1 if i < remainder else 0)
        result.append(index_list[start:sublist_end])
        start = sublist_end

    return result


def SubCircuitList(vec, tree_sublist):
    """Group binary-tree rotation indices into per-sub-circuit index lists.

    Given the level-by-level representation of the rotation-index tree and a
    target depth λ (``tree_sublist``), this function collects the indices that
    belong to each sub-circuit.  The sub-tree rooted at depth λ defines the
    granularity of the grouping: all ancestor indices up to the root are
    proportionally distributed across the sub-circuits at the target level.

    Parameters
    ----------
    vec : list of list
        Level-by-level sublists of rotation indices (from `TreeSubLists`),
        listed from the deepest level first (i.e. reversed).
    tree_sublist : int
        1-based depth index (λ) specifying which level defines the sub-circuits.
        Converted internally to 0-based.

    Returns
    -------
    list of list
        Each inner list contains the rotation indices assigned to one
        sub-circuit, ordered to match the TE circuit construction.
    """
    tree_sublist -= 1  # convert to 0-based depth index
    split_size = len(vec[tree_sublist])  # number of sub-circuits at this depth
    output = [[] for _ in range(split_size)]

    # Walk from the target depth up to the root, distributing indices
    for sublist in range(tree_sublist, -1, -1):
        for i in range(split_size - 1, -1, -1):
            split_list = SplitList(vec[sublist], split_size)
            # Assign the i-th split of this level to sub-circuit i
            output[i] += split_list[i]

    return output


def QubitGrouper(vec, indices):
    """Map a flat rotation-index list to grouped sub-circuit index lists.

    Given the pre-order-traversed sub-circuit metadata (each entry being a
    tuple of (index_list, qubit_count)), this function partitions *vec* into
    contiguous chunks whose sizes are determined by the qubit counts.

    Parameters
    ----------
    vec : list
        Flat list of rotation indices to be partitioned.
    indices : list of tuple
        List of (index_list, qubit_count) pairs describing each sub-circuit.
        Only the second element of each tuple (qubit count) is used here.

    Returns
    -------
    list of list
        Grouped rotation indices; the i-th inner list contains the indices
        belonging to sub-circuit i.
    """
    result = []
    n = 0  # running offset into vec

    for i in range(len(indices)):
        sublist = []
        sublist_size = indices[i][1]  # number of rotations in this sub-circuit
        for m in range(sublist_size):
            sublist.append(vec[n + m])
        result.append(sublist)
        n += sublist_size

    return result


# ---------------------------------------------------------------------------
# Main entry point: combined TE + SE circuit construction
# ---------------------------------------------------------------------------

def CombineandConquer(input_vec, λ=1, complex=False):
    """Build a full quantum state-preparation circuit using TE + SE encoding.

    This is the top-level routine that assembles the complete circuit for
    loading a classical data vector into a quantum state.  It combines:

    * **Time Encoding (TE)**: Each sub-circuit uses a tree of controlled RY
      (and optionally RZ) rotations to encode amplitudes (and phases) within
      a register of log₂(sub-circuit size) qubits.

    * **Space Encoding (SE)**: Controlled-SWAP (Fredkin) gates route the
      prepared amplitudes from the sub-circuit registers into the correct
      computational basis states of the full Hilbert space.

    The parameter λ controls the trade-off between circuit width and depth:
      - λ = 1 → one large sub-circuit per tree leaf (deeper, fewer qubits).
      - λ > 1 → more sub-circuits of smaller depth (shallower, more qubits).

    Parameters
    ----------
    input_vec : list of float or complex
        Classical data vector to encode.  Need not be normalised or a
        power-of-two length — both are handled internally.
    λ : int, optional
        Sub-tree depth parameter governing sub-circuit granularity. Default 1.
    complex : bool, optional
        If True, also encode phases using RZ gates (complex state preparation).
        If False (default), only amplitudes are encoded (real state preparation).

    Returns
    -------
    QuantumCircuit
        Qiskit circuit that prepares the quantum state |ψ⟩ corresponding to
        the (normalised) input vector.

    Notes
    -----
    The total qubit count is determined by the sum of qubit counts across all
    sub-circuits, which depends on both len(input_vec) and λ.
    """
    # ------------------------------------------------------------------
    # Step 1: Pre-process the input vector
    # ------------------------------------------------------------------
    # Pad to power-of-two length and decompose into amplitudes + phases
    mod_input = ModifyVector(input_vec)
    amplitude_list, phase_list = ComplexToAP(mod_input)

    # Generate rotation angles for amplitude (RY) and phase (RZ) encoding
    y_rotations = GenYAngle(NormVector(amplitude_list))
    z_rotations = GenZAngle(phase_list)

    # ------------------------------------------------------------------
    # Step 2: Build the binary-tree index structure
    # ------------------------------------------------------------------
    vec = list(range(len(y_rotations)))   # flat index list [0, 1, ..., N-1]
    vec_len = len(vec)
    vec_sublists = TreeSubLists(vec)      # level-by-level partition
    vec_sublists.reverse()                # deepest level first for SubCircuitList

    # Compute how many sub-circuits (tree leaves) there are at depth λ
    number_subtree = int((vec_len + 1) / (2 ** λ))
    number_subtree_qubit = (2 ** λ) - 1

    # Build the list of (index_list, qubit_count) pairs for each sub-circuit:
    #   - Singleton entries [[i]] for leaf rotations that don't span a full sub-tree
    #   - SubCircuitList output for the remaining grouped rotations
    tree_list = [[i] for i in range(vec_len - (number_subtree * number_subtree_qubit))]
    tree_list += SubCircuitList(vec_sublists, λ)

    # Create a dict: sub-circuit_id → (rotation_indices, qubit_count)
    tree_dict = {
        i: (tree_list[i], int(math.log(len(tree_list[i]) + 1, 2)))
        for i in range(len(tree_list))
    }

    # Determine the pre-order traversal ordering of sub-circuit IDs
    tree_indices = PreorderTraversal(BuildTree(list(range(len(tree_dict)))))
    # Reorder the sub-circuit metadata to match pre-order circuit composition
    ordered_list = [tree_dict[key] for key in tree_indices]

    # ------------------------------------------------------------------
    # Step 3: Create the blank combined circuit
    # ------------------------------------------------------------------
    num_qubits = sum([tree_dict[i][1] for i in tree_dict])
    circuit = QuantumCircuit(num_qubits)

    # ------------------------------------------------------------------
    # Step 4: Map rotation angles onto ordered sub-circuit metadata
    # ------------------------------------------------------------------
    # For each sub-circuit, collect the actual angle values (not just indices)
    ordered_list_y = [
        ([y_rotations[x] for x in tpl[0]], tpl[1]) for tpl in ordered_list
    ]
    ordered_list_z = [
        ([z_rotations[x] for x in tpl[0]], tpl[1]) for tpl in ordered_list
    ]

    # ------------------------------------------------------------------
    # Step 5: Compose RY (amplitude) Time Encoding sub-circuits
    # ------------------------------------------------------------------
    index = 0
    for unitary in range(len(ordered_list_y)):
        sub_circuit = TimeEncoding(ordered_list_y[unitary][0], 0)
        # Place sub-circuit on the next block of qubits in the full circuit
        circuit = circuit.compose(
            sub_circuit,
            list(range(index, index + ordered_list_y[unitary][1]))
        )
        index += ordered_list_y[unitary][1]

    # ------------------------------------------------------------------
    # Step 6 (optional): Compose RZ (phase) Time Encoding sub-circuits
    # ------------------------------------------------------------------
    if complex:
        index = 0
        for unitary in range(len(ordered_list_z)):
            sub_circuit = TimeEncoding(ordered_list_z[unitary][0], 1)
            circuit = circuit.compose(
                sub_circuit,
                list(range(index, index + ordered_list_z[unitary][1]))
            )
            index += ordered_list_z[unitary][1]

    # ------------------------------------------------------------------
    # Step 7: Determine SWAP scheduling via Space Encoding (SE)
    # ------------------------------------------------------------------
    # grouping_list[i] contains the rotation indices assigned to sub-circuit i
    grouping_list = QubitGrouper(vec, ordered_list)

    # Determine the pre-order traversal of sub-circuit assignments
    ordering = PreorderTraversal(BuildTree(list(range(len(grouping_list)))))
    traversed_dict = {ordering[i]: grouping_list[i] for i in range(len(grouping_list))}

    # Reconstruct level-by-level SWAP scheduling list
    swap_list = [traversed_dict[key] for key in range(len(grouping_list))]
    swap_list = TreeSubLists(swap_list)
    swap_list.reverse()  # deepest level first for iterating SWAP stages

    # ------------------------------------------------------------------
    # Step 8: Apply controlled-SWAP (Fredkin) gates for Space Encoding
    # ------------------------------------------------------------------
    # m = total number of levels in the sub-circuit binary tree
    m = (len(grouping_list)).bit_length()

    for level in range(1, m):
        circuit.barrier()  # visual separator between SE levels

        # Each SE level consists of `level` stages of pairwise swaps
        for stage in range(1, level + 1):
            for k in range(len(swap_list[level])):
                # control qubit index (first qubit of the controlling sub-circuit)
                control = swap_list[level][k]

                # The two sub-circuit qubit groups to be conditionally swapped
                qubit_1 = swap_list[level - stage][(2 ** stage) * k]
                qubit_2 = swap_list[level - stage][(2 ** stage) * k + (2 ** (stage - 1))]

                # Apply a CSWAP for each qubit position within the sub-circuits
                for n in range(len(qubit_1)):
                    circuit.cswap(control[0], qubit_1[n], qubit_2[n])

    circuit.barrier()  # final barrier to delimit the SE section

    return circuit